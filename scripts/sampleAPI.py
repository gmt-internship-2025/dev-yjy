# sampleAPI.py — Headless GRU inference (no GUI), numeric prints, low-latency, no cooldown
import os, sys, time, json
import numpy as np
import cv2
import mediapipe as mp
from collections import deque, Counter
from tensorflow.keras.models import load_model
import joblib

# ============ Paths ============
MODEL_DIR = "../model"
DATASET_DIR = "../dataset"
CANDIDATE_MODELS = [
    os.path.join(MODEL_DIR, "gesture_gru_best.keras"),
    os.path.join(MODEL_DIR, "gesture_gru_final.keras"),
    os.path.join(MODEL_DIR, "gesture_gru_model.h5"),
]
LABEL_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
CFG_PATHS = [
    os.path.join(DATASET_DIR, "dataset_config.json"),
    os.path.join("../data", "dataset_config.json"),
]

# ============ Runtime params (tunable) ============
# 추론/안정화
P_THRESH       = float(os.environ.get("P_THRESH", "0.55"))   # 기본 신뢰 임계값
STABLE_WINDOW  = int(os.environ.get("STABLE_WINDOW", "3"))   # 다수결 창 크기
STABLE_MIN     = int(os.environ.get("STABLE_MIN",  "2"))     # 다수결 최소 득표
FAST_P         = float(os.environ.get("FAST_P", "0.85"))     # 고확신 임계값(패스트패스)
FAST_HITS      = int(os.environ.get("FAST_HITS", "2"))       # 같은 라벨 연속 히트 수(패스트패스)

# 카메라/리사이즈 (Hands 성능과 속도 균형)
CAM_INDEX      = int(os.environ.get("CAM_INDEX", "0"))
CAP_W          = int(os.environ.get("CAP_W", "640"))
CAP_H          = int(os.environ.get("CAP_H", "480"))
CAP_FPS        = int(os.environ.get("CAP_FPS", "30"))
INFER_W        = int(os.environ.get("INFER_W", "256"))
INFER_H        = int(os.environ.get("INFER_H", "192"))

# ============ Label → Code mapping ============
LABEL2CODE = {
    "start": 0,
    "stop": 1,
    "swipe_right_to_left": 2,  # 좌 스와이프
    "swipe_left_to_right": 3,  # 우 스와이프
    "scroll_up": 4,
    "scroll_down": 5,
}

# freeze/freezing 별칭(출력 대상 아님, 정규화만)
ALIASES = {
    "freeze_left": "freezing_left",
    "freeze_right": "freezing_right",
    "freeze_down": "freezing_down",
    "freezing_left": "freezing_left",
    "freezing_right": "freezing_right",
    "freezing_down": "freezing_down",
}
def norm_label(lbl: str) -> str:
    return ALIASES.get(lbl, lbl)

# ============ Geometry / Preprocess ============
WRIST = 0
INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 5, 9, 17
EPS = 1e-6
USE_Z = False          # xy만 사용
USE_STANDARDIZE = True

def _pick_origin_and_scale(coords: np.ndarray):
    origin = coords[WRIST]
    scale = float(np.linalg.norm(coords[INDEX_MCP] - coords[PINKY_MCP]))
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(np.linalg.norm(coords[WRIST] - coords[MIDDLE_MCP]))
    if not np.isfinite(scale) or scale <= EPS:
        diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
        scale = diag if diag > EPS else 1.0
    return origin, scale

def preprocess_vec(hlm) -> np.ndarray | None:
    lm = hlm.landmark
    if USE_Z:
        curr = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)  # (21,3)
        D = 3
    else:
        curr = np.array([[l.x, l.y] for l in lm], dtype=np.float32)       # (21,2)
        D = 2
    origin, scale = _pick_origin_and_scale(curr)
    if scale <= 1e-3:
        return None
    s = scale if scale > EPS else 1.0
    coords = (curr - origin) / s
    if USE_STANDARDIZE:
        mean = coords.mean(axis=0, keepdims=True)
        std  = coords.std(axis=0, keepdims=True) + 1e-6
        coords = (coords - mean) / std
    return coords.flatten()  # (21*D,)

def adapt_feature(vec_1h: np.ndarray, expect_f: int) -> np.ndarray:
    """(1 hand) vec → 모델 입력 차원에 맞춤. expect_f가 2배면 뒤에 0패딩."""
    f = vec_1h.shape[0]  # 42 or 63
    if f == expect_f:
        return vec_1h.astype(np.float32)
    if expect_f == f * 2:
        pad = np.zeros_like(vec_1h)
        return np.concatenate([vec_1h, pad], axis=0).astype(np.float32)
    # 그 외는 지원 안 함(학습/전처리 포맷 불일치)
    raise ValueError(f"Cannot adapt features: have {f}, expect {expect_f}")

# ============ Model / Config load ============
model_path = next((p for p in CANDIDATE_MODELS if os.path.exists(p)), None)
if not model_path:
    sys.stderr.write("Model file not found.\n")
    sys.exit(1)
try:
    model = load_model(model_path)
except Exception:
    sys.stderr.write("Failed to load model.\n")
    sys.exit(1)

try:
    le = joblib.load(LABEL_PATH)
except Exception:
    sys.stderr.write("Label encoder not found.\n")
    sys.exit(1)

# 입력 (None, T, F) 추출
if hasattr(model, "input_shape") and isinstance(model.input_shape, tuple) and len(model.input_shape) == 3:
    SEQ_LEN = int(model.input_shape[1])
    FEAT_DIM_EXPECT = int(model.input_shape[2])
else:
    SEQ_LEN, FEAT_DIM_EXPECT = 10, 42
    for p in CFG_PATHS:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            SEQ_LEN = int(cfg.get("sequence_length", SEQ_LEN))
            FEAT_DIM_EXPECT = int(cfg.get("feature_dim", FEAT_DIM_EXPECT))
            break

# 모델-라벨 일치 체크
num_units = model.output_shape[-1]
if num_units != len(le.classes_):
    sys.stderr.write("LabelEncoder and model outputs mismatch.\n")
    sys.exit(1)

# 시퀀스 버퍼
seq = np.zeros((SEQ_LEN, FEAT_DIM_EXPECT), dtype=np.float32)
seq_ptr = 0
filled = False

# 안정화(다수결) & 패스트패스용 상태
pred_buf = deque(maxlen=STABLE_WINDOW)
last_lbl_for_fast = None
fast_hits = 0

# ============ Non-blocking stdin for quit ============
def read_quit_nonblocking():
    try:
        if os.name == "nt":
            import msvcrt
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ('q', 'Q', '\x1b'):
                    return True
        else:
            import select
            dr, _, _ = select.select([sys.stdin], [], [], 0)
            if dr:
                line = sys.stdin.readline().strip()
                if line.lower() == 'q':
                    return True
    except Exception:
        pass
    return False

# ============ Camera open (low-latency) ============
def open_camera(idx: int):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            return None
    # 최신 프레임만 사용(드라이버/백엔드에 따라 무시될 수 있음)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    return cap

# ============ MediaPipe Hands ============
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,          # 0이 가장 빠름
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============ Ready ============
print("Models Loaded")  # 요구사항

cap = open_camera(CAM_INDEX)
if not cap:
    sys.stderr.write("Cannot open camera.\n")
    sys.exit(1)

# ============ Main loop ============
try:
    while True:
        if read_quit_nonblocking():
            print("Exit motion mode")
            break

        ret, frame = cap.read()
        if not ret:
            # 드문 읽기 실패 — 다음 루프로
            continue

        frame = cv2.flip(frame, 1)  # 좌우 반전(사용자 관점 일치)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_small = cv2.resize(rgb, (INFER_W, INFER_H))
        rgb_small.flags.writeable = False
        result = hands.process(rgb_small)
        rgb_small.flags.writeable = True

        if not result.multi_hand_landmarks:
            # 필요시 과거 프레임 버리기(지연 축적 방지)
            # cap.grab()
            continue

        hlm = result.multi_hand_landmarks[0]
        v = preprocess_vec(hlm)
        if v is None:
            continue

        try:
            feat = adapt_feature(v, FEAT_DIM_EXPECT)  # (F_expect,)
        except ValueError:
            # 학습 포맷과 수집 포맷이 다르면 여기로 옴
            continue

        # 순환 버퍼 적재
        seq[seq_ptr, :] = feat
        seq_ptr = (seq_ptr + 1) % SEQ_LEN
        if seq_ptr == 0:
            filled = True
        if not filled:
            continue

        # 모델 추론
        x = np.expand_dims(np.roll(seq, -seq_ptr, axis=0), axis=0)  # (1,T,F)
        prob = model(x, training=False).numpy()[0]                   # (C,)
        pmax = float(prob.max())
        idx  = int(prob.argmax())
        lbl  = norm_label(str(le.classes_[idx]))

        # ===== 패스트패스: 고확신이 연속 FAST_HITS 회 ====
        if pmax >= FAST_P:
            if lbl == last_lbl_for_fast:
                fast_hits += 1
            else:
                last_lbl_for_fast = lbl
                fast_hits = 1
            if fast_hits >= FAST_HITS and lbl in LABEL2CODE:
                print(LABEL2CODE[lbl], flush=True)
                # 빠르게 한번 쐈으면 다음 감지를 위해 상태 리셋
                pred_buf.clear()
                fast_hits = 0
                last_lbl_for_fast = None
                continue
        else:
            # 고확신이 끊기면 패스트 상태 축소
            fast_hits = 0
            last_lbl_for_fast = lbl

        # ===== 일반 게이팅 + 다수결 안정화 ====
        pred_buf.append(lbl if pmax >= P_THRESH else "_")
        counts = Counter([z for z in pred_buf if z != "_"])
        if not counts:
            continue
        stable_lbl, stable_cnt = counts.most_common(1)[0]

        if stable_cnt >= STABLE_MIN and stable_lbl in LABEL2CODE:
            print(LABEL2CODE[stable_lbl], flush=True)
            # 다음 감지를 위해 버퍼/패스트 상태 리셋
            pred_buf.clear()
            fast_hits = 0
            last_lbl_for_fast = None

finally:
    cap.release()
