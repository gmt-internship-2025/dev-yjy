# live_inference_gru.py — MediaPipe HANDS (좌우 뒤바뀜 수정 통합본)
# 원본 좌표계로 추론, 시각화(표시)만 좌우 미러링

import os, time, json, warnings, subprocess
import numpy as np
import cv2
import mediapipe as mp
from collections import deque, Counter
from tensorflow.keras.models import load_model
import joblib
from mediapipe.framework.formats import landmark_pb2  # pb2 drawing

# ===================== GUI 사용 감지 =====================
def detect_gui_allowed() -> bool:
    if os.environ.get("NO_GUI", "0") == "1":
        return False
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return False
    if not os.environ.get("DISPLAY"):
        return False
    return True

USE_GUI = detect_gui_allowed()
if not USE_GUI:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # Qt/xcb 크래시 방지

# ===================== 경로/고정 설정 =====================
DATASET_DIR = "../dataset"          # config는 dataset에 저장
MODEL_DIR   = "../model"
CONFIG_PATH = os.path.join(DATASET_DIR, "dataset_config.json")

# LabelEncoder는 model 우선, 없으면 dataset 폴더에서 폴백
LABEL_PATH_CANDIDATES = [
    os.path.join(MODEL_DIR, "label_encoder.pkl"),
    os.path.join(DATASET_DIR, "label_encoder.pkl"),
]

CANDIDATE_MODELS = [
    os.path.join(MODEL_DIR, "gesture_gru_best.keras"),
    os.path.join(MODEL_DIR, "gesture_gru_final.keras"),
    os.path.join(MODEL_DIR, "gesture_gru_model.h5"),
]

# 추론 안정화/출력 파라미터
P_THRESH       = float(os.environ.get("P_THRESH",       "0.50"))
STABLE_WINDOW  = int(os.environ.get("STABLE_WINDOW",    "4"))
STABLE_MIN     = int(os.environ.get("STABLE_MIN",       "4"))
COOLDOWN_SEC   = float(os.environ.get("COOLDOWN_SEC",   "0.8"))
PRINT_INTERVAL = float(os.environ.get("PRINT_INTERVAL", "0.5"))
HAND_TARGET    = os.environ.get("HAND_TARGET", "ANY").upper()  # "RIGHT" | "LEFT" | "ANY"

# ===================== 도우미: 차원/레이아웃 추론 =====================
def infer_D_and_blocks(F: int):
    if F % (21 * 3) == 0:
        D = 3; blocks = F // (21 * D)
    elif F % (21 * 2) == 0:
        D = 2; blocks = F // (21 * D)
    else:
        raise ValueError(f"FEATURE_DIM={F} 에서 D/blocks 추론 실패 (Hands 전처리 아님).")
    if blocks not in (1, 2):
        raise ValueError(f"지원 blocks={blocks} (허용 1 or 2)")
    return D, blocks

def infer_layout(cfg, F, D, blocks):
    layout = cfg.get("layout", None)
    if layout in ("two_hands_xy", "coords", "coords+delta"):
        return layout
    use_delta = cfg.get("use_delta", None)
    if use_delta is True:
        return "coords+delta"
    if F in (84, 126) and cfg.get("use_z", False) is False:
        note = (cfg.get("note", "") or "").lower()
        if "two-hands" in note or "two hands" in note or cfg.get("feature_dim_per_block", None) == 42:
            return "two_hands_xy"
    return "coords" if blocks == 1 else "coords+delta"

# ===================== dataset_config 로드 =====================
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    SEQUENCE_LENGTH = int(cfg.get("sequence_length", 10))
    FEATURE_DIM     = int(cfg.get("feature_dim", 42))
    USE_STANDARDIZE = bool(cfg.get("use_standardize", True))
    modality        = cfg.get("modality", "hands")
    D = 3 if cfg.get("use_z", None) is True else (2 if cfg.get("use_z", None) is False else None)
    blocks = int(cfg.get("num_blocks", 0)) if "num_blocks" in cfg else 0
    if D is None or blocks not in (1, 2):
        D2, blocks2 = infer_D_and_blocks(FEATURE_DIM)
        D = D if D is not None else D2
        blocks = blocks if blocks in (1, 2) else blocks2
    MASKED_LANDMARKS = list(cfg.get("masked_landmarks", []))
    MASK_VALUE       = float(cfg.get("mask_value", 0.0))
else:
    warnings.warn("[경고] dataset_config.json 없음(../dataset). 기본값 사용 : F=42 로 가정되며 모델과 불일치할 수 있음.")
    SEQUENCE_LENGTH = 10
    FEATURE_DIM     = 42
    USE_STANDARDIZE = True
    modality        = "hands"
    D, blocks       = infer_D_and_blocks(FEATURE_DIM)
    MASKED_LANDMARKS = []
    MASK_VALUE       = 0.0

LAYOUT = infer_layout(cfg if 'cfg' in locals() else {}, FEATURE_DIM, D, blocks)
USE_Z = (D == 3)

if modality != "hands":
    warnings.warn(f"[경고] config.modality={modality} → Hands 전용 스크립트와 불일치")
if FEATURE_DIM not in (42, 63, 84, 126):
    warnings.warn(f"[경고] FEATURE_DIM={FEATURE_DIM} (허용: 42/63/84/126)")

print(f"[INFO] T={SEQUENCE_LENGTH}, F={FEATURE_DIM}, D={'xyz' if USE_Z else 'xy'}, "
      f"layout={LAYOUT}, standardize={USE_STANDARDIZE}, hand_target={HAND_TARGET}, GUI={'ON' if USE_GUI else 'OFF'}")
if MASKED_LANDMARKS:
    print(f"[INFO] 마스킹 랜드마크: {MASKED_LANDMARKS} (value={MASK_VALUE})")

# ===================== 모델/라벨 로드 & 검증 =====================
model_path = next((p for p in CANDIDATE_MODELS if os.path.exists(p)), None)
if model_path is None:
    raise FileNotFoundError("모델 파일을 찾을 수 없습니다:\n" + "\n".join(" - "+p for p in CANDIDATE_MODELS))
model = load_model(model_path)

label_path = next((p for p in LABEL_PATH_CANDIDATES if os.path.exists(p)), None)
if label_path is None:
    raise FileNotFoundError("label_encoder.pkl을 찾을 수 없습니다:\n" + "\n".join(" - "+p for p in LABEL_PATH_CANDIDATES))
le = joblib.load(label_path)

num_units = model.output_shape[-1]
num_labels = len(le.classes_)
print(f"[INFO] Loaded model: {model_path}")
print(f"[INFO] Loaded LabelEncoder: {label_path}")
print(f"[INFO] model units={num_units}, label classes={num_labels} -> {le.classes_.tolist()}")
if num_units != num_labels:
    raise RuntimeError("모델과 라벨 인코더 불일치. 같은 학습 세션 페어 사용 필요")

# ========== 라벨 정규화: freeze_* ↔ freezing_* 혼용 대비 ==========
LABEL_ALIASES = {
    "freeze_left": "freezing_left",
    "freeze_right": "freezing_right",
    "freeze_down": "freezing_down",
    "freezing_left": "freezing_left",
    "freezing_right": "freezing_right",
    "freezing_down": "freezing_down",
}
MODEL_LABELS = set(map(str, le.classes_))
def map_to_model_label(lbl: str) -> str:
    n = LABEL_ALIASES.get(lbl, lbl)
    return n if n in MODEL_LABELS else lbl

# ===================== 입력 소스 자동 열기 =====================
VIDEO_SRC = os.environ.get("VIDEO_SRC")
CAM_INDEX = os.environ.get("CAM_INDEX")

def is_jetson():
    try:
        with open("/proc/device-tree/model") as f:
            m = f.read()
        return ("NVIDIA" in m) or ("Jetson" in m)
    except Exception:
        return False

def open_capture():
    if VIDEO_SRC:
        cap = cv2.VideoCapture(VIDEO_SRC)
        if cap.isOpened():
            print(f"[INFO] Opened VIDEO_SRC: {VIDEO_SRC}")
            return cap
        print(f"[WARN] Failed to open VIDEO_SRC={VIDEO_SRC}")
    if is_jetson():
        pipe = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
            "video/x-raw,format=BGR ! appsink"
        )
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("[INFO] Opened Jetson Argus camera via GStreamer")
            return cap
        print("[WARN] Jetson Argus pipeline failed")
    if CAM_INDEX is not None:
        try:
            idx = int(CAM_INDEX)
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"[INFO] Camera opened at index {idx}")
                return cap
            cap.release()
            print(f"[WARN] Failed to open camera index {idx}")
        except Exception:
            print(f"[WARN] Invalid CAM_INDEX={CAM_INDEX}")
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"[INFO] Camera opened at index {i}")
            return cap
        cap.release()
    return None

cap = open_capture()
if cap is None:
    print("[ERROR] 입력 소스를 열 수 없습니다.")
    raise SystemExit

# ===================== MediaPipe Hands 준비 =====================
mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,          # 감지 약하면 1
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===================== 전역 파라미터 =====================
INFER_W, INFER_H = 320, 240
LANDMARK_DRAW_EVERY = 1
LM_SMOOTH_ALPHA = 0.4
LM_HOLD_FRAMES  = 6

# 버퍼
SEQUENCE_LENGTH = int(SEQUENCE_LENGTH)
FEATURE_DIM     = int(FEATURE_DIM)
seq_arr   = np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
seq_ptr   = 0
seq_filled= False
pred_buf  = deque(maxlen=STABLE_WINDOW)

# 모델 워밍업
dummy = np.zeros((1, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
_ = model(dummy, training=False)

print("🎥 실시간 GRU 제스처 인식 시작 (ESC 종료 | 헤드리스=Ctrl+C)")

# ===================== 전처리/벡터화 유틸 =====================
WRIST = 0
INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 5, 9, 17
EPS = 1e-6

def _pick_origin_and_scale(coords):
    origin = coords[WRIST]
    scale = float(np.linalg.norm(coords[INDEX_MCP] - coords[PINKY_MCP]))
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(np.linalg.norm(coords[WRIST] - coords[MIDDLE_MCP]))
    if not np.isfinite(scale) or scale <= EPS:
        diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
        scale = diag if diag > EPS else 1.0
    return origin, scale

def _standardize(a):
    mean = a.mean(axis=0, keepdims=True)
    std  = a.std(axis=0, keepdims=True) + 1e-6
    return (a - mean) / std

def _apply_mask_vec(vec, F, layout, masked_idx, mask_value=0.0, Dloc=2):
    if not masked_idx:
        return vec
    v = vec.copy()
    if layout == "two_hands_xy":
        for block in (0, 42):
            for lm in masked_idx:
                if 0 <= lm < 21:
                    s = block + lm * 2
                    v[s:s+2] = mask_value
    elif layout == "coords":
        for lm in masked_idx:
            if 0 <= lm < 21:
                s = lm * Dloc
                v[s:s+Dloc] = mask_value
    elif layout == "coords+delta":
        stride = 21 * Dloc
        for lm in masked_idx:
            if 0 <= lm < 21:
                s = lm * Dloc
                v[s:s+Dloc] = mask_value
                v[stride + s: stride + s + Dloc] = mask_value
    return v

def vectorize_hand(hlm, prev_coords_raw=None):
    # 1) 원시 좌표 (21,Dloc)
    if USE_Z:
        curr = np.array([[l.x, l.y, l.z] for l in hlm.landmark], dtype=np.float32); Dloc=3
    else:
        curr = np.array([[l.x, l.y]       for l in hlm.landmark], dtype=np.float32); Dloc=2

    # 2) 정규화(원점/스케일)
    origin, scale = _pick_origin_and_scale(curr)
    s = scale if scale > EPS else 1.0
    coords = (curr - origin) / s

    # 3) 표준화(옵션)
    if USE_STANDARDIZE:
        coords = _standardize(coords)

    # 4) 레이아웃 조립
    if LAYOUT == "coords+delta":
        if prev_coords_raw is not None and prev_coords_raw.shape == curr.shape:
            delta = (curr - prev_coords_raw) / s
        else:
            delta = np.zeros_like(coords)
        if USE_STANDARDIZE:
            delta = _standardize(delta)
        vec = np.concatenate([coords.flatten(), delta.flatten()], axis=0)  # (21*Dloc*2)
    elif LAYOUT == "two_hands_xy":
        if Dloc == 3:
            coords = coords[:, :2]
            Dloc = 2
        v42 = coords.flatten()  # (21*2)
        pad = np.zeros(42, dtype=np.float32)
        vec = np.concatenate([v42, pad], axis=0)  # (84)
    else:  # "coords"
        vec = coords.flatten()  # (21*Dloc)

    # 5) 마스킹 적용
    vec = _apply_mask_vec(vec, vec.shape[0], LAYOUT, MASKED_LANDMARKS, MASK_VALUE, Dloc=Dloc)

    # 6) 최종 차원 검증
    if vec.shape[0] != FEATURE_DIM:
        raise ValueError(f"전처리 결과 차원 불일치: {vec.shape[0]} vs FEATURE_DIM={FEATURE_DIM} (layout={LAYOUT})")

    return vec.astype(np.float32), curr  # curr은 델타/스무딩용

# ===== 랜드마크 스무딩/홀드 =====
def lm_to_np(hlm):
    if USE_Z:
        return np.array([[l.x, l.y, l.z] for l in hlm.landmark], dtype=np.float32)
    else:
        return np.array([[l.x, l.y] for l in hlm.landmark], dtype=np.float32)

def smooth_np(curr, prev, alpha):
    if prev is None: return curr.copy()
    return alpha * curr + (1.0 - alpha) * prev

def np_to_pb2(arr):
    arr = arr.copy()
    arr[:, 0:2] = np.clip(arr[:, 0:2], 0.0, 1.0)
    lms = []
    if arr.shape[1] == 3:
        for x, y, z in arr:
            lms.append(landmark_pb2.NormalizedLandmark(x=float(x), y=float(y), z=float(z)))
    else:
        for x, y in arr:
            lms.append(landmark_pb2.NormalizedLandmark(x=float(x), y=float(y), z=0.0))
    return landmark_pb2.NormalizedLandmarkList(landmark=lms)

def draw_landmarks_smooth(frame, np_coords):
    lmlist = np_to_pb2(np_coords)
    mp_draw.draw_landmarks(
        frame, lmlist, mp_hands.HAND_CONNECTIONS,
        mp_styles.get_default_hand_landmarks_style(),
        mp_styles.get_default_hand_connections_style()
    )

def choose_hand(result, prefer_side=None):
    if not result.multi_hand_landmarks:
        return None, None
    pairs = list(zip(result.multi_hand_landmarks, result.multi_handedness or []))
    if prefer_side is not None:
        for hlm, hd in pairs:
            side = hd.classification[0].label.upper() if getattr(hd, "classification", None) else None
            if side == prefer_side:
                return hlm, side
    annotated = []
    for hlm, hd in pairs:
        side = hd.classification[0].label.upper() if getattr(hd, "classification", None) else "ANY"
        annotated.append((hlm, side))
    if HAND_TARGET == "ANY":
        return annotated[0]
    for hlm, side in annotated:
        if side == HAND_TARGET:
            return hlm, side
    return None, None

# ===================== 액션 매핑 =====================
ACTION_MAP = {
    "sliding_left": ["xdotool", "key", "Alt+Left"],
    "swipe_right_to_left": ["xdotool", "key", "Alt+Left"],
    "sliding_right": ["xdotool", "key", "Alt+Right"],
    "swipe_left_to_right": ["xdotool", "key", "Alt+Right"],
    "scroll_up": ["xdotool", "key", "Prior"],
    "scroll_down": ["xdotool", "key", "Next"],
}

def trigger_action(label: str):
    label_n = map_to_model_label(label)
    cmd = ACTION_MAP.get(label_n)
    if not cmd:
        return
    try:
        subprocess.run(cmd, check=False)
        print(f"⚡ action fired: {label_n}")
    except FileNotFoundError:
        print("⚠ xdotool 미설치. sudo apt install xdotool 또는 Wayland 대안 권장")

# ===================== 메인 루프 =====================
raw_lbl, raw_p = "-", 0.0
gated_lbl, gated_p = "-", 0.0
stable_lbl, stable_cnt = "-", 0
last_fire = 0.0
last_print_time = 0.0
frame_idx = 0

prev_coords_raw = None
prev_hand_side = None

smoothed_draw_np = None
lm_hold = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # 모델 입력은 원본 프레임 사용 (flip 금지)
    # (보기용 미러링은 아래 USE_GUI 블록에서만 수행)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_small = cv2.resize(rgb, (INFER_W, INFER_H))
    rgb_small.flags.writeable = False
    result = hands.process(rgb_small)
    rgb_small.flags.writeable = True

    target, side = choose_hand(result, prefer_side=prev_hand_side if HAND_TARGET == "ANY" else HAND_TARGET)
    if target is not None:
        if side != prev_hand_side:
            prev_coords_raw = None
            prev_hand_side = side

        vec, curr_raw = vectorize_hand(target, prev_coords_raw)  # (F,), (21,D)
        prev_coords_raw = curr_raw

        seq_arr[seq_ptr, :] = vec
        seq_ptr = (seq_ptr + 1) % SEQUENCE_LENGTH
        if seq_ptr == 0:
            seq_filled = True

        if seq_filled:
            x = np.expand_dims(np.roll(seq_arr, -seq_ptr, axis=0), 0)  # (1,T,F)
            prob = model(x, training=False).numpy()[0]                 # (C,)
            p_max = float(prob.max())
            idx   = int(prob.argmax())
            raw_lbl = le.classes_[idx] if 0 <= idx < len(le.classes_) else "-"
            raw_lbl = map_to_model_label(raw_lbl)
            raw_p   = p_max

            if p_max >= P_THRESH:
                gated_lbl, gated_p = raw_lbl, p_max
                pred_buf.append(gated_lbl)
            else:
                gated_lbl, gated_p = "-", 0.0
                pred_buf.append("_")

            counts = Counter([p for p in pred_buf if p != "_"])
            stable_lbl, stable_cnt = (counts.most_common(1)[0] if counts else ("-", 0))

            now = time.time()
            if stable_cnt >= STABLE_MIN and (now - last_fire >= COOLDOWN_SEC):
                trigger_action(stable_lbl)
                last_fire = now
                pred_buf.clear()

        curr_np = lm_to_np(target)
        smoothed_draw_np = smooth_np(curr_np, smoothed_draw_np, LM_SMOOTH_ALPHA)
        lm_hold = LM_HOLD_FRAMES
    else:
        raw_lbl, raw_p = "no_hand", 0.0
        gated_lbl, gated_p = "-", 0.0
        prev_coords_raw = None
        if lm_hold > 0:
            lm_hold -= 1
        else:
            smoothed_draw_np = None

    if USE_GUI:
        # 좌표는 원본 프레임에서 그려야 위치가 일치함
        if (frame_idx % LANDMARK_DRAW_EVERY == 0) and (smoothed_draw_np is not None):
            draw_landmarks_smooth(frame, smoothed_draw_np)

        # 보기 전용으로만 미러링(사용자 직관 일치)
        vis = cv2.flip(frame, 1)

        y0 = 28
        cv2.putText(vis, f"Raw:    {raw_lbl} ({raw_p:.2f})", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(vis, f"Gated:  {gated_lbl} ({gated_p:.2f})  [pth={P_THRESH:.2f}]",
                    (10, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 255, 120), 2)
        cv2.putText(vis, f"Stable: {stable_lbl} (cnt={stable_cnt}/{STABLE_WINDOW})",
                    (10, y0+60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 220, 255), 2)
        cv2.putText(vis, f"T={SEQUENCE_LENGTH} F={FEATURE_DIM} | hand={HAND_TARGET} "
                           f"| res={INFER_W}x{INFER_H} | layout={LAYOUT} | smooth={LM_SMOOTH_ALPHA} hold={LM_HOLD_FRAMES}",
                    (10, y0+90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)

        cv2.imshow("Live Inference (GRU + MediaPipe HANDS)", vis)

        now = time.time()
        if now - last_print_time >= PRINT_INTERVAL:
            print(f"[{time.strftime('%H:%M:%S')}] raw={raw_lbl}({raw_p:.2f}) | "
                  f"gated={gated_lbl}({gated_p:.2f}) | stable={stable_lbl} cnt={stable_cnt}/{STABLE_WINDOW}")
            last_print_time = now

        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        now = time.time()
        if now - last_print_time >= PRINT_INTERVAL:
            print(f"[{time.strftime('%H:%M:%S')}] raw={raw_lbl}({raw_p:.2f}) | "
                  f"gated={gated_lbl}({gated_p:.2f}) | stable={stable_lbl} cnt={stable_cnt}/{STABLE_WINDOW}")
            last_print_time = now

cap.release()
if USE_GUI:
    cv2.destroyAllWindows()
