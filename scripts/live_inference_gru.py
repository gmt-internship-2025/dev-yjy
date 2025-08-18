# live_inference_gru.py — 붙여넣고 바로 실행
import os, time, json, subprocess, warnings
import numpy as np
import cv2
import mediapipe as mp
from collections import deque, Counter
from tensorflow.keras.models import load_model
import joblib

# ===================== 경로/고정 설정 =====================
DATA_DIR = "../data"
MODEL_DIR = "../model"
CONFIG_PATH = os.path.join(DATA_DIR, "dataset_config.json")
LABEL_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# 모델 후보(존재하는 첫 파일을 사용)
CANDIDATE_MODELS = [
    os.path.join(MODEL_DIR, "gesture_gru_best.keras"),
    os.path.join(MODEL_DIR, "gesture_gru_final.keras"),
    os.path.join(MODEL_DIR, "gesture_gru_model.h5"),  # 구버전 호환
]

# 추론 안정화/출력 파라미터
P_THRESH = 0.50         # softmax 확률 임계치
STABLE_WINDOW = 4       # 최근 N개 예측 보관
STABLE_MIN = 4          # 그중 동일 라벨 최소 개수
COOLDOWN_SEC = 0.8      # 재발동 쿨다운(초)
PRINT_INTERVAL = 0.5    # 콘솔 라벨 출력 주기(초)

# ===================== dataset_config 로드 =====================
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    SEQUENCE_LENGTH = int(cfg.get("sequence_length", 10))
    FEATURE_DIM = int(cfg.get("feature_dim", 66))
    USE_STANDARDIZE = bool(cfg.get("use_standardize", True))
    VIS_THRESH = float(cfg.get("vis_thresh", 0.5))
    USE_Z = (FEATURE_DIM == 99)
else:
    warnings.warn("[경고] dataset_config.json이 없습니다. 수집/학습과 입력 차원이 다르면 오류가 납니다.")
    SEQUENCE_LENGTH = 10
    FEATURE_DIM = 66
    USE_STANDARDIZE = True
    VIS_THRESH = 0.5
    USE_Z = False

print(f"[INFO] Expected per-seq shape: (T={SEQUENCE_LENGTH}, F={FEATURE_DIM}), "
      f"mode={'xyz' if USE_Z else 'xy'}, standardize={USE_STANDARDIZE}")

# ===================== 모델/라벨 로드 & 검증 =====================
model_path = next((p for p in CANDIDATE_MODELS if os.path.exists(p)), None)
if model_path is None:
    raise FileNotFoundError(
        "모델 파일을 찾을 수 없습니다. 아래 경로 중 하나에 모델이 있어야 합니다.\n" +
        "\n".join(" - " + p for p in CANDIDATE_MODELS)
    )

model = load_model(model_path)
le = joblib.load(LABEL_PATH)

num_units = model.output_shape[-1]
num_labels = len(le.classes_)
print(f"[INFO] Loaded model: {model_path}")
print(f"[INFO] model units={num_units}, label classes={num_labels} -> {le.classes_.tolist()}")

if num_units != num_labels:
    raise RuntimeError(
        f"모델과 라벨 인코더 불일치: model units={num_units}, labels={num_labels}\n"
        f"→ 같은 학습 세션에서 저장된 model(.keras/.h5)과 label_encoder.pkl 페어를 사용하세요."
    )

# ===================== MediaPipe Pose 준비 =====================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=1, enable_segmentation=False)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    raise SystemExit

print("🎥 실시간 GRU 제스처 인식 시작 (ESC로 종료)")
seq_buf = deque(maxlen=SEQUENCE_LENGTH)  # 전처리된 벡터 버퍼
pred_buf = deque(maxlen=STABLE_WINDOW)
last_fire = 0.0

# 콘솔 출력 상태
last_print_time = 0.0
display_lbl = "-"      # 화면/콘솔 표시용 현재 라벨
display_p = 0.0

# ===================== 전처리 유틸 (수집/학습 동일) =====================
# Mediapipe landmark 인덱스
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_HIP, R_HIP = 23, 24

EPS = 1e-6

def _is_visible(lmk, idx, thresh=VIS_THRESH):
    try:
        return lmk[idx].visibility >= thresh
    except Exception:
        return False

def _pick_origin_and_scale(coords, lmk):
    """
    coords: (33, D) (D=2 or 3)
    origin/scale 우선순위:
      - origin: (골반중심) > (어깨중심) > (어깨/코 중점) > (상체 평균) > (0)
      - scale:  (어깨너비) > (어깨-코) > (상체 박스 대각선) > 1
    """
    def center(a, b): return (a + b) / 2.0
    def dist(a, b): return float(np.linalg.norm(a - b))

    have_lhip = _is_visible(lmk, L_HIP)
    have_rhip = _is_visible(lmk, R_HIP)
    have_lsho = _is_visible(lmk, L_SHOULDER)
    have_rsho = _is_visible(lmk, R_SHOULDER)
    have_nose = _is_visible(lmk, NOSE)

    # 1) 골반 둘 다 보임
    if have_lhip and have_rhip:
        origin = center(coords[L_HIP], coords[R_HIP])
        if have_lsho and have_rsho:
            scale = dist(coords[L_SHOULDER], coords[R_SHOULDER])
            if scale > EPS: return origin, scale
        upper = [i for i in [NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW] if _is_visible(lmk, i)]
        if upper:
            box = coords[upper]
            diag = float(np.linalg.norm(box.max(axis=0) - box.min(axis=0)))
            return origin, (diag if diag > EPS else 1.0)
        return origin, 1.0

    # 2) 어깨 둘 다 보임
    if have_lsho and have_rsho:
        origin = center(coords[L_SHOULDER], coords[R_SHOULDER])
        scale = dist(coords[L_SHOULDER], coords[R_SHOULDER])
        if scale > EPS: return origin, scale
        if have_nose:
            alt = dist(coords[L_SHOULDER], coords[NOSE])
            if alt > EPS: return origin, alt
        upper = [i for i in [NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW] if _is_visible(lmk, i)]
        if upper:
            box = coords[upper]
            diag = float(np.linalg.norm(box.max(axis=0) - box.min(axis=0)))
            return origin, (diag if diag > EPS else 1.0)
        return origin, 1.0

    # 3) 한쪽 어깨 + 코
    if have_nose and (have_lsho or have_rsho):
        s_idx = L_SHOULDER if have_lsho else R_SHOULDER
        origin = (coords[s_idx] + coords[NOSE]) / 2.0
        scale = float(np.linalg.norm(coords[s_idx] - coords[NOSE]))
        return origin, (scale if scale > EPS else 1.0)

    # 4) 상체 평균/박스 대각선
    upper = [i for i in [NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW] if _is_visible(lmk, i)]
    if upper:
        box = coords[upper]
        origin = box.mean(axis=0)
        diag = float(np.linalg.norm(box.max(axis=0) - box.min(axis=0)))
        return origin, (diag if diag > EPS else 1.0)

    # 5) 정말 없으면 (0), 1
    return np.zeros((coords.shape[1],), dtype=np.float32), 1.0

def preprocess_landmarks(pose_landmarks):
    """
    입력: mediapipe pose_landmarks.landmark (길이 33)
    출력: (FEATURE_DIM,) 전처리 벡터
      - 기준점 이동(골반/어깨/코/상체)
      - 스케일 정규화(어깨너비/어깨-코/상체박스)
      - (옵션) 축별 표준화
    """
    lm = pose_landmarks.landmark
    if USE_Z:
        coords = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)  # (33,3) -> 99
    else:
        coords = np.array([[l.x, l.y] for l in lm], dtype=np.float32)       # (33,2) -> 66

    origin, scale = _pick_origin_and_scale(coords, lm)
    coords = (coords - origin) / (scale if scale > EPS else 1.0)

    if USE_STANDARDIZE:
        mean = coords.mean(axis=0, keepdims=True)
        std = coords.std(axis=0, keepdims=True) + 1e-6
        coords = (coords - mean) / std

    flat = coords.flatten()
    if flat.shape[0] != FEATURE_DIM:
        raise ValueError(f"전처리 결과 차원 불일치: {flat.shape[0]} != {FEATURE_DIM}")
    return flat

# ===================== 액션 트리거 =====================
def trigger_action(label: str):
    """라벨에 따라 화면 스와이프 단축키 전송"""
    try:
        if label == "sliding_left":
            subprocess.run(["xdotool", "key", "Alt+Left"], check=False)
        elif label == "sliding_right":
            subprocess.run(["xdotool", "key", "Alt+Right"], check=False)
        else:
            return
        print(f"⚡ action fired: {label}")
    except FileNotFoundError:
        print("⚠ xdotool을 찾지 못했습니다. sudo apt install xdotool 또는 Wayland 대안을 사용하세요.")

# ====== 메인 루프 (이 블록으로 교체) ======
raw_lbl, raw_p = "-", 0.0
gated_lbl, gated_p = "-", 0.0
stable_lbl, stable_cnt = "-", 0
last_fire = 0.0
last_print_time = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        # 뼈대 그리기
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(thickness=2),
        )

        # 전처리 + 버퍼 적재
        vec = preprocess_landmarks(result.pose_landmarks)  # (FEATURE_DIM,)
        seq_buf.append(vec)

        # 충분히 모였을 때 추론
        if len(seq_buf) == SEQUENCE_LENGTH:
            x = np.array([list(seq_buf)], dtype=np.float32)  # (1, T, F)
            prob = model.predict(x, verbose=0)[0]            # (num_classes,)
            p_max = float(np.max(prob))
            idx = int(np.argmax(prob))
            raw_lbl = le.classes_[idx] if 0 <= idx < len(le.classes_) else "-"
            raw_p = p_max

            # 임계치 통과(게이트) 라벨
            if p_max >= P_THRESH:
                gated_lbl, gated_p = raw_lbl, p_max
                pred_buf.append(gated_lbl)   # 안정화 버퍼에는 '게이트 통과'만 넣음
            else:
                gated_lbl, gated_p = "-", 0.0
                pred_buf.append("_")

            # 안정화 라벨 계산
            counts = Counter([p for p in pred_buf if p != "_"])
            if counts:
                stable_lbl, stable_cnt = counts.most_common(1)[0]
            else:
                stable_lbl, stable_cnt = "-", 0

            # 액션 트리거 (원하면 안정화 라벨에만 반응)
            now = time.time()
            if stable_lbl in ("sliding_left", "sliding_right") and stable_cnt >= STABLE_MIN:
                if now - last_fire >= COOLDOWN_SEC:
                    trigger_action(stable_lbl)
                    last_fire = now
                    pred_buf.clear()  # 연속 트리거 방지
    else:
        raw_lbl, raw_p = "no_pose", 0.0
        gated_lbl, gated_p = "-", 0.0
        # pred_buf에는 아무 것도 넣지 않음 (일관성)

    # ===== 화면 오버레이: Raw / Gated / Stable 모두 표시 =====
    y0 = 28
    cv2.putText(frame, f"Raw:    {raw_lbl} ({raw_p:.2f})",
                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Gated:  {gated_lbl} ({gated_p:.2f})  [pth={P_THRESH:.2f}]",
                (10, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 255, 120), 2)
    cv2.putText(frame, f"Stable: {stable_lbl} (cnt={stable_cnt}/{STABLE_WINDOW})",
                (10, y0+60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 220, 255), 2)
    cv2.putText(frame, f"T={SEQUENCE_LENGTH} F={FEATURE_DIM}",
                (10, y0+90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

    cv2.imshow("Live Inference (GRU + MediaPipe overlay)", frame)

    # ===== 콘솔에도 0.5초마다 동일 정보 출력 =====
    now = time.time()
    if now - last_print_time >= 0.5:
        print(f"[{time.strftime('%H:%M:%S')}] raw={raw_lbl}({raw_p:.2f}) | "
              f"gated={gated_lbl}({gated_p:.2f}) | stable={stable_lbl} cnt={stable_cnt}/{STABLE_WINDOW}")
        last_print_time = now

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break


cap.release()
cv2.destroyAllWindows()
