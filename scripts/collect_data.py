import mediapipe as mp
import cv2
import numpy as np
import os
import re

# ========= 설정 =========
DATA_DIR = "../data"
LABEL = "sliding_left"           # 제스처 이름
SEQUENCE_LENGTH = 10        # 한 번 누를 때 캡처할 프레임 수 (시퀀스 길이)
USE_Z = False               # z 좌표까지 사용할지 (True면 (10, 99), False면 (10, 66))
USE_STANDARDIZE = True      # 축별 표준화(평균0, 표준편차1) 적용 여부
VIS_THRESH = 0.5            # 가시성(visibility) 임계값
EPS = 1e-6

# MediaPipe Pose 인덱스
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_HIP, R_HIP = 23, 24

# ========= 저장 경로 생성 & 시작 인덱스 =========
label_path = os.path.join(DATA_DIR, LABEL)
os.makedirs(label_path, exist_ok=True)

existing_files = [f for f in os.listdir(label_path) if f.endswith(".npy")]
existing_nums = []
for f in existing_files:
    m = re.search(r"seq_(\d+)\.npy", f)
    if m:
        existing_nums.append(int(m.group(1)))

start_index = max(existing_nums) + 1 if existing_nums else 0
sequence_num = start_index
print(f"기존 시퀀스 파일: {len(existing_files)}개 → {start_index}번부터 저장 시작")

# ========= 전처리 유틸 =========
def _is_visible(lmk, idx, thresh=VIS_THRESH):
    try:
        return lmk[idx].visibility >= thresh
    except Exception:
        return False

def _pick_origin_and_scale(coords, lmk):
    """
    coords: (33, D)  (D=2 또는 3), lmk: mediapipe landmark list
    반환: origin (D,), scale (float)
    우선순위:
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

    # 1) 골반 둘 다 보임 → 골반중심, 스케일은 어깨너비/대체
    if have_lhip and have_rhip:
        origin = center(coords[L_HIP], coords[R_HIP])
        if have_lsho and have_rsho:
            scale = dist(coords[L_SHOULDER], coords[R_SHOULDER])
            if scale > EPS:
                return origin, scale
        upper_idxs = [i for i in [NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW] if _is_visible(lmk, i)]
        if upper_idxs:
            box = coords[upper_idxs]
            diag = float(np.linalg.norm(box.max(axis=0) - box.min(axis=0)))
            return origin, (diag if diag > EPS else 1.0)
        return origin, 1.0

    # 2) 어깨 둘 다 보임 → 어깨중심, 어깨너비
    if have_lsho and have_rsho:
        origin = center(coords[L_SHOULDER], coords[R_SHOULDER])
        scale = dist(coords[L_SHOULDER], coords[R_SHOULDER])
        if scale > EPS:
            return origin, scale
        # 어깨너비 이상치면 어깨-코 거리
        if have_nose:
            alt = dist(coords[L_SHOULDER], coords[NOSE])
            if alt > EPS:
                return origin, alt
        upper_idxs = [i for i in [NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW] if _is_visible(lmk, i)]
        if upper_idxs:
            box = coords[upper_idxs]
            diag = float(np.linalg.norm(box.max(axis=0) - box.min(axis=0)))
            return origin, (diag if diag > EPS else 1.0)
        return origin, 1.0

    # 3) 한쪽 어깨 + 코 → 중점/거리
    if have_nose and (have_lsho or have_rsho):
        s_idx = L_SHOULDER if have_lsho else R_SHOULDER
        origin = (coords[s_idx] + coords[NOSE]) / 2.0
        scale = float(np.linalg.norm(coords[s_idx] - coords[NOSE]))
        return origin, (scale if scale > EPS else 1.0)

    # 4) 상체 평균/박스 대각선
    upper_idxs = [i for i in [NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW] if _is_visible(lmk, i)]
    if upper_idxs:
        box = coords[upper_idxs]
        origin = box.mean(axis=0)
        diag = float(np.linalg.norm(box.max(axis=0) - box.min(axis=0)))
        return origin, (diag if diag > EPS else 1.0)

    # 5) 정말 없으면 (0,0[,0]), scale=1
    return np.zeros((coords.shape[1],), dtype=np.float32), 1.0

def preprocess_landmarks(pose_landmarks):
    """
    입력: mediapipe pose_landmarks.landmark (길이 33)
    출력: (66,) 또는 (99,) 벡터 (x,y[,z]) 전처리 후 플랫
      - 기준점: 상황별(골반/어깨/코/상체)
      - 스케일: 어깨너비/어깨-코/상체박스 대각선
      - (옵션) 축별 표준화
    """
    lm = pose_landmarks.landmark
    if USE_Z:
        coords = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)  # (33,3)
    else:
        coords = np.array([[l.x, l.y] for l in lm], dtype=np.float32)       # (33,2)

    origin, scale = _pick_origin_and_scale(coords, lm)
    coords = (coords - origin) / (scale if scale > EPS else 1.0)

    if USE_STANDARDIZE:
        mean = coords.mean(axis=0, keepdims=True)
        std = coords.std(axis=0, keepdims=True) + 1e-6
        coords = (coords - mean) / std

    return coords.flatten()  # (33*D,)

# ========= MediaPipe 설정 =========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

frames = []                 # 현재 시퀀스용 버퍼 (전처리된 벡터 저장)
collecting = False          # 지금 10프레임 수집 중인지
collected_frames = 0        # 수집된 유효 프레임 수(포즈가 검출된 프레임만 카운트)

print("준비 완료. 's'를 누를 때마다 전처리된 10프레임 시퀀스를 저장합니다. (ESC로 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    # 랜드마크 그리기(화면 표시용)
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

    # ===== 수집 로직: 포즈 검출된 프레임만 전처리하여 누적 =====
    if collecting and result.pose_landmarks:
        vec = preprocess_landmarks(result.pose_landmarks)  # (66,) or (99,)
        frames.append(vec)
        collected_frames += 1

        # 10프레임 채워지면 저장
        if collected_frames >= SEQUENCE_LENGTH:
            sequence_array = np.stack(frames, axis=0)  # (10, 66) 또는 (10, 99)
            save_path = os.path.join(label_path, f"seq_{sequence_num}.npy")
            np.save(save_path, sequence_array)

            print(f"시퀀스 저장: {save_path}  shape={sequence_array.shape}")

            # 다음 트리거 준비
            sequence_num += 1
            frames = []
            collected_frames = 0
            collecting = False  # 자동으로 수집 종료

    # 화면 안내 메시지
    status = "ON" if collecting else "OFF"
    mode = f"Mode: {'x,y,z' if USE_Z else 'x,y'} | Std: {USE_STANDARDIZE}"
    msg1 = f"Collecting: {status}  |  Next seq index: {sequence_num}"
    msg2 = "Press 's' to capture 10 frames  •  ESC to quit"
    cv2.putText(frame, msg1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    cv2.putText(frame, msg2, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
    cv2.putText(frame, mode, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
    if collecting:
        cv2.putText(frame, f"{collected_frames}/{SEQUENCE_LENGTH}",
                    (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Collect 10-frame sequence (preprocessed)", frame)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # 이미 수집 중이면 무시(디바운싱)
        if not collecting:
            collecting = True
            frames = []
            collected_frames = 0
            print("▶ 시퀀스 수집 시작 — 포즈가 검출된 프레임 기준으로 10개 모이면 자동 저장")
        else:
            print("… 이미 수집 중입니다. 잠시만 기다려주세요.")
    elif key == 27:  # ESC
        print("ESC 입력으로 종료")
        break

cap.release()
cv2.destroyAllWindows()
