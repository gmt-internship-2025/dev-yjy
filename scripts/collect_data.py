import mediapipe as mp
import cv2
import numpy as np
import os
import re
import sys
import time

# ========= 설정 =========
DATA_DIR = "../data"
LABEL = "stop"   # 제스처 이름
SEQUENCE_LENGTH = 10       # 한 번 누를 때 캡처할 프레임 수 (시퀀스 길이)
USE_Z = False              # z 좌표까지 사용할지 (True면 (10, 63), False면 (10, 42))
USE_STANDARDIZE = True     # 축별 표준화(평균0, 표준편차1) 적용 여부
DETECT_CONF_THRESH = 0.5   # hand detection 신뢰도 임계값
SCALE_MIN_THRESH = 1e-3    # 스케일 너무 작을 때(손이 너무 작게 잡힐 때) 필터링
EPS = 1e-6

# GUI 사용 여부 (환경변수로 제어 가능: NO_GUI=1 이면 GUI 끔)
NO_GUI = os.getenv("NO_GUI", "0") == "1"

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
print(f"[INIT] 기존 시퀀스 파일: {len(existing_files)}개 → {start_index}번부터 저장 시작")

# ========= MediaPipe Hands 인덱스 =========
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# ========= 전처리 유틸 =========
def _pick_origin_and_scale_hand(coords):
    """
    coords: (21, D)  (D=2 또는 3)
    반환: origin (D,), scale (float)
    기본 정책:
      - origin: 손목(WRIST) 중심
      - scale 우선순위: (INDEX_MCP ~ PINKY_MCP 너비) > (WRIST ~ MIDDLE_MCP 거리) > (바운딩박스 대각선) > 1
    """
    def dist(a, b): return float(np.linalg.norm(a - b))

    wrist = coords[WRIST]
    origin = wrist

    # 1) 손 너비
    scale = dist(coords[INDEX_MCP], coords[PINKY_MCP])
    if scale > EPS:
        return origin, scale

    # 2) 손 길이
    scale = dist(coords[WRIST], coords[MIDDLE_MCP])
    if scale > EPS:
        return origin, scale

    # 3) 바운딩 박스 대각선
    diag = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    if diag > EPS:
        return origin, diag

    # 4) 최후의 방책
    return origin, 1.0

def preprocess_hand_landmarks(hand_landmarks):
    """
    입력: mediapipe hands single hand_landmarks.landmark (길이 21)
    출력: (42,) 또는 (63,) 벡터 (x,y[,z]) 전처리 후 플랫
    """
    lm = hand_landmarks.landmark
    if USE_Z:
        coords = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)  # (21,3)
    else:
        coords = np.array([[l.x, l.y] for l in lm], dtype=np.float32)       # (21,2)

    origin, scale = _pick_origin_and_scale_hand(coords)
    if scale < SCALE_MIN_THRESH:
        # 손이 너무 작게 잡혔거나, 스케일이 불안정한 경우: 사용하지 않음(필터링)
        return None

    coords = (coords - origin) / (scale if scale > EPS else 1.0)

    if USE_STANDARDIZE:
        mean = coords.mean(axis=0, keepdims=True)
        std = coords.std(axis=0, keepdims=True) + 1e-6
        coords = (coords - mean) / std

    return coords.flatten()  # (21*D,)

# ========= NO-GUI 키 입력(비차단) =========
def read_stdin_key_nonblocking():
    """
    NO_GUI 환경에서 터미널로부터 s/q 키(엔터 포함)를 비차단으로 읽는다.
    리눅스/맥: select, 윈도우: msvcrt
    반환: 's', 'q', None
    """
    try:
        if os.name == "nt":
            import msvcrt
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                # 엔터 없이 바로 키 처리. 대소문자 무시
                if ch in ('s', 'S'):
                    return 's'
                if ch in ('q', 'Q', '\x1b'):
                    return 'q'
        else:
            import select
            dr, _, _ = select.select([sys.stdin], [], [], 0)
            if dr:
                line = sys.stdin.readline().strip()
                if line.lower() == 's':
                    return 's'
                if line.lower() == 'q':
                    return 'q'
    except Exception:
        # 입력 관련 에러는 무시하고 None
        pass
    return None

# ========= MediaPipe 설정 =========
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=DETECT_CONF_THRESH,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[WARN] 카메라(인덱스 0)를 열 수 없습니다. 장치를 확인하세요.")
    sys.exit(1)

frames = []                 # 현재 시퀀스용 버퍼 (전처리된 벡터 저장)
collecting = False          # 지금 10프레임 수집 중인지
collected_frames = 0        # 수집된 유효 프레임 수

if NO_GUI:
    print("[READY - NO_GUI] 터미널에서 's' 입력 후 엔터 → 수집 시작, 'q' → 종료")
else:
    print("준비 완료. 's'를 누를 때마다 전처리된 10프레임(손) 시퀀스를 저장합니다. (ESC로 종료)")

def log_status():
    mode = f"Mode: {'x,y,z' if USE_Z else 'x,y'} | Std: {USE_STANDARDIZE}"
    print(f"[STATUS] Collecting={collecting} | NextSeq={sequence_num} | Collected={collected_frames}/{SEQUENCE_LENGTH} | {mode}")

last_log_t = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 프레임을 읽을 수 없습니다. 카메라 상태를 확인하세요.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # ===== 수집 로직: 손 검출 + 전처리 통과한 프레임만 누적 =====
    if collecting and result.multi_hand_landmarks:
        det_ok = True
        if result.multi_handedness:
            score = result.multi_handedness[0].classification[0].score
            det_ok = (score >= DETECT_CONF_THRESH)

        if det_ok:
            hand_vec = preprocess_hand_landmarks(result.multi_hand_landmarks[0])  # (42,) or (63,) or None
            if hand_vec is not None:
                frames.append(hand_vec)
                collected_frames += 1
                if time.time() - last_log_t > 0.3:
                    print(f"[COLLECT] valid_frame {collected_frames}/{SEQUENCE_LENGTH}")
                    last_log_t = time.time()
            else:
                if time.time() - last_log_t > 0.7:
                    print("[SKIP] scale too small / unstable → 프레임 제외")
                    last_log_t = time.time()
        else:
            if time.time() - last_log_t > 0.7:
                print("[SKIP] detection score below threshold")
                last_log_t = time.time()

        # 10프레임 채워지면 저장
        if collected_frames >= SEQUENCE_LENGTH:
            sequence_array = np.stack(frames, axis=0)  # (10, 42) or (10, 63)
            save_path = os.path.join(label_path, f"seq_{sequence_num}.npy")
            np.save(save_path, sequence_array)
            print(f"[SAVE] {save_path}  shape={sequence_array.shape}")

            # 다음 트리거 준비
            sequence_num += 1
            frames = []
            collected_frames = 0
            collecting = False
            print("[INFO] 시퀀스 저장 완료. 다음 수집을 위해 's'를 눌러주세요.")

    # ===== 표시/키입력 처리 =====
    if not NO_GUI:
        # 랜드마크 그리기(화면 표시용)
        if result.multi_hand_landmarks:
            for hlmk in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hlmk, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

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

        try:
            cv2.imshow("Collect 10-frame HAND sequence (preprocessed)", frame)
            key = cv2.waitKey(1) & 0xFF
        except Exception as e:
            # X11 DISPLAY가 없을 때 등 예외 : NO_GUI로 전환
            print(f"[WARN] imshow 사용 불가({e}). NO_GUI 모드로 전환합니다. 터미널에서 's'/'q'를 입력하세요.")
            NO_GUI = True
            key = 255

        if not NO_GUI:
            if key == ord('s'):
                if not collecting:
                    collecting = True
                    frames = []
                    collected_frames = 0
                    print("[START] 시퀀스 수집 시작 — 전처리 통과 프레임 기준 10개 모이면 자동 저장")
                    log_status()
                else:
                    print("[INFO] 이미 수집 중입니다.")
            elif key == 27:  # ESC
                print("[EXIT] ESC 입력으로 종료")
                break
    else:
        # NO_GUI: 터미널 입력 확인
        k = read_stdin_key_nonblocking()
        if k == 's':
            if not collecting:
                collecting = True
                frames = []
                collected_frames = 0
                print("[START] (NO_GUI) 시퀀스 수집 시작")
                log_status()
            else:
                print("[INFO] 이미 수집 중입니다.")
        elif k == 'q':
            print("[EXIT] (NO_GUI) 종료 명령 수신")
            break

        # 너무 바쁘지 않게 살짝 쉼
        time.sleep(0.005)

cap.release()
cv2.destroyAllWindows()
print("[DONE] 종료되었습니다.")
