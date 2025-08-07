import mediapipe as mp
import cv2
import numpy as np
import os
import re

DATA_DIR = "../data"
LABEL = "default"           # 제스처 이름
SEQUENCE_LENGTH = 10             # 시퀀스 길이
SAVE_COUNT = 30                  # 저장할 시퀀스 수

# 저장 경로 생성
label_path = os.path.join(DATA_DIR, LABEL)
os.makedirs(label_path, exist_ok=True)

# ===== 기존 파일에서 시작 인덱스 찾기 =====
existing_files = [f for f in os.listdir(label_path) if f.endswith(".npy")]
existing_nums = []

for f in existing_files:
    match = re.search(r"seq_(\d+)\.npy", f)
    if match:
        existing_nums.append(int(match.group(1)))

start_index = max(existing_nums) + 1 if existing_nums else 0
sequence_num = start_index
print(f"기존 시퀀스 파일: {len(existing_files)}개 → {start_index}번부터 수집 시작")

# MediaPipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

frames = []
collecting = False  # 수집 상태 플래그

print(f"🕓 '{LABEL}' 제스처 수집 대기 중... 키보드에서 [s]를 누르면 시작")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
        )

        if collecting:
            lm = result.pose_landmarks.landmark
            frame_data = [l.x for l in lm] + [l.y for l in lm]
            frames.append(frame_data)

            if len(frames) == SEQUENCE_LENGTH:
                sequence_array = np.array(frames)
                np.save(os.path.join(label_path, f"seq_{sequence_num}.npy"), sequence_array)
                sequence_num += 1
                frames = []

                print(f"시퀀스 {sequence_num - start_index}/{SAVE_COUNT} 저장됨")

                if (sequence_num - start_index) >= SAVE_COUNT:
                    print("🎉 모든 시퀀스 수집 완료")
                    break

    # 화면 표시
    msg = f"Collecting: {collecting} | Sequence: {sequence_num - start_index}/{SAVE_COUNT}"
    cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    cv2.imshow("Collecting GRU Sequence...", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        collecting = True
        print("수집 시작 (s 키 입력)")
    elif key == 27:  # ESC
        print("ESC 입력으로 종료")
        break

cap.release()
cv2.destroyAllWindows()
