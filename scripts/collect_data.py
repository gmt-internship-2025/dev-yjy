import mediapipe as mp
import cv2
import csv
import os

DATA_PATH = "../data/pose_data.csv"
LABEL = "default"  # ← 이 부분만 바꾸면서 수집!

# MediaPipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

# 파일이 없으면 헤더 추가
write_header = not os.path.exists(DATA_PATH)
with open(DATA_PATH, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["label"] + [f"x{i}" for i in range(33)] + [f"y{i}" for i in range(33)])

    print(f"🟢 '{LABEL}' 제스처 수집 중... ESC로 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )
            lm = result.pose_landmarks.landmark
            row = [LABEL] + [l.x for l in lm] + [l.y for l in lm]
            writer.writerow(row)

        cv2.imshow("Collecting...", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
