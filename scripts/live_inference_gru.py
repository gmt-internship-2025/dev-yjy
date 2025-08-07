import mediapipe as mp
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import joblib

# 설정
SEQUENCE_LENGTH = 10
DATA_DIM = 66  # x33 + y33
LABEL_PATH = "../model/label_encoder.pkl"
MODEL_PATH = "../model/gesture_gru_model.h5"

# 모델 & 인코더 로드
model = load_model(MODEL_PATH)
label_encoder = joblib.load(LABEL_PATH)

# MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("실시간 GRU 제스처 인식 시작 (ESC로 종료)")
sequence = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        # 랜드마크 시각화
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        lm = result.pose_landmarks.landmark
        frame_data = [l.x for l in lm] + [l.y for l in lm]  # 66차원
        sequence.append(frame_data)

        # 시퀀스 유지
        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        if len(sequence) == SEQUENCE_LENGTH:
            input_seq = np.array([sequence])  # shape: (1, 10, 66)
            pred = model.predict(input_seq, verbose=0)
            pred_label = label_encoder.inverse_transform([np.argmax(pred)])
            cv2.putText(frame, f"Gesture: {pred_label[0]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            print("예측 결과:", pred_label[0])

    cv2.imshow("Live Inference (GRU)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
