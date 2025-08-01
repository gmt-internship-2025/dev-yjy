import mediapipe as mp
import cv2
import numpy as np
import joblib

clf = joblib.load("../model/gesture_model.pkl")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

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
        x = [l.x for l in lm]
        y = [l.y for l in lm]
        data = np.array([x + y])
        pred = clf.predict(data)
        cv2.putText(frame, f"Gesture: {pred[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Live Inference", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
