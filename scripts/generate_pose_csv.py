# generate_pose_csv_gru.py
import os
import numpy as np
import pandas as pd

DATA_DIR = "../data"
SEQUENCE_LENGTH = 10

X_data = []
y_data = []

for label_name in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label_name)
    if not os.path.isdir(label_path):
        continue

    for filename in os.listdir(label_path):
        if filename.endswith(".npy"):
            file_path = os.path.join(label_path, filename)
            data = np.load(file_path)

            if data.shape == (SEQUENCE_LENGTH, 66):  # 시퀀스 (10, 66) 맞는 경우만
                X_data.append(data)
                y_data.append(label_name)

# 넘파이 배열로 변환
X_data = np.array(X_data)  # shape: (num_samples, 10, 66)
y_data = np.array(y_data)

# 저장
np.save("../data/X_gru.npy", X_data)
np.save("../data/y_gru.npy", y_data)
print("X_gru.npy, y_gru.npy 저장 완료!")
