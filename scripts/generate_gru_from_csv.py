import pandas as pd
import numpy as np
import os

SEQUENCE_LENGTH = 10
FEATURE_DIM = 66
CSV_PATH = "../data/pose_data_augmented.csv"

df = pd.read_csv(CSV_PATH)
labels = df["label"].values
features = df.drop("label", axis=1).values

# 시퀀스 수 계산
num_samples = features.shape[0]
X_data = features.reshape((num_samples, SEQUENCE_LENGTH, FEATURE_DIM))
y_data = labels

# 저장
np.save("../data/X_gru.npy", X_data)
np.save("../data/y_gru.npy", y_data)
print("정규화+증강된 X_gru.npy, y_gru.npy 저장 완료!")
