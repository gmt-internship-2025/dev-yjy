import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Masking
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# ===== 1. 데이터 로딩 =====
df = pd.read_csv("../data/pose_data.csv")

# ===== 2. 시계열 윈도우 만들기 =====
SEQUENCE_LENGTH = 10  # 연속된 10프레임을 하나의 sample로 사용

X_data = []
y_data = []

features = df.drop("label", axis=1).values
labels = df["label"].values

for i in range(len(df) - SEQUENCE_LENGTH):
    x_seq = features[i:i+SEQUENCE_LENGTH]
    y_seq = labels[i+SEQUENCE_LENGTH - 1]  # 마지막 프레임의 레이블
    X_data.append(x_seq)
    y_data.append(y_seq)

X_data = np.array(X_data)  # shape: (num_samples, seq_len, num_features)
y_data = np.array(y_data)

# ===== 3. 레이블 숫자로 인코딩 =====
le = LabelEncoder()
y_encoded = le.fit_transform(y_data)
y_onehot = to_categorical(y_encoded)

# ===== 4. 학습/테스트 분리 =====
X_train, X_test, y_train, y_test = train_test_split(X_data, y_onehot, test_size=0.2)

# ===== 5. 모델 정의 (GRU) =====
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(SEQUENCE_LENGTH, X_data.shape[2])))
model.add(GRU(64, return_sequences=False))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_onehot.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ===== 6. 학습 =====
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# ===== 7. 평가 및 저장 =====
loss, acc = model.evaluate(X_test, y_test)
print("🎯 테스트 정확도:", acc)

# 모델 저장
model.save("../model/gesture_gru_model.h5")
joblib.dump(le, "../model/label_encoder.pkl")
