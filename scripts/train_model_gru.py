import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Masking
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# ===== 1. 데이터 로딩 =====
X_data = np.load("../data/X_gru.npy")  # shape: (num_samples, 10, 66)
y_data = np.load("../data/y_gru.npy")  # shape: (num_samples,)

# ===== 2. 레이블 숫자로 인코딩 =====
le = LabelEncoder()
y_encoded = le.fit_transform(y_data)
y_onehot = to_categorical(y_encoded)

# ===== 3. 학습/테스트 분리 =====
X_train, X_test, y_train, y_test = train_test_split(X_data, y_onehot, test_size=0.2, stratify=y_onehot)

# ===== 4. 모델 정의 (GRU) =====
SEQUENCE_LENGTH = X_data.shape[1]
FEATURE_DIM = X_data.shape[2]

model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)))
model.add(GRU(64, return_sequences=False))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_onehot.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ===== 5. 학습 =====
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# ===== 6. 평가 및 저장 =====
loss, acc = model.evaluate(X_test, y_test)
print("테스트 정확도:", acc)

# ===== 7. 저장 =====
os.makedirs("../model", exist_ok=True)
model.save("../model/gesture_gru_model.h5")
joblib.dump(le, "../model/label_encoder.pkl")

model.summary()

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
save_path = "../result/training_curve.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)

print(f"✅ 학습 곡선 그래프가 저장되었습니다: {save_path}")
