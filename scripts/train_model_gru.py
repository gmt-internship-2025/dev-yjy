import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# ===== 0. 경로/설정 =====
DATA_DIR = "../data"
MODEL_DIR = "../model"
RESULT_DIR = "../result"
CONFIG_PATH = os.path.join(DATA_DIR, "dataset_config.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ===== 1. 데이터 로딩 =====
X_data = np.load(os.path.join(DATA_DIR, "X_gru.npy"))  # (num_samples, T, F)
# (수정) object dtype 대비: allow_pickle=True 로 로드 후 문자열로 캐스팅
y_data = np.load(os.path.join(DATA_DIR, "y_gru.npy"), allow_pickle=True).astype(str)  # (num_samples, )

# 타입/형상 정리
X_data = X_data.astype(np.float32)
assert X_data.ndim == 3, f"X_data shape expected (N,T,F), got {X_data.shape}"

SEQUENCE_LENGTH = X_data.shape[1]
FEATURE_DIM = X_data.shape[2]

# dataset_config.json이 있으면 모양 검증 (수집 스크립트와 일관성 체크)
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    cfg_T = cfg.get("sequence_length", SEQUENCE_LENGTH)
    cfg_F = cfg.get("feature_dim", FEATURE_DIM)
    if (cfg_T != SEQUENCE_LENGTH) or (cfg_F != FEATURE_DIM):
        raise ValueError(
            f"데이터셋 설정 불일치: config(T={cfg_T}, F={cfg_F}) vs 데이터(T={SEQUENCE_LENGTH}, F={FEATURE_DIM})"
        )
else:
    print("[경고] dataset_config.json을 찾지 못했습니다. "
          "수집/전처리와 입력 차원이 항상 동일한지 직접 확인하세요.")

print(f"[INFO] X_data: {X_data.shape} (T={SEQUENCE_LENGTH}, F={FEATURE_DIM})")

# ===== 2. 레이블 인코딩 =====
le = LabelEncoder()
y_encoded = le.fit_transform(y_data)       # 1D 정수 라벨
num_classes = len(le.classes_)
y_onehot = to_categorical(y_encoded, num_classes=num_classes)

print(f"[INFO] classes({num_classes}): {list(le.classes_)}")

# ===== 3. 학습/테스트 분리 =====
# stratify는 1차원 라벨을 사용해야 함(원-핫 X)
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
)

# ===== 4. 모델 정의 (GRU) =====
model = Sequential()
# 패딩이 없는 고정 길이라면 Masking은 필수는 아님. 남겨둬도 문제는 없음.
model.add(Masking(mask_value=0.0, input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)))
model.add(GRU(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(64))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ===== 5. 콜백 =====
ckpt_path = os.path.join(MODEL_DIR, "gesture_gru_best.keras")
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy", mode="max"),
    ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5, monitor="val_loss"),
    ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max")
]

# ===== 6. 학습 =====
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,   # 혹은 별도 검증셋을 만들고 싶으면 한 번 더 split
    callbacks=callbacks,
    verbose=1
)

# ===== 7. 평가 =====
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("테스트 정확도:", acc)

# ===== 8. 저장 =====
final_model_path = os.path.join(MODEL_DIR, "gesture_gru_final.keras")
model.save(final_model_path)                       # 권장 포맷
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
print(f"[INFO] 모델 저장: {final_model_path}")
print(f"[INFO] Best checkpoint: {ckpt_path}")

# ===== 9. 학습 곡선 저장 =====
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy')
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss')
plt.legend(); plt.grid(True)

plt.tight_layout()
save_path = os.path.join(RESULT_DIR, "training_curve.png")
plt.savefig(save_path)
print(f"✅ 학습 곡선 그래프가 저장되었습니다: {save_path}")
