# train_gru.py — Hands GRU classifier (paths unified + dataset LabelEncoder respected)
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

# ===== 0. 경로/설정 =====
DATASET_DIR = os.path.join("..", "dataset")   # build_dataset_hands.py --out-dir
MODEL_DIR   = os.path.join("..", "model")
RESULT_DIR  = os.path.join("..", "result")
CONFIG_PATH = os.path.join(DATASET_DIR, "dataset_config.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ===== helper =====
def infer_D_and_blocks(F: int):
    """
    FEATURE_DIM에서 좌표 차원 D(2 or 3)와 블록 수 blocks(1=coords, 2=coords+delta) 추론.
    허용 F: 42(=21*2*1), 63(=21*3*1), 84(=21*2*2), 126(=21*3*2)
    """
    if F % (21 * 3) == 0:
        D = 3; blocks = F // (21 * D)
    elif F % (21 * 2) == 0:
        D = 2; blocks = F // (21 * D)
    else:
        raise ValueError(f"FEATURE_DIM={F} 에서 D/blocks 추론 불가 (Hands 전처리 포맷인지 확인).")
    if blocks not in (1, 2):
        raise ValueError(f"지원하지 않는 blocks={blocks} (허용: 1 or 2)")
    return D, blocks

# ===== 1. 데이터 로딩 =====
# build_dataset_hands.py 에서 저장한 파일명 그대로 사용
X_path = os.path.join(DATASET_DIR, "X.npy")
y_path = os.path.join(DATASET_DIR, "y.npy")
y_txt_path = os.path.join(DATASET_DIR, "y_text.npy")
le_path = os.path.join(DATASET_DIR, "label_encoder.pkl")

X_data = np.load(X_path).astype(np.float32)     # (N, T, F)
y_encoded = np.load(y_path)                     # (N,) int-encoded labels

assert X_data.ndim == 3, f"X_data shape expected (N,T,F), got {X_data.shape}"
SEQUENCE_LENGTH = int(X_data.shape[1])
FEATURE_DIM = int(X_data.shape[2])

if FEATURE_DIM not in (42, 63, 84, 126):
    warnings.warn(f"[경고] FEATURE_DIM={FEATURE_DIM} (허용: 42/63/84/126). 전처리/수집 스펙 확인 요망.")

D, BLOCKS = infer_D_and_blocks(FEATURE_DIM)
USE_Z = (D == 3)
USE_DELTA = (BLOCKS == 2)

print(f"[INFO] X_data: {X_data.shape} (T={SEQUENCE_LENGTH}, F={FEATURE_DIM})  "
      f"D={'xyz' if USE_Z else 'xy'}  blocks={BLOCKS} ({'coords+delta' if USE_DELTA else 'coords only'})")

# ===== 1-1. dataset_config.json 일관성 검증/보완 =====
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg_T = int(cfg.get("sequence_length", SEQUENCE_LENGTH))
    cfg_F = int(cfg.get("feature_dim", FEATURE_DIM))
    if (cfg_T != SEQUENCE_LENGTH) or (cfg_F != FEATURE_DIM):
        raise ValueError(f"config(T={cfg_T}, F={cfg_F}) vs 데이터(T={SEQUENCE_LENGTH}, F={FEATURE_DIM}) 불일치")
else:
    # config가 없으면 Hands 기준으로 생성
    cfg = {
        "sequence_length": SEQUENCE_LENGTH,
        "feature_dim": FEATURE_DIM,
        "feature_dim_per_block": int(21 * D),
        "num_blocks": int(BLOCKS),
        "use_delta": bool(USE_DELTA),
        "use_z": bool(USE_Z),
        "modality": "hands",
        "use_standardize": True,
        "layout": "concat([coords, delta])" if USE_DELTA else "coords",
        "note": "F = 21 * D * blocks ; D∈{2(xy),3(xyz)}, blocks∈{1(coords),2(coords+delta)}"
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 생성: {CONFIG_PATH}")

# ===== 2. 라벨 인코더 =====
# 반드시 build 시 저장된 LabelEncoder를 로드 (클래스 순서/매핑 완전 일치)
if not os.path.exists(le_path):
    raise FileNotFoundError(f"label_encoder.pkl이 없습니다: {le_path}\n"
                            f"build_dataset_hands.py 실행 여부를 확인하세요.")
le = joblib.load(le_path)
classes = list(le.classes_)
num_classes = len(classes)

# (선택) y_text 존재 시 매핑 검증
if os.path.exists(y_txt_path):
    y_text = np.load(y_txt_path, allow_pickle=True)  # (N,)
    # encoded → text 역변환 후 비교(샘플 확인)
    inv = le.inverse_transform(y_encoded[:min(10, len(y_encoded))])
    if not all(isinstance(x, (str, np.str_)) for x in inv):
        warnings.warn("[경고] 라벨 역변환 검증 스킵(형식 불일치)")

# 원-핫
y_onehot = to_categorical(y_encoded, num_classes=num_classes)

print(f"[INFO] classes({num_classes}): {classes}")

# ===== 3. 학습/테스트 분리 =====
X_train, X_test, y_train, y_test, y_train_idx, y_test_idx = train_test_split(
    X_data, y_onehot, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ===== 4. 모델 정의 (GRU) =====
model = Sequential(name="gesture_gru_classifier")
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
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# ===== 7. 평가 =====
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("테스트 정확도:", acc)

# ===== 8. 저장 =====
final_model_path = os.path.join(MODEL_DIR, "gesture_gru_final.keras")
model.save(final_model_path)
# 추론 스크립트에서 동일 매핑을 사용하도록, dataset의 인코더를 model 폴더에도 복사 저장
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
print(f"[INFO] 모델 저장: {final_model_path}")
print(f"[INFO] Best checkpoint: {ckpt_path}")
print(f"[INFO] LabelEncoder 저장: {os.path.join(MODEL_DIR, 'label_encoder.pkl')} (원본: {le_path})")

# ===== 9. 학습 곡선 & 혼동행렬 =====
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
plt.savefig(os.path.join(RESULT_DIR, "training_curve.png"))
plt.close()

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, xticks_rotation=45, cmap=None)   # 기본 컬러맵
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
plt.close()

print(f"[DONE] 결과 저장: {os.path.join(RESULT_DIR, 'training_curve.png')}, "
      f"{os.path.join(RESULT_DIR, 'confusion_matrix.png')}")
