import os
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = "../data"
SEQUENCE_LENGTH = 10
AUG_PER_SAMPLE = 3  # 증강 횟수

def normalize_landmarks(sequence):
    normalized_seq = []
    for frame in sequence:
        frame = frame.reshape(-1, 2)  # (33, 2)
        center = frame[0]  # nose 기준
        frame -= center

        left_shoulder = frame[11]
        right_shoulder = frame[12]
        dist = np.linalg.norm(left_shoulder - right_shoulder)

        if dist > 0:
            frame /= dist

        normalized_seq.append(frame.flatten())
    return np.array(normalized_seq)  # (10, 66)

def add_noise(sequence, std=0.01):
    noise = np.random.normal(0, std, sequence.shape)
    return sequence + noise

def time_warp(sequence):
    """
    프레임을 중복 또는 삭제하여 시퀀스 왜곡 (길이는 10 유지)
    """
    warped = sequence.copy()
    if np.random.rand() < 0.5:
        # 중복
        idx = np.random.randint(0, SEQUENCE_LENGTH)
        warped = np.insert(warped, idx, warped[idx], axis=0)
    else:
        # 삭제
        if SEQUENCE_LENGTH > 1:
            idx = np.random.randint(0, SEQUENCE_LENGTH)
            warped = np.delete(warped, idx, axis=0)

    # 길이 조정
    if warped.shape[0] < SEQUENCE_LENGTH:
        pad = np.tile(warped[-1], (SEQUENCE_LENGTH - warped.shape[0], 1))
        warped = np.vstack([warped, pad])
    elif warped.shape[0] > SEQUENCE_LENGTH:
        warped = warped[:SEQUENCE_LENGTH]

    return warped

# ===== 실행 =====
all_data = []
all_labels = []

for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue

    for fname in tqdm(os.listdir(label_path), desc=f"Processing {label}"):
        if not fname.endswith(".npy"):
            continue

        filepath = os.path.join(label_path, fname)
        sequence = np.load(filepath)  # shape: (10, 66) or (10, 33, 2)

        # 33x2 → 66 평탄화
        if sequence.ndim == 3:
            sequence = sequence.reshape(SEQUENCE_LENGTH, -1)

        if sequence.shape != (SEQUENCE_LENGTH, 66):
            continue  # skip invalid

        # 정규화
        norm_seq = normalize_landmarks(sequence)
        all_data.append(norm_seq.flatten())
        all_labels.append(label)

        # 증강
        for _ in range(AUG_PER_SAMPLE):
            aug_seq = add_noise(norm_seq)
            aug_seq = time_warp(aug_seq)
            all_data.append(aug_seq.flatten())
            all_labels.append(label)

# ===== CSV 저장 =====
df = pd.DataFrame(all_data)
df["label"] = all_labels
os.makedirs("../data", exist_ok=True)
df.to_csv("../data/pose_data_augmented.csv", index=False)
print("정규화 + 증강 완료 → pose_data_augmented.csv 저장됨")
