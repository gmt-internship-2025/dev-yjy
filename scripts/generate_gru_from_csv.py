# build_xy_from_seqs.py
import os
import json
import numpy as np
from glob import glob
from collections import defaultdict

DATA_DIR = "../data"
CONFIG_PATH = os.path.join(DATA_DIR, "dataset_config.json")
X_OUT = os.path.join(DATA_DIR, "X_gru.npy")
Y_OUT = os.path.join(DATA_DIR, "y_gru.npy")

# 1) config 읽어서 (T, F) 고정 (없으면 첫 샘플로 추론)
SEQ_LEN = None
FEAT_DIM = None
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    SEQ_LEN = int(cfg.get("sequence_length", 10))
    FEAT_DIM = int(cfg.get("feature_dim", 66))
    print(f"[INFO] config: sequence_length={SEQ_LEN}, feature_dim={FEAT_DIM}")
else:
    print("[WARN] dataset_config.json 없음. 첫 샘플 shape로 (T,F) 추론합니다.")

# 2) 라벨 폴더 탐색 (seq_*.npy가 있는 폴더만 라벨로 간주)
subdirs = [
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith(".")
]

label_dirs = []
for d in sorted(subdirs):
    seq_files = glob(os.path.join(DATA_DIR, d, "seq_*.npy"))
    if seq_files:
        label_dirs.append((d, sorted(seq_files)))

if not label_dirs:
    raise RuntimeError("라벨 폴더를 찾지 못했습니다. ../data/<LABEL>/seq_*.npy 형태인지 확인하세요.")

print(f"[INFO] 감지된 라벨 수: {len(label_dirs)}  -> {[ld[0] for ld in label_dirs]}")

# 3) 로딩 & 검증 & 누적
X_list, y_list = [], []
per_label_count = defaultdict(int)

for label, files in label_dirs:
    for path in files:
        arr = np.load(path)  # 예상: (T, F)
        if arr.ndim != 2:
            print(f"[SKIP] 2D가 아닌 샘플: {path}, shape={arr.shape}")
            continue

        # (T,F) 확정/검증
        if SEQ_LEN is None or FEAT_DIM is None:
            SEQ_LEN, FEAT_DIM = arr.shape
            print(f"[INFO] 첫 샘플로 (T,F) 추론: ({SEQ_LEN}, {FEAT_DIM})")
        if arr.shape != (SEQ_LEN, FEAT_DIM):
            print(f"[SKIP] shape 불일치: {path}  got={arr.shape}, want={(SEQ_LEN, FEAT_DIM)}")
            continue

        X_list.append(arr.astype(np.float32))
        y_list.append(label)
        per_label_count[label] += 1

# 4) 스택 & 저장
if not X_list:
    raise RuntimeError("유효한 시퀀스를 하나도 모으지 못했습니다.")

X = np.stack(X_list, axis=0)        # (N, T, F)
y = np.array(y_list, dtype=object)  # (N,) 문자열 라벨

np.save(X_OUT, X)
np.save(Y_OUT, y)

print(f"[DONE] X -> {X_OUT}  shape={X.shape}, dtype={X.dtype}")
print(f"[DONE] y -> {Y_OUT}  shape={y.shape}, dtype={y.dtype}")

# 라벨별 개수 출력
for k in sorted(per_label_count.keys()):
    print(f"  - {k}: {per_label_count[k]}")
