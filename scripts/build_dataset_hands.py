# build_dataset_hands.py — (T,F) 통합 빌더: 두 손 xy(84) 정규화 + 랜드마크 마스킹 지원
import os
import re
import json
import argparse
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import joblib

# =========================
# MediaPipe Hands 인덱스 (0~20)
# =========================
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

NAME2IDX = {
    "WRIST": WRIST,
    "THUMB_CMC": THUMB_CMC, "THUMB_MCP": THUMB_MCP, "THUMB_IP": THUMB_IP, "THUMB_TIP": THUMB_TIP,
    "INDEX_MCP": INDEX_MCP, "INDEX_PIP": INDEX_PIP, "INDEX_DIP": INDEX_DIP, "INDEX_TIP": INDEX_TIP,
    "MIDDLE_MCP": MIDDLE_MCP, "MIDDLE_PIP": MIDDLE_PIP, "MIDDLE_DIP": MIDDLE_DIP, "MIDDLE_TIP": MIDDLE_TIP,
    "RING_MCP": RING_MCP, "RING_PIP": RING_PIP, "RING_DIP": RING_DIP, "RING_TIP": RING_TIP,
    "PINKY_MCP": PINKY_MCP, "PINKY_PIP": PINKY_PIP, "PINKY_DIP": PINKY_DIP, "PINKY_TIP": PINKY_TIP,
}

# =========================
# (T, F) → 두 손 xy(84) 정규화
# =========================
def to_xy_only(x: np.ndarray) -> np.ndarray:
    """
    x: (T, F) with F in {42, 63, 84, 126}
    -> z를 제거해서 xy만 남김
    """
    T, F = x.shape
    if F == 42:   # 1 hand, xy
        return x
    if F == 63:   # 1 hand, xyz -> drop z
        return x.reshape(T, 21, 3)[:, :, :2].reshape(T, 42)
    if F == 84:   # 2 hands, xy
        return x
    if F == 126:  # 2 hands, xyz -> drop z
        return x.reshape(T, 2, 21, 3)[:, :, :, :2].reshape(T, 84)
    raise ValueError(f"지원하지 않는 피처 차원 F={F} (허용: 42/63/84/126)")

def pad_to_two_hands_xy(x: np.ndarray) -> np.ndarray:
    """
    x: (T, 42) or (T, 84) -> (T, 84)
    1손(42)이면 뒤 42를 0으로 패딩해서 2손(84)로 확장
    """
    T, F = x.shape
    if F == 84:
        return x
    if F == 42:
        pad = np.zeros((T, 42), dtype=x.dtype)
        return np.concatenate([x, pad], axis=1)
    raise ValueError(f"pad_to_two_hands_xy는 F=42/84만 허용, 현재 F={F}")

def canonicalize_sequence(arr: np.ndarray, targetF: int = 84) -> np.ndarray:
    """다양한 형상(42/63/84/126)을 받아 '두 손 x,y = 84차원'으로 변환."""
    x = to_xy_only(arr)        # z 제거
    x = pad_to_two_hands_xy(x) # 1손이면 84로 패딩
    assert x.shape[1] == targetF, f"정규화 후 F={x.shape[1]} (기대={targetF})"
    return x

# =========================
# 마스킹: (T,84)에서 특정 랜드마크의 (x,y)를 0(or 값)으로 설정
# =========================
def apply_landmark_mask_xy84(x: np.ndarray, mask_indices: list[int], mask_value: float = 0.0) -> np.ndarray:
    """
    x: (T,84), mask_indices: 0..20의 랜드마크 인덱스 리스트
    각 손 블록(42) 안에서 해당 랜드마크의 x,y 2차원을 mask_value로 설정
    """
    if x.shape[1] != 84:
        raise ValueError(f"apply_landmark_mask_xy84는 (T,84)만 지원, 현재 {x.shape}")
    if not mask_indices:
        return x
    x = x.copy()
    for hand_block in (0, 42):
        for lm in mask_indices:
            if 0 <= lm < 21:
                s = hand_block + lm * 2
                x[:, s:s+2] = mask_value
    return x

# =========================
# 파일/스캔 유틸
# =========================
NPY_PAT = re.compile(r".+\.npy$", re.IGNORECASE)

def list_label_dirs(data_dir: str):
    labels = []
    for name in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, name)
        if os.path.isdir(p):
            labels.append(name)
    return labels

def iter_npy_files(label_dir: str):
    for name in sorted(os.listdir(label_dir)):
        if NPY_PAT.match(name):
            yield os.path.join(label_dir, name)

def scan_shape_distribution(data_dir: str):
    dist = Counter()
    bad = []
    for label in list_label_dirs(data_dir):
        d = os.path.join(data_dir, label)
        for fp in iter_npy_files(d):
            try:
                arr = np.load(fp)
                if arr.ndim != 2:
                    bad.append((fp, f"ndim={arr.ndim}"))
                    continue
                dist[(arr.shape[0], arr.shape[1])] += 1
            except Exception as e:
                bad.append((fp, str(e)))
    return dist, bad

# =========================
# 메인 빌더
# =========================
def build_dataset(data_dir: str,
                  out_dir: str,
                  seq_len: int = 10,
                  targetF: int = 84,
                  strict_T: bool = True,
                  allow_trim_or_skip: str = "skip",  # "skip" | "trim_head" | "trim_tail" | "center"
                  mask_indices: list[int] | None = None,
                  mask_value: float = 0.0
                  ):
    os.makedirs(out_dir, exist_ok=True)
    mask_indices = mask_indices or []

    labels = list_label_dirs(data_dir)
    if not labels:
        raise RuntimeError(f"[ERROR] 데이터 폴더에 라벨 디렉토리가 없습니다: {data_dir}")

    print(f"[INFO] 라벨 폴더: {labels}")
    if mask_indices:
        print(f"[INFO] 랜드마크 마스킹 활성: {mask_indices} (각 손의 해당 (x,y) → {mask_value})")

    X_list = []
    y_list = []
    per_label_counts = defaultdict(int)
    skipped = 0

    for label in labels:
        label_dir = os.path.join(data_dir, label)
        files = list(iter_npy_files(label_dir))
        if not files:
            print(f"[WARN] '{label}'에 .npy 파일이 없습니다. 건너뜀.")
            continue

        pbar = tqdm(files, desc=f"load {label}", ncols=100)
        for fp in pbar:
            try:
                arr = np.load(fp)  # (T, Fvar)
                if arr.ndim != 2:
                    print(f"[SKIP] {fp} : 2차원 배열이 아닙니다. ndim={arr.ndim}")
                    skipped += 1
                    continue

                T, F = arr.shape
                if T != seq_len:
                    if strict_T and allow_trim_or_skip == "skip":
                        print(f"[SKIP] {fp} : T={T} != seq_len={seq_len}")
                        skipped += 1
                        continue
                    # 트리밍 전략
                    if T > seq_len:
                        if allow_trim_or_skip == "trim_head":
                            arr = arr[:seq_len, :]
                        elif allow_trim_or_skip == "trim_tail":
                            arr = arr[-seq_len:, :]
                        elif allow_trim_or_skip == "center":
                            s = (T - seq_len) // 2
                            arr = arr[s:s+seq_len, :]
                        else:
                            print(f"[SKIP] {fp} : T={T} (트리밍 전략 미설정)")
                            skipped += 1
                            continue
                    else:
                        # T < seq_len : 패딩은 권장하지 않음
                        print(f"[SKIP] {fp} : T={T} < seq_len={seq_len}")
                        skipped += 1
                        continue

                # ---- 핵심: (T,84) 정규화 ----
                x84 = canonicalize_sequence(arr, targetF=targetF)  # (T,84)

                # ---- 마스킹 적용 ----
                if mask_indices:
                    x84 = apply_landmark_mask_xy84(x84, mask_indices, mask_value)

                X_list.append(x84)
                y_list.append(label)
                per_label_counts[label] += 1

            except Exception as e:
                print(f"[SKIP] {fp} : {e}")
                skipped += 1

    if not X_list:
        raise RuntimeError("[ERROR] 유효한 샘플이 없습니다. 입력 데이터를 확인하세요.")

    X = np.stack(X_list, axis=0)  # (N, T, 84)
    y = np.array(y_list, dtype=object)

    # 라벨 인코딩
    le = LabelEncoder()
    y_enc = le.fit_transform(y)  # (N,)

    # 저장
    X_path = os.path.join(out_dir, "X.npy")
    y_path = os.path.join(out_dir, "y.npy")
    ytxt_path = os.path.join(out_dir, "y_text.npy")  # 가독성용
    le_path = os.path.join(out_dir, "label_encoder.pkl")
    cfg_path = os.path.join(out_dir, "dataset_config.json")

    np.save(X_path, X)
    np.save(y_path, y_enc)
    np.save(ytxt_path, y)
    joblib.dump(le, le_path)

    config = {
        "data_dir": os.path.abspath(data_dir),
        "out_dir": os.path.abspath(out_dir),
        "num_samples": int(X.shape[0]),
        "sequence_length": int(X.shape[1]),
        "feature_dim": int(X.shape[2]),                 # 84
        "labels": list(le.classes_),
        "per_label_counts": {k: int(v) for k, v in per_label_counts.items()},
        "skipped": int(skipped),
        "modality": "hands",
        "use_standardize": True,
        "use_z": False,
        "num_blocks": 2,                 # coords + (빈)패딩 블록 취급 아님(두 손 xy)
        "feature_dim_per_block": 21*2,   # 42
        "masked_landmarks": mask_indices,
        "mask_value": float(mask_value),
        "note": "X shape = (N, T, 84) with two-hands (x,y). 특정 랜드마크의 (x,y)를 양손 블록 모두 mask_value로 설정."
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n[SUMMARY]")
    print(f"- Saved X: {X.shape} -> {X_path}")
    print(f"- Saved y(enc): {y_enc.shape} -> {y_path}")
    print(f"- Saved y(text): {y.shape} -> {ytxt_path}")
    print(f"- Saved LabelEncoder -> {le_path}")
    print(f"- Saved config -> {cfg_path}")
    print(f"- Per-label counts: {config['per_label_counts']}")
    print(f"- Skipped files: {skipped}")
    if mask_indices:
        print(f"- Masked landmarks (0..20): {mask_indices}  (value={mask_value})")

# =========================
# 파서 & main
# =========================
def parse_mask_indices(mask_arg: str | None):
    """
    "--mask" 인자 파싱: 콤마로 구분된 이름/숫자 혼용 허용.
    예) "WRIST,THUMB_TIP,8" → [0,4,8]
    """
    if not mask_arg:
        return []
    items = [s.strip() for s in mask_arg.split(",") if s.strip()]
    idxs = []
    for it in items:
        it_upper = it.upper()
        if it_upper in NAME2IDX:
            idxs.append(NAME2IDX[it_upper])
        else:
            try:
                val = int(it)
                if 0 <= val <= 20:
                    idxs.append(val)
                else:
                    raise ValueError
            except Exception:
                raise ValueError(f"--mask 인자 해석 실패: '{it}' (이름 또는 0..20 정수 사용)")
    # 중복 제거 & 정렬
    return sorted(set(idxs))

def main():
    ap = argparse.ArgumentParser(description="Build hands dataset → (N,T,84) two-hands xy, with optional landmark masking.")
    ap.add_argument("--data-dir", default="../data", help="라벨 폴더들이 들어있는 루트 경로")
    ap.add_argument("--out-dir", default="../dataset", help="출력 저장 경로")
    ap.add_argument("--seq-len", type=int, default=10, help="시퀀스 길이 T")
    ap.add_argument("--targetF", type=int, default=84, help="피처 차원(기본 84: 두 손 xy)")
    ap.add_argument("--strict-T", action="store_true",
                    help="T가 seq-len과 다르면 스킵(기본 False: 트리밍 전략 적용 가능)")
    ap.add_argument("--trim", choices=["skip", "trim_head", "trim_tail", "center"],
                    default="skip", help="T>seq-len일 때 처리 방식 (기본 skip)")
    ap.add_argument("--mask", type=str, default=None,
                    help="마스킹할 랜드마크. 예: 'WRIST,THUMB_TIP,8' 또는 '0,4,8'")
    ap.add_argument("--mask-value", type=float, default=0.0,
                    help="마스킹 값(기본 0.0)")

    args = ap.parse_args()

    print(f"[INFO] 라벨 폴더 스캔 전 형상 분포를 훑습니다... ({args.data_dir})")
    dist, bad = scan_shape_distribution(args.data_dir)
    if dist:
        print("[INFO] 발견된 (T, F) 분포:", dict(dist))
    if bad:
        print(f"[WARN] 읽기 실패 파일 {len(bad)}개 (상위 5개만 표시):")
        for fp, msg in bad[:5]:
            print("   -", fp, ":", msg)

    mask_indices = parse_mask_indices(args.mask)

    build_dataset(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        seq_len=args.seq_len,
        targetF=args.targetF,
        strict_T=args.strict_T,
        allow_trim_or_skip=args.trim,
        mask_indices=mask_indices,
        mask_value=args.mask_value
    )

if __name__ == "__main__":
    main()
