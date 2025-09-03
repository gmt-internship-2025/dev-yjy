# 🖐️ MediaPipe 기반 제스처 인식 프로젝트

> 실시간 웹캠 입력으로 사람의 포즈를 인식하고,
> 직접 수집한 제스처 데이터를 기반으로 딥러닝(GRU) 분류기를 통해 동작을 분류하는 프로젝트입니다.

---
mediapipe-gesture (Hands + GRU)

MediaPipe Hands로 손 랜드마크를 추출하고, GRU 모델로 제스처를 분류합니다.
파이프라인은 데이터 수집 → 데이터셋 빌드 → 모델 학습 → 실시간 추론(API/영상) 순서로 진행됩니다.

디렉터리 구조
```
mediapipe-gesture/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ dataset_config.json          # 입력 레이아웃/스케일링 설정
│  ├─ default/                     # negative(기본) 클래스 시퀀스
│  └─ <label>/seq_###.npy ...      # 라벨별 시퀀스(collect_data.py가 생성)
├─ model/
│  ├─ gesture_gru_best.keras       # 최고 성능 가중치(콜백 저장)
│  ├─ gesture_gru_final.keras      # 최종 모델
│  └─ label_encoder.pkl            # 라벨 인코더
├─ result/
│  ├─ history.png                  # 학습 곡선(정확도/손실)
│  └─ metrics.json                 # 성능 요약
└─ scripts/
   ├─ collect_data.py              # (학습용) 제스처 시퀀스 캡처
   ├─ build_dataset_hands.py       # (학습용) 수집본으로 학습 데이터셋 빌드
   ├─ train_model_gru.py           # (학습용) GRU 모델 학습
   ├─ live_inference_gru.py        # (실시간) 카메라 스트림 추론/시각화
   └─ sampleAPI.py                 # (실시간) API 서버용 샘플 엔드포인트
```
### 가상환경(venv) 구성
| Python 3.10.x 권장 (호환성 최적)

1) 생성/활성화
```
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows PowerShell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2) 패키지 설치
```
pip install --upgrade pip
pip install -r requirements.txt
```
3) 현재 환경 내용 확인(버전 점검)
```
pip list
python -c "import cv2, mediapipe, tensorflow as tf; \
print('cv2:', cv2.__version__); \
print('mediapipe:', mediapipe.__version__); \
print('tensorflow:', tf.__version__)"
```
(참고) 본 프로젝트 기준 핵심 요구사항

`requirements.txt` 예시:
```
numpy==1.26.4
pandas==2.3.1
scipy==1.15.3
scikit-learn==1.7.1
joblib==1.5.1

tensorflow==2.19.0
keras==3.11.1

opencv-python==4.9.0.80
opencv-contrib-python==4.11.0.86
mediapipe==0.10.18

matplotlib==3.10.3
tqdm==4.67.1
sounddevice==0.5.2
```
---
### 학습 과정(Training Pipeline)

1) 데이터 수집 — collect_data.py

스크립트 내 LABEL = "stop" 과 같이 라벨 이름을 직접 지정해야 합니다.
실행 시 카메라 프레임을 받아 연속된 10프레임 시퀀스를 data/<label>/seq_###.npy로 저장합니다.

```
python scripts/collect_data.py 
# 라벨은 스크립트 상단 변수(LABEL = "...")를 수정해서 지정
```

2) 데이터셋 빌드 — build_dataset_hands.py

수집된 시퀀스(NPY)를 모아 학습용 배열/라벨 인코딩을 생성합니다.
결과는 data/ 하위의 표준 구조(예: X_train.npy, y_train.npy 형태 또는 스크립트 내 저장 규칙)에 맞게 저장됩니다.
```
python build_dataset_hands.py 

또는

python build_dataset_hands.py --mask "2, 3, 4, 6, 10, 14, 18"
숫자를 통해 마스킹할 손의 point를 지정합니다.
```
dataset_config.json의 레이아웃(coords, coords+delta, two_hands_xy 등)과 스케일링 옵션이 적용됩니다.

3) 모델 학습 — train_model_gru.py

GRU 기반 분류 모델을 학습하고 model/에 가중치와 라벨 인코더를 저장합니다.

```
python train_model_gru.py 
```

출력물

model/gesture_gru_best.keras / model/gesture_gru_final.keras
model/label_encoder.pkl
result/history.png, result/metrics.json

---
### 실시간 추론(Real-Time Inference)
A) API용 — sampleAPI.py

간단한 API 형태로 모델을 로드하여 인식한 motion의 label을 터미널에 출력합니다.

```
python scripts/sampleAPI.py 
```

B) 영상 실시간 확인 — live_inference_gru.py

웹캠 스트림을 입력 받아, 예측 라벨/확률을 영상 오버레이로 확인합니다.

```
python ive_inference_gru.py
```
