# 🖐️ MediaPipe 기반 제스처 인식 프로젝트

> 실시간 웹캠 입력으로 사람의 포즈를 인식하고,
> 직접 수집한 제스처 데이터를 기반으로 머신러닝(KNN) 분류기를 통해 동작을 분류하는 프로젝트입니다.

---

## 📁 프로젝트 구조

mediapipe-gesture/
├── data/
│ └── pose_data.csv # 제스처 학습용 landmark 데이터
├── model/
│ └── gesture_model.pkl # 학습된 KNN 모델
├── scripts/
│ ├── collect_data.py # 제스처 데이터 수집 (MediaPipe Pose)
│ ├── train_model.py # KNN 모델 학습
│ └── live_inference.py # 실시간 제스처 추론 및 스켈레톤 시각화
└── mediapipe/ # Google MediaPipe 공식 레포 (참고용)

---

## ✅ 주요 기능

- MediaPipe로 33개 포즈 landmark 추출
- 제스처별 데이터 수집 (ex. hands_up, sit, stand 등)
- KNN 알고리즘으로 사용자 정의 제스처 분류기 학습
- 실시간 웹캠 기반 제스처 인식
- 화면에 스켈레톤(관절) 및 예측 결과 시각화

---

## ⚙️ 실행 환경

- Python 3.10
- Ubuntu / Jetson Orin Nano 기반
- 가상환경 사용 권장: `python -m venv motion-env`

### 📦 필수 패키지 (권장 버전)
mediapipe==0.10.9
opencv-python==4.9.0.80
numpy==1.24.4
pandas==2.0.3
scikit-learn==1.3.0
joblib==1.3.2

---

## 🧪 사용 방법

### 1. 제스처 데이터 수집
`python collect_data.py`
- LABEL = "hands_up" 등으로 제스처 이름을 직접 지정
- 실시간 포즈를 수집하고 pose_data.csv에 저장

### 2. 모델 학습
`python train_model.py`
- 수집된 데이터를 기반으로 KNN 분류기를 학습
- gesture_model.pkl로 저장

### 3. 실시간 추론 실행
`python live_inference.py`
- 웹캠을 통해 제스처를 인식하고, 결과와 스켈레톤을 화면에 출력

### 📌 MediaPipe 사용 이유
- 사전 학습된 포즈 추정기로 별도 학습 없이 관절 위치 추출 가능
- 고정밀 landmark + 실시간 처리 가능
- 좌표 기반 입력으로 머신러닝과 결합하기 용이

### 💡 향후 확장 가능성
- 시계열 기반 딥러닝(GRU, LSTM) 모델로 동작 분류 강화
- 낙상 감지 / 손짓 제어 등 응용 서비스 개발
- GCN 기반 구조적 포즈 인식 연구

### 🙋 프로젝트 목적
- MediaPipe 기반 제스처 데이터 수집 및 모델링 실습
- 라벨링, 학습, 추론까지의 전체 파이프라인 경험
- Edge AI 기기(Jetson Orin 등)에서의 실시간 동작 인식 가능성 탐색

