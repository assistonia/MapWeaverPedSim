# DiPPeR-Social: Diffusion-based Path Planning with Social Awareness

CGIP(Context-based Intention-guided Planning) 프레임워크에 DiPPeR(Diffusion-based Path Planning) 모델을 통합한 사회적 인식 로봇 네비게이션 시스템입니다.

## 🚀 주요 기능

- **DiPPeR-Social**: 사회적 상호작용을 고려한 Diffusion 기반 경로 계획
- **3가지 경로 스타일**: A* 기본, 사회적 비용 고려, 우회 경로 혼합 학습
- **DDPM 기반 생성**: 1000 타임스텝 노이즈 제거로 고품질 경로 생성
- **안전성 보장**: 3단계 검증 시스템 (학습/입력/추론)
- **실시간 동적 계획**: 보행자 움직임에 따른 실시간 경로 재계획
- **CCTV 기반 모니터링**: 특정 영역 내 보행자만 추적

## 🔄 **실행 순서 (처음 사용자)**

### **1단계: 데이터 수집 및 학습** (처음 한 번만)
```bash
# 1-1. 학습 데이터 수집 + 모델 학습 (한 번에)
python train_dipperp.py \
    --xml_file scenarios/Circulation1.xml \
    --num_episodes 200 \
    --epochs 100 \
    --batch_size 16 \
    --save_data training_data.json \
    --model_save_path dipperp_model.pth
```

### **2단계: 데이터셋 확인** (선택사항)
```bash
# 2-1. 대화형 데이터 뷰어
python visualize_dataset.py --data_path training_data.json --mode interactive

# 2-2. 빠른 분석 (5개 샘플 + 통계)
python visualize_dataset.py --data_path training_data.json --mode analyze

# 2-3. 특정 샘플 확인
python visualize_dataset.py --data_path training_data.json --mode sample --sample_idx 100
```

### **3단계: 학습된 모델 사용**
```bash
# 3-1. DiPPeR 모델로 시뮬레이션
python robot_simuator_dippeR.py scenarios/Circulation1.xml models/dipperp_model_best.pth

# 3-2. 성능 평가
python evaluate_dipperp.py --model_path models/dipperp_model_best.pth --xml_file scenarios/Circulation1.xml

# 3-3. 대화형 추론 테스트
python inference_dipperp.py --model_path models/dipperp_model_best.pth
```

## 🔄 **실행 순서 (기존 데이터 활용)**

### **이미 데이터가 있는 경우:**
```bash
# 1. 기존 데이터로 모델만 학습
python train_dipperp.py \
    --load_data training_data.json \
    --epochs 100 \
    --batch_size 16 \
    --model_save_path dipperp_model_v2.pth

# 2. 학습된 모델 사용
python robot_simuator_dippeR.py scenarios/Circulation1.xml models/dipperp_model_v2_best.pth
```

## 📁 파일 구조

```
MapWeaverPedSim/
├── 🔥 핵심 실행 파일
│   ├── train_dipperp.py              # 1️⃣ 데이터 수집 + 모델 학습
│   ├── visualize_dataset.py          # 2️⃣ 데이터셋 시각적 확인
│   ├── robot_simuator_dippeR.py      # 3️⃣ DiPPeR 시뮬레이션 실행
│   ├── evaluate_dipperp.py           # 4️⃣ 모델 성능 평가
│   └── inference_dipperp.py          # 5️⃣ 대화형 추론 테스트
├── 📊 평가 및 분석
│   ├── evaluate_integration.py       # 통합 시스템 평가
│   └── run_evaluation.py            # 자동화 평가 실행
├── 🗺️ 시나리오 파일
│   └── scenarios/
│       ├── Circulation1.xml         # 순환 패턴 맵
│       ├── Circulation2.xml         # 복잡한 순환 맵
│       ├── Congestion1.xml          # 혼잡 환경 맵
│       └── Congestion2.xml          # 고밀도 혼잡 맵
├── 📦 생성 파일
│   ├── training_data.json           # 학습 데이터셋
│   ├── models/                      # 학습된 모델들
│   └── evaluation_results/          # 평가 결과들
└── 📋 설정 파일
    ├── requirements.txt             # Python 패키지
    └── README.md                    # 이 파일
```

## 🔧 **설치 및 설정**

### **Conda 환경 설정** (권장)
```bash
# 1. Conda 환경 생성 및 활성화
conda create -n dipperp python=3.9
conda activate dipperp

# 2. 필수 패키지 설치
pip install -r requirements.txt

# 3. PyTorch 설치 (GPU 사용 시)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 또는 CPU만 사용하는 경우
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### **pip 환경 설정**
```bash
# 1. 가상환경 생성 (선택사항)
python -m venv dipperp_env
source dipperp_env/bin/activate  # Linux/Mac
# dipperp_env\Scripts\activate  # Windows

# 2. 패키지 설치
pip install -r requirements.txt
```

### **설치 확인**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

## 🎯 **빠른 시작 가이드**

### **처음 사용자 (3단계만!)**
```bash
# 1️⃣ 데이터 수집 + 학습 (한 번에)
python train_dipperp.py --xml_file scenarios/Circulation1.xml --num_episodes 200 --epochs 100 --save_data training_data.json

# 2️⃣ 데이터 확인 (선택사항)
python visualize_dataset.py --mode interactive

# 3️⃣ DiPPeR 시뮬레이션 실행
python robot_simuator_dippeR.py scenarios/Circulation1.xml models/dipperp_model_best.pth
```

### **기존 사용자 (데이터 재활용)**
```bash
# 기존 데이터로 재학습
python train_dipperp.py --load_data training_data.json --epochs 100

# 바로 실행
python robot_simuator_dippeR.py scenarios/Circulation1.xml models/dipperp_model_best.pth
```

## 📊 **데이터셋 시각화 도구**

### **대화형 뷰어** (추천!)
```bash
python visualize_dataset.py --mode interactive
```
**명령어:**
- `숫자`: 특정 샘플 보기 (예: `42`)
- `random`: 랜덤 샘플 보기
- `type astar`: A* 경로 샘플 보기
- `type social`: 사회적 비용 경로 보기
- `type detour`: 우회 경로 보기
- `stats`: 데이터셋 통계
- `dist`: 분포 시각화
- `quit`: 종료

### **빠른 분석**
```bash
# 5개 샘플 + 통계 + 분포 시각화
python visualize_dataset.py --mode analyze --num_samples 5

# 특정 샘플만 확인
python visualize_dataset.py --mode sample --sample_idx 100
```

## 🎯 학습 과정

### 1단계: 데이터 수집
- CGIP 시뮬레이션에서 A* 경로를 정답으로 수집
- Fused Cost Map과 (시작점, 목표점, 경로) 쌍 생성
- 다양한 시나리오에서 충분한 데이터 확보

### 2단계: 모델 학습
- DiPPeR Diffusion 모델 학습
- Noise Prediction Network 훈련
- 정답 A* 경로를 목표로 Denoising 과정 학습

### 3단계: 추론 및 평가
- 학습된 모델로 실시간 경로 생성
- A* 대비 성능 비교
- 다양한 시나리오에서 안정성 검증

## 🔄 알고리즘 플로우

```
1. CGIP Fused Cost Map 생성
   ├── 정적 장애물 맵
   ├── Individual Space 계산
   └── 의도 정렬 기반 사회적 비용

2. DiPPeR 경로 계획
   ├── Cost Map → 이미지 변환
   ├── Diffusion 모델 추론
   └── 경로 생성

3. 폴백 A* (DiPPeR 실패 시)
   ├── 장애물 회피
   ├── 사회적 비용 고려
   └── 안전한 경로 보장
```

## ⚙️ 주요 매개변수

### DiPPeR 모델
- `visual_feature_dim`: 512 (Visual Encoder 출력 차원)
- `path_dim`: 2 (2D 경로)
- `max_timesteps`: 1000 (Diffusion 스텝 수)
- `path_length`: 50 (생성 경로 길이)

### CGIP 파라미터
- `gamma1`: 0.5 (사회적 비용 가중치)
- `is_threshold`: exp(-4) (Individual Space 임계값)
- `grid_size`: 0.2m (그리드 해상도)

## 🐛 문제 해결

### 1. "mat1 and mat2 must have the same dtype" 오류
```python
# 수정됨: dtype=torch.float32 명시적 지정
start_normalized = torch.tensor([[x/6.0, y/6.0]], dtype=torch.float32, device=device)
```

### 2. 로봇이 벽을 넘어가는 문제
```python
# 수정됨: is_position_safe() 함수로 안전성 체크
if self.is_position_safe(new_pos):
    self.robot_pos = new_pos
```

### 3. 학습 데이터 부족
- `--num_episodes` 값을 늘려서 더 많은 데이터 수집
- 다양한 맵 파일(`Circulation1.xml`, `Congestion2.xml` 등) 사용

## 📊 성능 지표

- **성공률**: 목표점 도달 성공 비율
- **평균 시간**: 목표점까지 도달 시간
- **평균 스텝**: 목표점까지 필요한 스텝 수
- **경로 품질**: 장애물 회피 및 사회적 비용 최소화

## 🔬 실험 시나리오

### 기본 맵들
- `Congestion1.xml`: 혼잡한 실내 환경
- `Circulation1.xml`: 순환 패턴 보행자
- `Circulation2.xml`: 복잡한 순환 패턴

### 테스트 케이스
1. **정적 장애물 회피**: 기본 장애물 환경
2. **동적 보행자 회피**: 움직이는 보행자 환경
3. **혼잡 환경**: 다수 보행자가 있는 복잡한 환경

## 📚 참고 문헌

- DiPPeR: Diffusion-based Path Planning in Robot Navigation
- CGIP: Context-based Intention-guided Planning framework
- Individual Space: 보행자 개인 공간 모델링 