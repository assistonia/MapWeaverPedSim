# DiPPeR: Diffusion-based Path Planning in Robot Navigation

CGIP(Context-based Intention-guided Planning) 프레임워크에 DiPPeR(Diffusion-based Path Planning) 모델을 통합한 로봇 네비게이션 시스템입니다.

## 🚀 주요 기능

- **CGIP 프레임워크**: Individual Space와 의도 정렬 기반 사회적 비용 계산
- **DiPPeR 통합**: A* 알고리즘을 Diffusion 모델로 대체
- **실시간 동적 계획**: 보행자 움직임에 따른 실시간 경로 재계획
- **CCTV 기반 모니터링**: 특정 영역 내 보행자만 추적

## 📁 파일 구조

```
MapWeaverPedSim/
├── robot_simuator_dippeR.py    # 메인 시뮬레이션 코드
├── train_dipperp.py            # DiPPeR 모델 학습
├── inference_dipperp.py        # 학습된 모델로 추론
├── requirements.txt            # 필요한 패키지 목록
├── Congestion1.xml            # 시뮬레이션 맵 파일
└── README_DiPPeR.md           # 사용법 (이 파일)
```

## 🔧 설치 및 설정

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. PyTorch 설치 확인
CUDA가 있는 경우:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

CPU만 사용하는 경우:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 📖 사용법

### 1. 기본 시뮬레이션 실행 (A* 폴백 모드)
```bash
python robot_simuator_dippeR.py
```

### 2. 다른 맵 파일 사용
```bash
python robot_simuator_dippeR.py Circulation1.xml
```

### 3. 데이터 수집 및 모델 학습

#### 데이터 수집 (시각화 포함)
```bash
python train_dipperp.py --num_episodes 100 --visualize --save_data training_data.json
```

#### 데이터 수집 (시각화 없음, 빠른 수집)
```bash
python train_dipperp.py --num_episodes 500 --save_data training_data.json
```

#### 모델 학습
```bash
python train_dipperp.py --load_data training_data.json --epochs 100 --batch_size 16
```

### 4. 학습된 모델로 추론

#### 대화형 모드
```bash
python inference_dipperp.py --model_path dipperp_model.pth
```

#### 성능 벤치마크
```bash
python inference_dipperp.py --model_path dipperp_model.pth --mode benchmark --num_tests 20
```

#### 단일 목표점 테스트
```bash
python inference_dipperp.py --model_path dipperp_model.pth --target_x 4.0 --target_y 3.0
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