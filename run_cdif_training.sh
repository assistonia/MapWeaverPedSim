#!/bin/bash

# CDIF Training Script for GPU A6000 Server
# CCTV-Diffusion 모델 학습 자동화 스크립트

set -e  # 에러 발생 시 즉시 종료

echo "🚀 CDIF (CCTV-Diffusion) 학습 시작!"
echo "================================"

# 환경 설정
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 디렉토리 생성
mkdir -p models
mkdir -p training_data_cdif
mkdir -p logs
mkdir -p validation_results

# 로그 파일 설정
LOG_FILE="logs/cdif_training_$(date +%Y%m%d_%H%M%S).log"
echo "📝 로그 파일: $LOG_FILE"

# GPU 상태 확인
echo "🔍 GPU 상태 확인..."
nvidia-smi | tee -a $LOG_FILE

# Python 환경 확인
echo "🐍 Python 환경 확인..."
python --version | tee -a $LOG_FILE
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}')" | tee -a $LOG_FILE

# 필요한 패키지 설치 (필요시)
echo "📦 패키지 의존성 확인..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q tqdm tensorboard matplotlib opencv-python
pip install -q numpy scipy scikit-learn

# 1단계: 데이터 수집 (필요시)
echo "📊 1단계: 데이터 수집 시작..."
if [ ! -f "training_data_cdif/cdif_training_data.json" ]; then
    echo "새로운 데이터 수집 시작..."
    python train_cdif.py --data_only 2>&1 | tee -a $LOG_FILE
else
    echo "기존 데이터 발견, 수집 단계 건너뛰기"
fi

# 2단계: 모델 학습
echo "🎓 2단계: CDIF 모델 학습 시작..."
python train_cdif.py 2>&1 | tee -a $LOG_FILE

# 학습 완료 확인
if [ -f "models/cdif_best.pth" ]; then
    echo "✅ 학습 완료! 최고 성능 모델 저장됨"
    
    # 3단계: 모델 검증
    echo "🔍 3단계: 모델 검증 시작..."
    
    # 단일 테스트
    echo "단일 테스트 실행..."
    python visualize_cdif_training.py \
        --model_path models/cdif_best.pth \
        --output_dir validation_results \
        --single_test 2>&1 | tee -a $LOG_FILE
    
    # 배치 검증
    echo "배치 검증 실행..."
    python visualize_cdif_training.py \
        --model_path models/cdif_best.pth \
        --output_dir validation_results \
        --num_tests 50 2>&1 | tee -a $LOG_FILE
    
    echo "✅ 검증 완료!"
    
else
    echo "❌ 학습 실패: 모델 파일을 찾을 수 없습니다"
    exit 1
fi

# 결과 요약
echo ""
echo "🎉 CDIF 학습 파이프라인 완료!"
echo "================================"
echo "📁 생성된 파일들:"
echo "  - models/cdif_best.pth (최고 성능 모델)"
echo "  - training_data_cdif/cdif_training_data.json (학습 데이터)"
echo "  - validation_results/ (검증 결과)"
echo "  - logs/ (학습 로그)"
echo "  - runs/ (TensorBoard 로그)"
echo ""
echo "📊 TensorBoard 실행 방법:"
echo "  tensorboard --logdir=runs --port=6006"
echo ""
echo "🔍 검증 결과 확인:"
echo "  ls -la validation_results/"
echo ""

# 디스크 사용량 확인
echo "💾 디스크 사용량:"
du -sh models/ training_data_cdif/ validation_results/ logs/ runs/ 2>/dev/null || echo "일부 디렉토리가 존재하지 않습니다"

echo "🎯 학습 완료!" 