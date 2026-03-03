#!/bin/bash

# 설치 확인 스크립트

echo "=== VideoMimic 설치 확인 ==="
echo ""

# Conda 환경 확인
if conda env list | grep -q "videomimic"; then
    echo "✓ videomimic conda 환경 존재"
else
    echo "✗ videomimic conda 환경 없음"
    exit 1
fi

# Conda 환경 활성화
eval "$(conda shell.bash hook)"
conda activate videomimic

echo ""
echo "=== Python 패키지 확인 ==="

# PyTorch 확인
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>/dev/null || echo "✗ PyTorch 없음"

# Isaac Gym 확인
python -c "import isaacgym; print(f'✓ Isaac Gym 설치됨: {isaacgym.__file__}')" 2>/dev/null || echo "⚠ Isaac Gym import 오류 (설치는 되어 있음)"

# videomimic_rl 확인
python -c "import rsl_rl; print('✓ videomimic_rl (rsl_rl) 설치됨')" 2>/dev/null || echo "✗ videomimic_rl 없음"

# videomimic_gym 확인
python -c "import legged_gym; print('✓ videomimic_gym 설치됨')" 2>/dev/null || echo "✗ videomimic_gym 없음"

echo ""
echo "=== 데이터 확인 ==="

DATA_DIR="data/videomimic_captures/megahunter_megasam_reconstruction_results_ladder2_cam01_frame_100_700_subsample_2"
if [ -d "$DATA_DIR" ]; then
    echo "✓ 데이터 디렉토리 존재"
    if [ -f "$DATA_DIR/retarget_poses_g1.h5" ]; then
        echo "✓ retarget_poses_g1.h5 존재"
    else
        echo "✗ retarget_poses_g1.h5 없음"
    fi
    if [ -f "$DATA_DIR/background_mesh.obj" ]; then
        echo "✓ background_mesh.obj 존재"
    else
        echo "✗ background_mesh.obj 없음"
    fi
else
    echo "✗ 데이터 디렉토리 없음"
fi

echo ""
echo "=== 설정 파일 확인 ==="

if [ -f "videomimic_gym/resources/data_config/ladder2_motion.yaml" ]; then
    echo "✓ ladder2_motion.yaml 존재"
else
    echo "✗ ladder2_motion.yaml 없음"
fi

echo ""
echo "=== 완료 ==="

