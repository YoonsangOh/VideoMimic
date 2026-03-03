#!/bin/bash

# Ladder2 학습 결과 재생 스크립트
# 사용법: bash play_ladder2.sh [RUN_NAME] [CHECKPOINT]

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"

RUN_NAME=${1:-""}
CHECKPOINT=${2:-""}

if [ -z "$RUN_NAME" ]; then
    echo "Usage: bash play_ladder2.sh [RUN_NAME] [CHECKPOINT]"
    echo "Example: bash play_ladder2.sh 20250121_120000_ladder2_training 1000"
    echo ""
    echo "Available runs:"
    ls -1 videomimic_gym/logs/g1_deepmimic/ 2>/dev/null | tail -5
    exit 1
fi

# Conda 환경 활성화 확인
if ! conda env list | grep -q "videomimic"; then
    echo "Warning: 'videomimic' conda environment not found."
    exit 1
fi

echo "Playing ladder2 with run: $RUN_NAME"
if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
fi

# Set Python path and library path for Isaac Gym
PYTHON_BIN="/home/nas5/kyungminlee/anaconda3/envs/videomimic/bin/python"
export LD_LIBRARY_PATH=/home/nas5/kyungminlee/anaconda3/envs/videomimic/lib:$LD_LIBRARY_PATH
export PATH=/home/nas5/kyungminlee/anaconda3/envs/videomimic/bin:$PATH

# Set WandB API key
export WANDB_API_KEY="54d5951df3502e196e3a1b895e227e9969fc8e7b"

# Use g1_deepmimic instead of g1_deepmimic_proj_heightfield for compatibility with pretrained model
CMD="$PYTHON_BIN videomimic_gym/legged_gym/scripts/play.py \
  --task=g1_deepmimic \
  --num_envs=1 \
  --headless \
  --env.viser.enable=True \
  --load_run=$RUN_NAME \
  --train.runner.load_model_strict=False"

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint=$CHECKPOINT"
fi

CMD="$CMD --env.deepmimic.human_motion_source=resources/data_config/ladder2_motion.yaml \
  --env.deepmimic.use_human_videos=True \
  --env.deepmimic.use_amass=False"

echo "Command: $CMD"
echo ""
echo "Viser will be available at http://localhost:8080"
echo "Press Ctrl+C to stop"

eval $CMD

