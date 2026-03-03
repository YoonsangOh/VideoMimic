#!/bin/bash

# Ladder2 강화학습 스크립트
# 사용법: bash train_ladder2.sh [GPU개수]

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR"

# GPU 개수 확인 (기본값: 1)
NUM_GPUS=${1:-1}

# Conda 환경 활성화 확인
if ! conda env list | grep -q "videomimic"; then
    echo "Warning: 'videomimic' conda environment not found."
    echo "Please create it first: conda create -n videomimic python=3.8"
    exit 1
fi

# 데이터 경로 확인
DATA_DIR="data/videomimic_captures/megahunter_megasam_reconstruction_results_ladder2_cam01_frame_100_700_subsample_2"
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Please create a symlink or copy the data first."
    echo "Run: ln -s /home/nas5/kyungminlee/VideoMimic/real2sim/demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_ladder2_cam01_frame_100_700_subsample_2 $DATA_DIR"
    exit 1
fi

# 필요한 파일 확인
if [ ! -f "$DATA_DIR/retarget_poses_g1.h5" ]; then
    echo "Error: retarget_poses_g1.h5 not found in $DATA_DIR"
    exit 1
fi

if [ ! -f "$DATA_DIR/background_mesh.obj" ]; then
    echo "Error: background_mesh.obj not found in $DATA_DIR"
    exit 1
fi

echo "Starting ladder2 training with $NUM_GPUS GPU(s)..."
echo "Using CUDA_VISIBLE_DEVICES=2"

# Set Python path and library path
PYTHON_BIN="/home/nas5/kyungminlee/anaconda3/envs/videomimic/bin/python"
export LD_LIBRARY_PATH=/home/nas5/kyungminlee/anaconda3/envs/videomimic/lib:$LD_LIBRARY_PATH
export PATH=/home/nas5/kyungminlee/anaconda3/envs/videomimic/bin:$PATH

# Set WandB API key
export WANDB_API_KEY="54d5951df3502e196e3a1b895e227e9969fc8e7b"

if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU
    CUDA_VISIBLE_DEVICES=2 $PYTHON_BIN videomimic_gym/legged_gym/scripts/train.py \
      --task=g1_deepmimic_proj_heightfield \
      --headless \
      --env.terrain.n_rows=1 \
      --num_envs=2048 \
      --wandb_note "ladder2_training" \
      --env.deepmimic.human_motion_source=resources/data_config/ladder2_motion.yaml \
      --env.deepmimic.use_human_videos=True \
      --env.deepmimic.use_amass=False \
      --load_run=20250410_063030_g1_deepmimic \
      --resume \
      --train.policy.re_init_std=True \
      --train.policy.init_noise_std=0.5 \
      --train.algorithm.learning_rate=2e-5 \
      --train.algorithm.schedule=fixed \
      --env.deepmimic.upsample_data=True \
      --env.deepmimic.link_pos_error_threshold=0.5 \
      --train.runner.save_interval=500 \
      --env.deepmimic.respawn_z_offset=0.1 \
      --env.deepmimic.randomize_terrain_offset=False \
      --env.terrain.cast_mesh_to_heightfield=False \
      --env.deepmimic.truncate_rollout_length=500 \
      --train.runner.load_model_strict=False \
      --env.rewards.scales.termination=-2000 \
      --env.rewards.scales.alive=200.0 \
      --env.rewards.scales.ankle_action=-3.0 \
      --env.rewards.scales.action_rate=-3.0
else
    # Multi GPU
    TORCHRUN_BIN="/home/nas5/kyungminlee/anaconda3/envs/videomimic/bin/torchrun"
    CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=/home/nas5/kyungminlee/anaconda3/envs/videomimic/lib:$LD_LIBRARY_PATH $TORCHRUN_BIN --nproc-per-node $NUM_GPUS videomimic_gym/legged_gym/scripts/train.py \
      --task=g1_deepmimic_proj_heightfield \
      --multi_gpu \
      --headless \
      --env.terrain.n_rows=1 \
      --num_envs=4096 \
      --wandb_note "ladder2_training" \
      --env.deepmimic.human_motion_source=resources/data_config/ladder2_motion.yaml \
      --env.deepmimic.use_human_videos=True \
      --env.deepmimic.use_amass=False \
      --load_run=20250410_063030_g1_deepmimic \
      --resume \
      --train.policy.re_init_std=True \
      --train.policy.init_noise_std=0.5 \
      --train.algorithm.learning_rate=2e-5 \
      --train.algorithm.schedule=fixed \
      --env.deepmimic.upsample_data=True \
      --env.deepmimic.link_pos_error_threshold=0.5 \
      --train.runner.save_interval=500 \
      --env.deepmimic.respawn_z_offset=0.1 \
      --env.deepmimic.randomize_terrain_offset=False \
      --env.terrain.cast_mesh_to_heightfield=False \
      --env.deepmimic.truncate_rollout_length=500 \
      --train.runner.load_model_strict=False \
      --env.rewards.scales.termination=-2000 \
      --env.rewards.scales.alive=200.0 \
      --env.rewards.scales.ankle_action=-3.0 \
      --env.rewards.scales.action_rate=-3.0
fi

