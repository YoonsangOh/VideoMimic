#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
GYM_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
CONDA_ROOT="/home/nas4_user/kyungminlee/anaconda3"

CONDA_ENV="${CONDA_ENV:-OpenHL}"
GPU_INDEX="${GPU_INDEX:-2}"
LOAD_RUN="${LOAD_RUN:-20250410_063030_g1_deepmimic}"
RUN_NAME="${RUN_NAME:-holosoma_stairs_heightfield_single_gpu}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERATIONS="${MAX_ITERATIONS:-30}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

: "${WANDB_API_KEY:?Set WANDB_API_KEY before running this script.}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

cd "${GYM_ROOT}"

python legged_gym/scripts/train.py \
  --task=g1_deepmimic_proj_heightfield \
  --headless \
  --run_name="${RUN_NAME}" \
  --load_run="${LOAD_RUN}" \
  --resume \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --wandb_note="holosoma stairs single-gpu heightfield finetune" \
  --env.deepmimic.human_motion_source=resources/data_config/holosoma_stairs_motion.yaml \
  --env.deepmimic.use_human_videos=True \
  --env.deepmimic.use_amass=False \
  --env.deepmimic.upsample_data=True \
  --env.deepmimic.link_pos_error_threshold=0.5 \
  --env.deepmimic.respawn_z_offset=0.1 \
  --env.deepmimic.randomize_terrain_offset=False \
  --env.deepmimic.truncate_rollout_length=500 \
  --env.deepmimic.amass_terrain_difficulty=1 \
  --env.terrain.n_rows=1 \
  --env.terrain.cast_mesh_to_heightfield=False \
  --train.runner.save_interval="${SAVE_INTERVAL}" \
  --train.runner.load_model_strict=False \
  --train.policy.re_init_std=True \
  --train.policy.init_noise_std=0.5 \
  --train.algorithm.learning_rate=2e-5 \
  --train.algorithm.schedule=fixed \
  --env.rewards.scales.termination=-2000 \
  --env.rewards.scales.alive=200.0 \
  --env.rewards.scales.ankle_action=-3.0 \
  --env.rewards.scales.action_rate=-3.0
