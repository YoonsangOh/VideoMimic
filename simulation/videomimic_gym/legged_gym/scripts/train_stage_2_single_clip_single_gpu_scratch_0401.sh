#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
GYM_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
CONDA_ROOT="/home/nas4_user/kyungminlee/anaconda3"

CONDA_ENV="${CONDA_ENV:-OpenHL}"
GPU_INDEX="${GPU_INDEX:-2}"
RUN_NAME="${RUN_NAME:-0401_sc_single_clip}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERS="${MAX_ITERS:-30000}"
SEED="${SEED:-1}"
HUMAN_SOURCE="${HUMAN_SOURCE:?Set HUMAN_SOURCE to a single-clip YAML.}"
WANDB_NOTE="${WANDB_NOTE:-videomimic_stage_2_single_clip_scratch_0401}"
TERRAIN_N_ROWS="${TERRAIN_N_ROWS:-8}"
HUMAN_OVERSAMPLE_FACTOR="${HUMAN_OVERSAMPLE_FACTOR:-8}"

: "${WANDB_API_KEY:?Set WANDB_API_KEY before running this script.}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export TZ="${TZ:-Asia/Seoul}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONPATH="${GYM_ROOT}:${PYTHONPATH:-}"

python - <<'PY'
import os
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
PY

cd "${GYM_ROOT}"

python legged_gym/scripts/train.py \
  --task=g1_deepmimic_proj_heightfield \
  --headless \
  --run_name="${RUN_NAME}" \
  --seed="${SEED}" \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITERS}" \
  --wandb_note="${WANDB_NOTE}" \
  --env.terrain.n_rows="${TERRAIN_N_ROWS}" \
  --env.deepmimic.human_motion_source="${HUMAN_SOURCE}" \
  --train.policy.re_init_std=True \
  --train.policy.init_noise_std=0.5 \
  --train.algorithm.learning_rate=2e-5 \
  --train.algorithm.schedule=fixed \
  --env.deepmimic.amass_terrain_difficulty=1 \
  --env.deepmimic.upsample_data=True \
  --env.deepmimic.use_human_videos=True \
  --env.deepmimic.human_video_oversample_factor="${HUMAN_OVERSAMPLE_FACTOR}" \
  --env.deepmimic.link_pos_error_threshold=0.5 \
  --train.runner.save_interval=500 \
  --env.deepmimic.respawn_z_offset=0.1 \
  --env.deepmimic.randomize_terrain_offset=False \
  --env.terrain.cast_mesh_to_heightfield=False \
  --env.deepmimic.truncate_rollout_length=500 \
  --env.deepmimic.use_amass=False \
  --env.rewards.scales.termination=-2000 \
  --env.rewards.scales.alive=200.0 \
  --env.rewards.scales.ankle_action=-3.0 \
  --env.rewards.scales.action_rate=-3.0
