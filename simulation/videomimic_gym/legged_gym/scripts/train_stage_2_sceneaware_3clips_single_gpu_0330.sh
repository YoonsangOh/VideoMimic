#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
GYM_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
CONDA_ROOT="/home/nas4_user/kyungminlee/anaconda3"

CONDA_ENV="${CONDA_ENV:-OpenHL}"
GPU_INDEX="${GPU_INDEX:-1}"
LOAD_RUN="${LOAD_RUN:-20250410_063030_g1_deepmimic}"
RUN_NAME="${RUN_NAME:-0330_stage2_sceneaware_3clips_random_mcpt}"
NUM_ENVS="${NUM_ENVS:-4096}"
HUMAN_SOURCE="${HUMAN_SOURCE:-resources/data_config/human_motion_list_3_random_0330.yaml}"

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
  --num_envs="${NUM_ENVS}" \
  --wandb_note="videomimic_stage_2_random_3clips_from_mcpt" \
  --env.terrain.n_rows=1 \
  --env.deepmimic.human_motion_source="${HUMAN_SOURCE}" \
  --load_run="${LOAD_RUN}" \
  --resume \
  --train.policy.re_init_std=True \
  --train.policy.init_noise_std=0.5 \
  --train.algorithm.learning_rate=2e-5 \
  --train.algorithm.schedule=fixed \
  --env.deepmimic.amass_terrain_difficulty=1 \
  --env.deepmimic.upsample_data=True \
  --env.deepmimic.use_human_videos=True \
  --env.deepmimic.link_pos_error_threshold=0.5 \
  --train.runner.save_interval=500 \
  --env.deepmimic.respawn_z_offset=0.1 \
  --env.deepmimic.randomize_terrain_offset=False \
  --env.terrain.cast_mesh_to_heightfield=False \
  --env.deepmimic.truncate_rollout_length=500 \
  --train.runner.load_model_strict=False \
  --env.deepmimic.use_amass=False \
  --env.rewards.scales.termination=-2000 \
  --env.rewards.scales.alive=200.0 \
  --env.rewards.scales.ankle_action=-3.0 \
  --env.rewards.scales.action_rate=-3.0
