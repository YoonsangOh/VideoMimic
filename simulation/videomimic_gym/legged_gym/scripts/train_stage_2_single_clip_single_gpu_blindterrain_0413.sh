#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
GYM_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
CONDA_ROOT="/home/nas4_user/kyungminlee/anaconda3"

CONDA_ENV="${CONDA_ENV:-OpenHL}"
GPU_INDEX="${GPU_INDEX:-3}"
LOAD_RUN="${LOAD_RUN:-20250410_063030_g1_deepmimic}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERS="${MAX_ITERS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

: "${WANDB_API_KEY:?Set WANDB_API_KEY before running this script.}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export TZ="${TZ:-Asia/Seoul}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONPATH="${GYM_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb_videomimic_blindterrain_0413}"

mkdir -p "${WANDB_DIR}"

python - <<'PY'
import os
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
PY

cd "${GYM_ROOT}"

run_clip() {
  local run_name="$1"
  local wandb_note="$2"
  local human_source="$3"
  local terrain_n_rows="$4"
  local oversample_factor="$5"

  echo "[Sequence] Starting ${run_name} using ${human_source} on GPU ${GPU_INDEX}"

  python legged_gym/scripts/train.py \
    --task=g1_deepmimic_proj_heightfield \
    --headless \
    --run_name="${run_name}" \
    --num_envs="${NUM_ENVS}" \
    --max_iterations="${MAX_ITERS}" \
    --wandb_note="${wandb_note}" \
    --env.terrain.n_rows="${terrain_n_rows}" \
    --env.deepmimic.human_motion_source="${human_source}" \
    --load_run="${LOAD_RUN}" \
    --resume \
    --train.policy.re_init_std=True \
    --train.policy.init_noise_std=0.5 \
    --train.policy.disable_actor_terrain_input=True \
    --train.algorithm.learning_rate=2e-5 \
    --train.algorithm.schedule=fixed \
    --train.runner.use_wandb=True \
    --env.deepmimic.amass_terrain_difficulty=1 \
    --env.deepmimic.upsample_data=True \
    --env.deepmimic.use_human_videos=True \
    --env.deepmimic.human_video_oversample_factor="${oversample_factor}" \
    --env.deepmimic.link_pos_error_threshold=0.5 \
    --train.runner.save_interval="${SAVE_INTERVAL}" \
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

  echo "[Sequence] Finished ${run_name}"
}

run_clip \
  "0413_bt_ft_holosoma_stairs" \
  "videomimic_stage2_single_clip_blindterrain_0413_holosoma_stairs" \
  "resources/data_config/holosoma_stairs_motion.yaml" \
  "1" \
  "1"

run_clip \
  "0413_bt_ft_7276seg2" \
  "videomimic_stage2_single_clip_blindterrain_0413_7276seg2" \
  "resources/data_config/human_motion_single_0331_7276seg2.yaml" \
  "8" \
  "8"

run_clip \
  "0413_bt_ft_5568" \
  "videomimic_stage2_single_clip_blindterrain_0413_5568" \
  "resources/data_config/human_motion_single_0331_5568.yaml" \
  "8" \
  "8"

run_clip \
  "0413_bt_ft_5585" \
  "videomimic_stage2_single_clip_blindterrain_0413_5585" \
  "resources/data_config/human_motion_single_0331_5585.yaml" \
  "8" \
  "8"

echo "[Sequence] All blind-terrain single-clip runs completed."
