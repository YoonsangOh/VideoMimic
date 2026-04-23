#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CONDA_ROOT="/home/nas4_user/kyungminlee/anaconda3"
CONDA_ENV="${CONDA_ENV:-OpenHL}"

GPU_INDEX="${GPU_INDEX:?Set GPU_INDEX}"
LOAD_RUN="${LOAD_RUN:?Set LOAD_RUN}"
MOTION_SOURCE="${MOTION_SOURCE:?Set MOTION_SOURCE}"
CHECKPOINT="${CHECKPOINT:-330000}"
NUM_EVALS="${NUM_EVALS:-5}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
CAMERA_MODE="${CAMERA_MODE:-follow}"
FUTURE_OFFSETS="${FUTURE_OFFSETS:-10,20,30}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/data/single_clip_policy_eval}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONPATH="${ROOT_DIR}/videomimic_gym:${PYTHONPATH:-}"

RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${LOAD_RUN}_ckpt${CHECKPOINT}"

python - <<PY
import shutil
from pathlib import Path
path = Path(r"${RUN_OUTPUT_DIR}")
if path.exists():
    shutil.rmtree(path)
PY

cd "${ROOT_DIR}/.."

python "${ROOT_DIR}/eval_single_clip_checkpoint_to_mp4.py" \
  --mode eval \
  --load-run "${LOAD_RUN}" \
  --checkpoint "${CHECKPOINT}" \
  --motion-source "${MOTION_SOURCE}" \
  --num-evals "${NUM_EVALS}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --camera-mode "${CAMERA_MODE}" \
  --future-offsets "${FUTURE_OFFSETS}" \
  --output-root "${OUTPUT_ROOT}"

for rollout_dir in "${RUN_OUTPUT_DIR}"/rollouts/eval_*; do
  eval_name="$(basename "${rollout_dir}")"
  mp4_path="${RUN_OUTPUT_DIR}/mp4_${CAMERA_MODE}_future_ref/${eval_name}.mp4"

  python "${ROOT_DIR}/eval_single_clip_checkpoint_to_mp4.py" \
    --mode render \
    --rollout-dir "${rollout_dir}" \
    --mp4-path "${mp4_path}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --camera-mode "${CAMERA_MODE}" \
    --future-offsets "${FUTURE_OFFSETS}" || true

  python - <<PY
from pathlib import Path
import cv2
mp4 = Path(r"${mp4_path}")
if not mp4.exists():
    raise SystemExit(f"missing mp4: {mp4}")
cap = cv2.VideoCapture(str(mp4))
ok = cap.isOpened()
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
if (not ok) or frames <= 0:
    raise SystemExit(f"invalid mp4: {mp4}")
print(f"validated {mp4} frames={frames}")
PY
done
