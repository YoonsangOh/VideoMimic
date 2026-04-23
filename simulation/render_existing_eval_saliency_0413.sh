#!/usr/bin/env bash

set -u -o pipefail

REPO_ROOT="/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic"
CONDA_SH="/home/nas4_user/kyungminlee/anaconda3/etc/profile.d/conda.sh"
GPU_ID="${CUDA_VISIBLE_DEVICES:-5}"
SAL_MODES="${SAL_MODES:-global_mean local_mean gradient}"

source "${CONDA_SH}"
conda activate OpenHL

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

validate_mp4() {
    local path="$1"
    python - <<'PY' "${path}"
import cv2
import sys

path = sys.argv[1]
cap = cv2.VideoCapture(path)
ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 0
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 0
fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
cap.release()
if not ok:
    raise SystemExit(1)
print(f"ok frames={frames} fps={fps:.2f} size={width}x{height}")
PY
}

run_render_job() {
    local script="$1"
    local task="$2"
    local load_run="$3"
    local checkpoint="$4"
    local eval_root="$5"
    local saliency_mode="$6"

    local out_dir="${eval_root}/mp4_follow_future_ref_saliency_${saliency_mode}"
    mkdir -p "${out_dir}"

    for rollout_dir in "${eval_root}"/rollouts/eval_*; do
        local eval_name
        local out_mp4
        local status

        eval_name="$(basename "${rollout_dir}")"
        out_mp4="${out_dir}/${eval_name}.mp4"
        rm -f "${out_mp4}"

        echo "[Render][START] ${rollout_dir} -> ${out_mp4}"
        python "${script}" \
            --mode render \
            --task "${task}" \
            --load-run "${load_run}" \
            --checkpoint "${checkpoint}" \
            --rollout-dir "${rollout_dir}" \
            --mp4-path "${out_mp4}" \
            --camera-mode follow \
            --show-terrain-saliency \
            --terrain-saliency-mode "${saliency_mode}"
        status=$?

        if validate_mp4 "${out_mp4}"; then
            if [[ ${status} -ne 0 ]]; then
                echo "[Render][WARN] non-zero exit (${status}) but readable mp4 was produced: ${out_mp4}"
            else
                echo "[Render][OK] ${out_mp4}"
            fi
        else
            echo "[Render][ERROR] invalid mp4: ${out_mp4} (exit=${status})" >&2
            return 1
        fi
    done
}

for saliency_mode in ${SAL_MODES}; do
    run_render_job \
        "simulation/eval_single_clip_checkpoint_to_mp4.py" \
        "g1_deepmimic_proj_heightfield" \
        "20260331_040113_0331_ft_5568" \
        "330000" \
        "${REPO_ROOT}/simulation/data/single_clip_policy_eval/20260331_040113_0331_ft_5568_ckpt330000" \
        "${saliency_mode}" || exit 1

    run_render_job \
        "simulation/eval_single_clip_checkpoint_to_mp4.py" \
        "g1_deepmimic_proj_heightfield" \
        "20260331_040113_0331_ft_5585" \
        "330000" \
        "${REPO_ROOT}/simulation/data/single_clip_policy_eval/20260331_040113_0331_ft_5585_ckpt330000" \
        "${saliency_mode}" || exit 1

    run_render_job \
        "simulation/eval_single_clip_checkpoint_to_mp4.py" \
        "g1_deepmimic_proj_heightfield" \
        "20260331_040157_0331_ft_7276seg2" \
        "330000" \
        "${REPO_ROOT}/simulation/data/single_clip_policy_eval/20260331_040157_0331_ft_7276seg2_ckpt330000" \
        "${saliency_mode}" || exit 1

    run_render_job \
        "simulation/eval_holosoma_stairs_checkpoint_to_mp4.py" \
        "g1_deepmimic_proj_heightfield" \
        "20260316_034146_holosoma_stairs_heightfield_single_gpu_30k_20260316" \
        "330000" \
        "${REPO_ROOT}/simulation/data/holosoma_stairs_policy_eval/20260316_034146_holosoma_stairs_heightfield_single_gpu_30k_20260316_ckpt330000" \
        "${saliency_mode}" || exit 1
done

echo "[Render][DONE] all saliency renders completed on GPU ${GPU_ID} for modes: ${SAL_MODES}"
