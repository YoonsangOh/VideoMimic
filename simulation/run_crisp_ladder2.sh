#!/bin/bash
# Run CRISP ladder2 simulation with VideoMimic
#
# Usage:
#   ./run_crisp_ladder2.sh retarget   # Run retargeting first
#   ./run_crisp_ladder2.sh play       # Play with trained policy
#   ./run_crisp_ladder2.sh train      # Train new policy

set -e

# Set LD_LIBRARY_PATH for Isaac Gym (needs libpython3.8.so.1.0)
export LD_LIBRARY_PATH="/home/nas5/kyungminlee/anaconda3/envs/videomimic/lib:${LD_LIBRARY_PATH}"

# Disable WandB to prevent logging to supervisor's account
export WANDB_MODE=disabled
unset WANDB_API_KEY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEOMIMIC_ROOT="/home/nas5/kyungminlee/VideoMimic"
REAL2SIM_ROOT="${VIDEOMIMIC_ROOT}/real2sim"
SIM_ROOT="${VIDEOMIMIC_ROOT}/simulation"

CRISP_DATA_DIR="${REAL2SIM_ROOT}/demo_data/output_calib_mesh/crisp_ladder2_cam01_frame_0_375_subsample_2"
MOTION_YAML="crisp_ladder2_motion.yaml"
TEACHER_CHECKPOINT="20250410_063030_g1_deepmimic"

cd "${SIM_ROOT}"

case "${1:-play}" in
    retarget)
        echo "=== Running Retargeting for CRISP ladder2 ==="
        echo "Note: Retargeting requires 'vm1rs' conda environment"
        echo "Please run: conda activate vm1rs"
        echo "Then run: cd ${REAL2SIM_ROOT} && python stage4_retargeting/robot_motion_retargeting.py --src_dir demo_data/output_calib_mesh/crisp_ladder2_cam01_frame_0_375_subsample_2 --contact_dir demo_data/input_contacts/ladder2/cam01"
        echo ""
        echo "Or run directly:"
        cd "${REAL2SIM_ROOT}"
        # Note: robot type is hardcoded in the script (g1_29dof_anneal_23dof_foot.urdf)
        # Reuse existing VideoMimic ladder2 contact data for foot skating prevention
        # Try to run, but warn if wrong environment
        if ! python -c "import jax" 2>/dev/null; then
            echo "ERROR: 'jax' module not found. Please activate 'vm1rs' conda environment first:"
            echo "  conda activate vm1rs"
            echo "  cd ${REAL2SIM_ROOT}"
            echo "  python stage4_retargeting/robot_motion_retargeting.py \\"
            echo "      --src_dir demo_data/output_calib_mesh/crisp_ladder2_cam01_frame_0_375_subsample_2 \\"
            echo "      --contact_dir demo_data/input_contacts/ladder2/cam01"
            exit 1
        fi
        python stage4_retargeting/robot_motion_retargeting.py \
            --src_dir demo_data/output_calib_mesh/crisp_ladder2_cam01_frame_0_375_subsample_2 \
            --contact_dir demo_data/input_contacts/ladder2/cam01
        echo "Retargeting complete! Now run: ./run_crisp_ladder2.sh play"
        ;;
    
    play)
        echo "=== Playing CRISP ladder2 with Isaac Gym viewer + Viser ==="
        if [ ! -f "${CRISP_DATA_DIR}/retarget_poses_g1.h5" ]; then
            echo "ERROR: retarget_poses_g1.h5 not found!"
            echo "Please run: ./run_crisp_ladder2.sh retarget"
            exit 1
        fi
        
        # Isaac Gym OpenGL viewer + Viser web UI (both enabled)
        python videomimic_gym/legged_gym/scripts/play.py \
            --task=g1_deepmimic \
            --num_envs=1 \
            --load_run=${TEACHER_CHECKPOINT} \
            --checkpoint=300000 \
            --env.viser.enable=True \
            --env.deepmimic.human_motion_source=resources/data_config/${MOTION_YAML} \
            --env.deepmimic.use_human_videos=True \
            --env.deepmimic.use_amass=False \
            --train.runner.load_model_strict=False \
            --no_use_wandb
        ;;
    
    play_viser)
        echo "=== Playing CRISP ladder2 with Viser web UI only (http://localhost:8080) ==="
        if [ ! -f "${CRISP_DATA_DIR}/retarget_poses_g1.h5" ]; then
            echo "ERROR: retarget_poses_g1.h5 not found!"
            echo "Please run: ./run_crisp_ladder2.sh retarget"
            exit 1
        fi
        
        # Viser web UI only (Isaac Gym viewer disabled via --headless)
        python videomimic_gym/legged_gym/scripts/play.py \
            --task=g1_deepmimic \
            --num_envs=1 \
            --headless \
            --load_run=${TEACHER_CHECKPOINT} \
            --checkpoint=300000 \
            --env.viser.enable=True \
            --env.deepmimic.human_motion_source=resources/data_config/${MOTION_YAML} \
            --env.deepmimic.use_human_videos=True \
            --env.deepmimic.use_amass=False \
            --train.runner.load_model_strict=False \
            --no_use_wandb
        ;;
    
    train)
        echo "=== Training on CRISP ladder2 ==="
        if [ ! -f "${CRISP_DATA_DIR}/retarget_poses_g1.h5" ]; then
            echo "ERROR: retarget_poses_g1.h5 not found!"
            echo "Please run: ./run_crisp_ladder2.sh retarget"
            exit 1
        fi
        
        NUM_GPUS="${2:-1}"
        echo "Training with ${NUM_GPUS} GPU(s)..."
        
        python videomimic_gym/legged_gym/scripts/train_stage_2_mcpt.py \
            --task g1_deepmimic \
            --motion_file ${MOTION_YAML} \
            --headless \
            --num_envs 4096 \
            --run_name "crisp_ladder2_training" \
            ${@:3}
        ;;
    
    check)
        echo "=== Checking CRISP ladder2 setup ==="
        echo ""
        echo "1. CRISP data directory:"
        ls -la "${CRISP_DATA_DIR}/" 2>/dev/null || echo "   NOT FOUND!"
        echo ""
        echo "2. Required files:"
        for f in "gravity_calibrated_megahunter.h5" "background_mesh.obj" "retarget_poses_g1.h5"; do
            if [ -f "${CRISP_DATA_DIR}/${f}" ]; then
                echo "   ✓ ${f}"
            else
                echo "   ✗ ${f} (missing)"
            fi
        done
        echo ""
        echo "3. YAML config:"
        if [ -f "${SIM_ROOT}/videomimic_gym/resources/data_config/${MOTION_YAML}" ]; then
            echo "   ✓ ${MOTION_YAML}"
            cat "${SIM_ROOT}/videomimic_gym/resources/data_config/${MOTION_YAML}"
        else
            echo "   ✗ ${MOTION_YAML} (missing)"
        fi
        echo ""
        echo "4. Symlink:"
        ls -la "${SIM_ROOT}/data/videomimic_captures/crisp_ladder2_cam01_frame_0_375_subsample_2" 2>/dev/null || echo "   NOT FOUND!"
        ;;
    
    *)
        echo "Usage: $0 {retarget|play|play_viser|train|check}"
        echo ""
        echo "Commands:"
        echo "  retarget   - Run robot motion retargeting (required first)"
        echo "  play       - Play with trained policy"
        echo "  play_viser - Play with Viser web visualization"
        echo "  train      - Train new policy"
        echo "  check      - Check setup status"
        exit 1
        ;;
esac

