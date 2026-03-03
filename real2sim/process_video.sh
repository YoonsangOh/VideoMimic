#!/bin/bash

# Check if required arguments are provided
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <video_name> <start_frame> <end_frame> <subsample_factor> <robot_name> <height> [depth_model] [output_tag]"
    echo "Example: $0 my_video 0 100 2 g1 1.8 moge exp1"
    exit 1
fi

# Parse arguments
VIDEO_NAME=$1
START_FRAME=$2
END_FRAME=$3
SUBSAMPLE_FACTOR=$4
ROBOT_NAME=$5
HEIGHT=$6
DEPTH_MODEL=${7:-moge}
OUTPUT_TAG=${8:-}

echo "Processing video: $VIDEO_NAME"
echo "Frames: $START_FRAME to $END_FRAME"
echo "Subsample factor: $SUBSAMPLE_FACTOR"
echo "Robot name: $ROBOT_NAME"
echo "Height: $HEIGHT"
echo "Depth model: $DEPTH_MODEL"
echo "Output tag: ${OUTPUT_TAG:-<none>}"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vm1rs


# Step 0: Run preprocessing
echo "Step 0: Running preprocessing..."
bash preprocess_human.sh "$VIDEO_NAME" 0 # 0 or 1 for vis


conda deactivate
conda activate vm1recon

# Step 1: Run Megasam
echo "Step 1: Running Megasam..."
STAGE1_CMD=(
    python stage1_reconstruction/megasam_reconstruction.py
    --video-dir "./demo_data/input_images/${VIDEO_NAME}/cam01"
    --out-dir "./demo_data/input_megasam"
    --start-frame "$START_FRAME"
    --end-frame "$END_FRAME"
    --stride "$SUBSAMPLE_FACTOR"
    --depth-model "$DEPTH_MODEL"
    --gsam2
)
if [ -n "$OUTPUT_TAG" ]; then
    STAGE1_CMD+=(--output-tag "$OUTPUT_TAG")
fi
printf '%q ' "${STAGE1_CMD[@]}"; echo
"${STAGE1_CMD[@]}"

STAGE1_VIDEO_TOKEN="$VIDEO_NAME"
if [ -n "$OUTPUT_TAG" ]; then
    STAGE1_VIDEO_TOKEN="${VIDEO_NAME}_${OUTPUT_TAG}"
fi
MEGASAM_OUTPUT=$(ls -t ./demo_data/input_megasam/megasam_reconstruction_results_${STAGE1_VIDEO_TOKEN}_cam01_frame_${START_FRAME}_${END_FRAME}_subsample_${SUBSAMPLE_FACTOR}.h5 2>/dev/null | head -n 1)


if [ -z "$MEGASAM_OUTPUT" ]; then
    echo "Error: Could not find Megasam output file"
    echo "Megasam output file: $MEGASAM_OUTPUT"
    exit 1
fi

conda deactivate
conda activate vm1rs


# Add --use-g1-shape flag for g1 robot if no height specified
if [ -n "$HEIGHT" ] && [ "$HEIGHT" == "0" ] && [ "$ROBOT_NAME" = "g1" ]; then
    MEGAHUNTER_FLAGS="--use-g1-shape"
    echo "Using g1 shape"
else
    MEGAHUNTER_FLAGS=""
fi

if [ -n "$HEIGHT" ] && [ "$HEIGHT" != "0" ] && [ "$HEIGHT" != "-1" ]; then
    echo "Using height: $HEIGHT"
    python stage2_optimization/optimize_smpl_shape_for_height.py --height "$HEIGHT" --output-dir "./demo_data/input_3d_meshes/${VIDEO_NAME}/cam01"
fi

# Step 2: Run MegaHunter (SMPL optimization)
echo "Step 2: Running MegaHunter..."
if [ "$MEGAHUNTER_FLAGS" == "--use-g1-shape" ]; then
    echo "python stage2_optimization/megahunter_optimization.py \
        --world-env-path \"$MEGASAM_OUTPUT\" \
        --bbox-dir \"./demo_data/input_masks/${VIDEO_NAME}/cam01/json_data\" \
        --pose2d-dir \"./demo_data/input_2d_poses/${VIDEO_NAME}/cam01\" \
        --smpl-dir \"./demo_data/input_3d_meshes/${VIDEO_NAME}/cam01\" \
        --out-dir \"./demo_data/output_smpl_and_points\" \
        $MEGAHUNTER_FLAGS"

    python stage2_optimization/megahunter_optimization.py \
        --world-env-path "$MEGASAM_OUTPUT" \
        --bbox-dir "./demo_data/input_masks/${VIDEO_NAME}/cam01/json_data" \
        --pose2d-dir "./demo_data/input_2d_poses/${VIDEO_NAME}/cam01" \
        --smpl-dir "./demo_data/input_3d_meshes/${VIDEO_NAME}/cam01" \
        --out-dir "./demo_data/output_smpl_and_points" \
        $MEGAHUNTER_FLAGS
else
    echo "python stage2_optimization/megahunter_optimization.py \
        --world-env-path \"$MEGASAM_OUTPUT\" \
        --bbox-dir \"./demo_data/input_masks/${VIDEO_NAME}/cam01/json_data\" \
        --pose2d-dir \"./demo_data/input_2d_poses/${VIDEO_NAME}/cam01\" \
        --smpl-dir \"./demo_data/input_3d_meshes/${VIDEO_NAME}/cam01\" \
        --out-dir \"./demo_data/output_smpl_and_points\""

    python stage2_optimization/megahunter_optimization.py \
        --world-env-path "$MEGASAM_OUTPUT" \
        --bbox-dir "./demo_data/input_masks/${VIDEO_NAME}/cam01/json_data" \
        --pose2d-dir "./demo_data/input_2d_poses/${VIDEO_NAME}/cam01" \
        --smpl-dir "./demo_data/input_3d_meshes/${VIDEO_NAME}/cam01" \
        --out-dir "./demo_data/output_smpl_and_points"
fi

# Get the MegaHunter output file
MEGASAM_BASENAME=$(basename "$MEGASAM_OUTPUT" .h5)
MEGAHUNTER_OUTPUT="./demo_data/output_smpl_and_points/megahunter_${MEGASAM_BASENAME}.h5"

if [ ! -f "$MEGAHUNTER_OUTPUT" ]; then
    echo "Error: Could not find MegaHunter output file"
    exit 1
fi

MEGAHUNTER_BASENAME=$(basename "$MEGAHUNTER_OUTPUT" .h5)
POSTPROCESS_OUT_DIR="./demo_data/output_calib_mesh/${MEGAHUNTER_BASENAME}"

# conda deactivate
eval "$(conda shell.bash hook)"
conda activate vm1recon

# Step 3.1: Run postprocessing; GeoCalib + NDC 
echo "Step 3.1: Running postprocessing..."
if [ "$MEGAHUNTER_FLAGS" == "--use-g1-shape" ]; then
    echo "Using male shape"
    python stage3_postprocessing/postprocessing_pipeline.py \
        --megahunter-path "$MEGAHUNTER_OUTPUT" \
        --out-dir "$POSTPROCESS_OUT_DIR" \
        --gender "male" \
        --is-megasam
else
    # echo "Using neutral shape"
    echo "Using male shape"
    python stage3_postprocessing/postprocessing_pipeline.py \
        --megahunter-path "$MEGAHUNTER_OUTPUT" \
        --out-dir "$POSTPROCESS_OUT_DIR" \
        --gender "male" \
        --is-megasam
fi

conda deactivate
conda activate vm1rs

# Step 4: Run retargeting
echo "Step 4: Running retargeting..."
python stage4_retargeting/robot_motion_retargeting.py \
    --src-dir "$POSTPROCESS_OUT_DIR" \
    --contact-dir "./demo_data/input_contacts/${VIDEO_NAME}/cam01"

echo "Processing complete!"
