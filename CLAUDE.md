# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**VideoMimic** (CoRL 2025 Best Student Paper) converts single-camera RGB videos of humans into 3D humanoid robot motion data for imitation learning. The pipeline: monocular video → 3D human pose reconstruction → human-environment alignment → robot motion retargeting → RL training → physical robot deployment.

Three major sub-pipelines:
- **`real2sim/`** — Vision pipeline: video frames → simulation-ready motion + environment mesh
- **`simulation/`** — RL training pipeline using Isaac Gym
- **`sim2real/`** — C++ deployment code for physical Unitree G1 robot

## Conda Environments

Two environments are required for `real2sim` due to CUDA compatibility constraints:

| Env | Python | CUDA | Purpose |
|-----|--------|------|---------|
| `vm1rs` | 3.12 | 12.4+ | All stages except reconstruction/meshification |
| `vm1recon` | 3.10 | 11.8 | MegaSam, NKSR, GeoCalib (requires xformers ≤0.0.27) |
| `rlgpu` | 3.8 | 12.1 | Simulation training with Isaac Gym |

## real2sim Pipeline

All commands run from `real2sim/` directory. Demo data is under `real2sim/demo_data/`.

### Full pipeline (single command)
```bash
# Extract frames first
python utilities/extract_frames_from_video.py --video-path {video}.{ext} \
    --output-dir ./demo_data/input_images/{video}/cam01 --start-frame 0 --end-frame 300

# Run complete pipeline
conda activate vm1rs
./process_video.sh <video_name> <start_frame> <end_frame> <subsample_factor> g1 <height>
# Example: ./process_video.sh my_video 0 100 2 g1 1.8
# height=0: use robot-fitted SMPL shape; height=-1: use estimated SMPL shape
```

### Stage 0: Human Preprocessing (vm1rs)
```bash
bash preprocess_human.sh <video_name> 1   # with visualization
bash preprocess_human.sh <video_name> 0   # without visualization
```

Individual steps: `sam2_segmentation.py` → `vitpose_2d_poses.py` → `vimo_3d_mesh.py` → `bstro_contact_detection.py`

### Stage 1: Environment Reconstruction
- **MegaSam** (recommended, ~24GB VRAM for 300 frames): requires `vm1recon`
- **Align3r/Monst3r** (textureless scenes, ~80GB VRAM for 150 frames): use `vm1rs`

```bash
conda activate vm1recon
python stage1_reconstruction/megasam_reconstruction.py \
    --out-dir ./demo_data/input_megasam \
    --video-dir ./demo_data/input_images/my_video/cam01 \
    --start-frame 0 --end-frame 100 --stride 1 --gsam2
```

### Stage 2: MegaHunter Optimization (vm1rs)
Aligns 3D human pose with the reconstructed environment. Uses JAX JIT; pads frames to next multiple of 100. Recommend <300 frames (A100 80GB limit).
```bash
conda activate vm1rs
python stage2_optimization/megahunter_optimization.py \
    --world-env-path ./demo_data/input_megasam/megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1.h5 \
    --bbox-dir ./demo_data/input_masks/my_video/cam01/json_data \
    --pose2d-dir ./demo_data/input_2d_poses/my_video/cam01 \
    --smpl-dir ./demo_data/input_3d_meshes/my_video/cam01 \
    --out-dir ./demo_data/output_smpl_and_points
    # Add --use-g1-shape for G1 robot proportions
```

### Stage 3: Postprocessing — Gravity Calibration & Mesh (vm1recon)
```bash
conda activate vm1recon
python stage3_postprocessing/postprocessing_pipeline.py \
    --megahunter-path ./demo_data/output_smpl_and_points/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1.h5 \
    --out-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1 \
    --conf-thr 0.0 --is-megasam --scale-bbox3d 1.5 --vis
```

### Stage 4: Robot Motion Retargeting (vm1rs)
Retargeting is sensitive to cost weights. If results look wrong, set `--foot-skating-cost-weight`, `--ground-contact-cost-weight`, and `--world-coll-factor-weight` to 0.0 first.
```bash
conda activate vm1rs
python stage4_retargeting/robot_motion_retargeting.py \
    --src-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_my_video_cam01_frame_0_100_subsample_1/ \
    --contact-dir ./demo_data/input_contacts/my_video/cam01 --vis
```

### Output Files (simulation-ready)
```
output_calib_mesh/<result_dir>/
├── gravity_calibrated_keypoints.h5          # Human keypoints + world rotation
├── gravity_calibrated_megahunter.h5         # Full optimized data with gravity calibration
├── background_mesh.obj                       # Environment mesh (for RL terrain)
├── background_{less,more}_filtered_colored_pointcloud.ply
└── retarget_poses_g1.h5                     # Robot reference motion (for RL)
```

### Visualization (all use Viser at localhost:8080)
```bash
# Complete results with ego-view
python visualization/complete_results_egoview_visualization.py \
    --postprocessed-dir ./demo_data/output_calib_mesh/<result_dir> \
    --robot-name g1 --bg-pc-downsample-factor 4 --is-megasam --save-ego-view

# Per-stage visualizations
python visualization/optimization_results_visualization.py --world-env-path <path.h5> --bg-pc-downsample-factor 4
python visualization/retargeting_visualization.py --postprocessed-dir <dir> --robot-name g1 --bg-pc-downsample-factor 4
python visualization/mesh_generation_visualization.py --mesh-path <.obj> --points-path <.ply>
python visualization/gravity_calibration_visualization.py --calib-out-dir <dir>
python visualization/environment_only_visualization.py --world-env-path <path.h5> --world-scale-factor 1.5 --conf-thr 0.0
```

### Batch/Sequential Processing
For multiple videos, use `sequential_processing/` scripts to amortize JAX compilation:
```bash
python sequential_processing/stage2_sequential_megahunter_optimization.py --pattern {common_substring} --use-g1-shape
```

## Simulation Training

```bash
conda activate rlgpu
cd simulation

# Install (one-time)
cd videomimic_rl && pip install -e . && cd ..
cd videomimic_gym && pip install -e . && cd ..

# Download data
cd data && bash download_videomimic_data.sh && cd ..

# 4-stage training
bash videomimic_gym/legged_gym/scripts/train_stage_1_mcpt.sh
bash videomimic_gym/legged_gym/scripts/train_stage_2_terrain_rl.sh
LOAD_RUN=<stage2_run_name>
bash videomimic_gym/legged_gym/scripts/train_stage_3_distillation.sh ${LOAD_RUN}
LOAD_RUN=<stage3_run_name>
bash videomimic_gym/legged_gym/scripts/train_stage_4_rl_finetune.sh ${LOAD_RUN}

# Inference/Play (Viser UI at localhost:8080)
bash videomimic_gym/legged_gym/scripts/play_terrain_policy.sh
bash videomimic_gym/legged_gym/scripts/play_flat_policy.sh
bash videomimic_gym/legged_gym/scripts/play_mcpt_policy.sh
```

Direct training command:
```bash
# Single GPU
python legged_gym/scripts/train.py --task=g1_deepmimic

# Multi-GPU
torchrun --nproc-per-node <num_gpus> legged_gym/scripts/train.py --multi_gpu --task=g1_deepmimic

# Key task options: g1_deepmimic, g1_deepmimic_dagger (distillation)
# Override config: --env.deepmimic.use_human_videos=True --train.x=y
```

## Sim-to-Real Deployment

```bash
cd sim2real
mkdir build && cd build
export CMAKE_PREFIX_PATH=/home/unitree/.local/lib/python3.8/site-packages/torch:/home/unitree/noetic_ws/devel:/opt/ros/noetic
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CUDA_ARCHITECTURES="87" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
make
# Binary: build/bin/videomimic_inference_real
# Run: ./bin/videomimic_inference_real eth0
```
Requires ROS 1 Noetic and elevation mapping on Jetson/Unitree G1. Update checkpoint paths in `videomimic_real/videomimic_inference_real.cpp`.

## Architecture Notes

- **Data format**: HDF5 (`.h5`) for intermediate data between stages; `.obj`/`.ply` for meshes
- **Body model**: SMPL (human) → SMPL-X (with hands, deprecated) → robot joints (G1)
- **Optimization**: JAX + jaxls for differentiable optimization in MegaHunter; PyRoki for kinematics
- **Robot**: Unitree G1 humanoid (configs in `simulation/videomimic_gym/legged_gym/envs/g1/`)
- **Multi-human**: Add `--multihuman --top-k N` flags consistently across stages 0, 2, 3, 4
- **PyTorch version**: Use 2.5.1 for `vm1rs` (avoid 2.6 — unstable); 2.3.1 for `rlgpu`
- **Third-party code**: Installed into `real2sim/third_party/` (not committed to repo)

## Key Docs
- `real2sim/docs/setup.md` — Detailed installation with troubleshooting
- `real2sim/docs/commands.md` — Full command reference with all options
- `real2sim/docs/multihuman.md` — Multi-person processing
- `real2sim/docs/sequential_processing.md` — Batch processing guide
- `simulation/videomimic_gym/README.md` — Training parameter reference
- `simulation/docs/SIMULATION_GUIDE.md` — Full training workflow
