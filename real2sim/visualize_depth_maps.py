#!/usr/bin/env python3
"""
Visualize depth maps from reconstruction results.
This script extracts and visualizes depth maps from each frame's camera view.
"""

import h5py
import numpy as np
import cv2
import os
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_preview_compat import CompatibleVideoWriter

def load_dict_from_hdf5(h5file, path="/"):
    """Recursively load a nested dictionary from an HDF5 file."""
    result = {}
    for key in h5file[path].keys():
        key_path = f"{path}{key}"
        if isinstance(h5file[key_path], h5py.Group):
            result[key] = load_dict_from_hdf5(h5file, key_path + "/")
        else:
            result[key] = h5file[key_path][:]
    return result

def visualize_depth_maps(h5_path, output_dir, max_frames=None, save_rgb=False, create_video=False, fps=30):
    """
    Extract and visualize depth maps from reconstruction H5 file.

    Args:
        h5_path: Path to H5 file (Stage 1 or Stage 2 output)
        output_dir: Directory to save depth map visualizations
        max_frames: Maximum number of frames to process (None for all)
        save_rgb: Whether to also save RGB images for comparison
        create_video: Whether to create video from depth maps
        fps: Frames per second for video
    """
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        data = load_dict_from_hdf5(f)

    # Determine data structure
    if 'monst3r_ga_output' in data:
        # Stage 1 file
        world_env = data['monst3r_ga_output']
        print("Detected Stage 1 file (MegaSam reconstruction)")
    elif 'our_pred_world_cameras_and_structure' in data:
        # Stage 2 file
        world_env = data['our_pred_world_cameras_and_structure']
        print("Detected Stage 2 file (MegaHunter optimization)")
    else:
        raise ValueError("Unknown file format. Expected 'monst3r_ga_output' or 'our_pred_world_cameras_and_structure'")

    frame_names = sorted([k for k in world_env.keys()])
    if max_frames:
        frame_names = frame_names[:max_frames]

    print(f"Processing {len(frame_names)} frames...")

    # First pass: compute global depth range for consistent coloring
    print("Computing global depth range...")
    global_depth_min = float('inf')
    global_depth_max = float('-inf')
    for frame_name in tqdm(frame_names, desc="Scanning depths"):
        try:
            if 'depths' in world_env[frame_name]:
                depths = world_env[frame_name]['depths']
                global_depth_min = min(global_depth_min, depths.min())
                global_depth_max = max(global_depth_max, depths.max())
        except:
            continue

    print(f"Global depth range: {global_depth_min:.3f}m - {global_depth_max:.3f}m")

    # Get image dimensions from first frame
    first_frame_name = frame_names[0]
    depths_shape = world_env[first_frame_name]['depths'].shape
    rgb_shape = world_env[first_frame_name]['rgbimg'].shape[:2]

    # Initialize video writers if needed
    depth_video_writer = None
    comparison_video_writer = None
    if create_video:
        h, w = depths_shape[:2]
        depth_video_path = os.path.join(output_dir, "depth_map_video.mp4")
        depth_video_writer = CompatibleVideoWriter(depth_video_path, fps, (w, h), input_color="bgr")

        comparison_video_path = os.path.join(output_dir, "rgb_depth_comparison_video.mp4")
        comparison_video_writer = CompatibleVideoWriter(comparison_video_path, fps, (w * 2, h), input_color="bgr")
        print(f"Creating videos: {depth_video_path} and {comparison_video_path}")

    all_depths = []
    all_rgb = []

    # Second pass: process frames and create visualizations
    for i, frame_name in enumerate(tqdm(frame_names, desc="Processing frames")):
        try:
            # Get depth map
            if 'depths' in world_env[frame_name]:
                depths = world_env[frame_name]['depths']
            else:
                print(f"Warning: No depth map found for {frame_name}")
                continue

            # Get RGB image for comparison
            rgb_img = world_env[frame_name]['rgbimg']
            if rgb_img.max() <= 1.0:
                rgb_img = (rgb_img * 255).astype(np.uint8)
            else:
                rgb_img = rgb_img.astype(np.uint8)

            # Normalize using global range for consistent coloring across frames
            depth_norm = (depths - global_depth_min) / (global_depth_max - global_depth_min + 1e-8)
            depth_norm = np.clip(depth_norm, 0, 1)

            # Apply color map
            depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

            # Save depth map image
            depth_path = os.path.join(output_dir, f"{frame_name}_depth.png")
            cv2.imwrite(depth_path, depth_colored)

            # Save RGB if requested
            if save_rgb:
                rgb_path = os.path.join(output_dir, f"{frame_name}_rgb.png")
                cv2.imwrite(rgb_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

            # Create side-by-side comparison
            if rgb_img.shape[:2] == depths.shape[:2]:
                h, w = depths.shape[:2]
                comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
                comparison[:, :w] = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                comparison[:, w:] = depth_colored
                comparison_path = os.path.join(output_dir, f"{frame_name}_comparison.png")
                cv2.imwrite(comparison_path, comparison)

                # Add to video
                if create_video:
                    depth_video_writer.write(depth_colored)
                    comparison_video_writer.write(comparison)

            all_depths.append(depths)
            all_rgb.append(rgb_img)

        except Exception as e:
            print(f"Error processing {frame_name}: {e}")
            continue

    # Release video writers
    if create_video:
        if depth_video_writer:
            depth_video_writer.release()
            print(f"Depth video saved: {os.path.join(output_dir, 'depth_map_video.mp4')}")
        if comparison_video_writer:
            comparison_video_writer.release()
            print(f"Comparison video saved: {os.path.join(output_dir, 'rgb_depth_comparison_video.mp4')}")

    # Print statistics
    if all_depths:
        all_depths_array = np.array(all_depths)
        print(f"\nDepth Map Statistics:")
        print(f"  Total frames: {len(all_depths)}")
        print(f"  Depth range: {all_depths_array.min():.3f}m - {all_depths_array.max():.3f}m")
        print(f"  Mean depth: {all_depths_array.mean():.3f}m")
        print(f"  Std depth: {all_depths_array.std():.3f}m")
        print(f"\nDepth maps saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Visualize depth maps from reconstruction results")
    parser.add_argument("--h5-path", type=str, required=True, help="Path to H5 file")
    parser.add_argument("--output-dir", type=str, default="./depth_visualizations", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--save-rgb", action="store_true", help="Also save RGB images")
    parser.add_argument("--create-video", action="store_true", help="Create video from depth maps")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for video")

    args = parser.parse_args()

    visualize_depth_maps(args.h5_path, args.output_dir, args.max_frames, args.save_rgb, args.create_video, args.fps)

if __name__ == "__main__":
    main()
