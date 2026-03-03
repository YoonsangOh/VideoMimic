#!/usr/bin/env python3
"""
Visualize contact prediction results from BSTRO model - Version 2.
This version creates a mini-heatmap panel in the top-right corner and 
provides better debugging information.
"""

import pickle
import numpy as np
import cv2
import os
import argparse
import glob
from pathlib import Path
from tqdm import tqdm
import json
import h5py
import torch
import smplx


def load_smpl_vert_segmentation():
    """Load SMPL vertex segmentation for foot regions."""
    real2sim_root = Path(__file__).parent
    smpl_vert_seg_path = real2sim_root / "assets" / "body_models" / "smpl" / "smpl_vert_segmentation.json"
    
    if not smpl_vert_seg_path.exists():
        print(f"Warning: SMPL vertex segmentation not found at {smpl_vert_seg_path}")
        return None, None
    
    with open(smpl_vert_seg_path, 'r') as f:
        smpl_vert_seg = json.load(f)
    
    left_foot_vert_ids = np.array(smpl_vert_seg['leftFoot'], dtype=np.int32)
    right_foot_vert_ids = np.array(smpl_vert_seg['rightFoot'], dtype=np.int32)
    
    return left_foot_vert_ids, right_foot_vert_ids


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


def get_turbo_color(prob_norm):
    """Get turbo colormap color for probability (returns BGR)."""
    if prob_norm > 0.8:
        return (0, 0, 255)  # Red
    elif prob_norm > 0.6:
        t = (prob_norm - 0.6) / 0.2
        return (0, int(255 * (1 - t)), 255)  # Yellow to red
    elif prob_norm > 0.4:
        t = (prob_norm - 0.4) / 0.2
        return (0, 255, int(255 * t))  # Green to yellow
    elif prob_norm > 0.2:
        t = (prob_norm - 0.2) / 0.2
        return (int(255 * (1 - t)), 255, 0)  # Cyan to green
    else:
        t = prob_norm / 0.2
        return (255, int(255 * t), 0)  # Blue to cyan


def create_heatmap_panel(contact_probs, panel_size=200):
    """
    Create a mini heatmap panel showing all 6890 SMPL vertex contact probabilities.
    Arranges vertices in a grid layout.
    
    Args:
        contact_probs: (6890,) contact probabilities
        panel_size: Size of the square panel
        
    Returns:
        panel: (panel_size, panel_size, 3) BGR image
    """
    panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    
    # Arrange 6890 vertices in roughly 83x83 grid
    grid_size = int(np.ceil(np.sqrt(len(contact_probs))))  # ~83
    cell_size = panel_size / grid_size
    
    for i, prob in enumerate(contact_probs):
        row = i // grid_size
        col = i % grid_size
        
        x1 = int(col * cell_size)
        y1 = int(row * cell_size)
        x2 = int((col + 1) * cell_size)
        y2 = int((row + 1) * cell_size)
        
        # Get color based on probability
        prob_norm = float(np.clip(prob, 0, 1))
        color = get_turbo_color(prob_norm)
        
        # Fill cell
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, -1)
    
    return panel


def create_body_heatmap_panel(contact_probs, panel_width=150, panel_height=300):
    """
    Create a body-shaped heatmap panel showing contact probabilities on SMPL body parts.
    Uses a simplified body silhouette layout.
    
    Args:
        contact_probs: (6890,) contact probabilities
        panel_width: Width of the panel
        panel_height: Height of the panel
        
    Returns:
        panel: (panel_height, panel_width, 3) BGR image
    """
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    
    # SMPL vertex regions (approximate indices)
    body_regions = {
        'head': list(range(0, 400)),
        'torso': list(range(400, 1200)),
        'left_arm': list(range(1200, 1900)),
        'right_arm': list(range(1900, 2600)),
        'left_leg': list(range(2600, 4000)),
        'right_leg': list(range(4000, 5400)),
        'left_foot': list(range(5400, 6100)),
        'right_foot': list(range(6100, 6890)),
    }
    
    # Body part positions on panel (x_center, y_center, width, height)
    body_layout = {
        'head': (panel_width // 2, 25, 40, 50),
        'torso': (panel_width // 2, 90, 60, 80),
        'left_arm': (panel_width // 2 - 45, 90, 25, 80),
        'right_arm': (panel_width // 2 + 45, 90, 25, 80),
        'left_leg': (panel_width // 2 - 20, 200, 30, 100),
        'right_leg': (panel_width // 2 + 20, 200, 30, 100),
        'left_foot': (panel_width // 2 - 20, 275, 25, 25),
        'right_foot': (panel_width // 2 + 20, 275, 25, 25),
    }
    
    # Draw each body part
    for part_name, indices in body_regions.items():
        # Clamp indices to valid range
        valid_indices = [i for i in indices if i < len(contact_probs)]
        if not valid_indices:
            continue
            
        # Get average contact probability for this region
        part_probs = contact_probs[valid_indices]
        avg_prob = float(np.mean(part_probs))
        max_prob = float(np.max(part_probs))
        
        # Use max probability for color (more sensitive to contact)
        prob_norm = np.clip(max_prob, 0, 1)
        color = get_turbo_color(prob_norm)
        
        # Get layout
        x, y, w, h = body_layout[part_name]
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2
        
        # Draw rounded rectangle
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (255, 255, 255), 1)
    
    return panel


def create_foot_contact_panel(left_prob, right_prob, left_contact, right_contact, 
                               panel_width=200, panel_height=100):
    """
    Create a panel showing foot contact status.
    
    Args:
        left_prob: Left foot average probability
        right_prob: Right foot average probability
        left_contact: Boolean for left foot contact
        right_contact: Boolean for right foot contact
        panel_width: Width of the panel
        panel_height: Height of the panel
        
    Returns:
        panel: (panel_height, panel_width, 3) BGR image
    """
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    
    # Background
    cv2.rectangle(panel, (0, 0), (panel_width, panel_height), (40, 40, 40), -1)
    
    # Left foot
    left_color = (0, 255, 0) if left_contact else (100, 100, 100)
    cv2.circle(panel, (50, 50), 30, left_color, -1)
    cv2.circle(panel, (50, 50), 30, (255, 255, 255), 2)
    cv2.putText(panel, 'L', (40, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(panel, f'{left_prob:.2f}', (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Right foot
    right_color = (0, 255, 0) if right_contact else (100, 100, 100)
    cv2.circle(panel, (150, 50), 30, right_color, -1)
    cv2.circle(panel, (150, 50), 30, (255, 255, 255), 2)
    cv2.putText(panel, 'R', (140, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(panel, f'{right_prob:.2f}', (125, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return panel


def create_colorbar(width=20, height=200):
    """Create a vertical colorbar."""
    colorbar = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        prob = 1.0 - (i / height)  # Top is high, bottom is low
        color = get_turbo_color(prob)
        cv2.line(colorbar, (0, i), (width, i), color, 1)
    
    return colorbar


def visualize_frame(
    rgb_img: np.ndarray,
    contact_probs: np.ndarray,
    left_contact: bool,
    right_contact: bool,
    left_foot_prob: float,
    right_foot_prob: float,
    frame_idx: int = 0
) -> np.ndarray:
    """
    Create visualization with contact heatmap panels.
    
    Args:
        rgb_img: RGB image (H, W, 3)
        contact_probs: (6890,) contact probabilities for all SMPL vertices
        left_contact: Boolean for left foot contact
        right_contact: Boolean for right foot contact
        left_foot_prob: Average left foot probability
        right_foot_prob: Average right foot probability
        frame_idx: Frame index for display
        
    Returns:
        Visualization image (H, W, 3) BGR
    """
    h, w = rgb_img.shape[:2]
    vis_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    
    # Panel dimensions
    body_panel_w, body_panel_h = 150, 300
    foot_panel_w, foot_panel_h = 200, 100
    grid_panel_size = 150
    margin = 10
    
    # Create panels
    body_panel = create_body_heatmap_panel(contact_probs, body_panel_w, body_panel_h)
    foot_panel = create_foot_contact_panel(left_foot_prob, right_foot_prob, 
                                            left_contact, right_contact,
                                            foot_panel_w, foot_panel_h)
    grid_panel = create_heatmap_panel(contact_probs, grid_panel_size)
    colorbar = create_colorbar(20, body_panel_h)
    
    # Create info panel background
    info_panel_w = max(body_panel_w + 30, foot_panel_w, grid_panel_size) + margin * 2
    info_panel_h = body_panel_h + foot_panel_h + grid_panel_size + margin * 5 + 80
    
    # Position: top-right corner
    panel_x = w - info_panel_w - margin
    panel_y = margin
    
    # Draw semi-transparent background
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                  (panel_x + info_panel_w, panel_y + min(info_panel_h, h - margin * 2)), 
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, vis_img, 0.2, 0, vis_img)
    
    # Draw border
    cv2.rectangle(vis_img, (panel_x, panel_y), 
                  (panel_x + info_panel_w, panel_y + min(info_panel_h, h - margin * 2)), 
                  (100, 100, 100), 2)
    
    # Title
    cv2.putText(vis_img, "BSTRO Contact", (panel_x + 10, panel_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, f"Frame: {frame_idx}", (panel_x + 10, panel_y + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Place body heatmap
    body_y = panel_y + 60
    body_x = panel_x + margin
    if body_y + body_panel_h < h:
        vis_img[body_y:body_y + body_panel_h, body_x:body_x + body_panel_w] = body_panel
        # Colorbar next to body
        cb_x = body_x + body_panel_w + 5
        vis_img[body_y:body_y + body_panel_h, cb_x:cb_x + 20] = colorbar
        # Labels
        cv2.putText(vis_img, "High", (cb_x, body_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(vis_img, "Low", (cb_x, body_y + body_panel_h + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Place foot contact panel
    foot_y = body_y + body_panel_h + margin
    foot_x = panel_x + margin
    if foot_y + foot_panel_h < h:
        vis_img[foot_y:foot_y + foot_panel_h, foot_x:foot_x + foot_panel_w] = foot_panel
        cv2.putText(vis_img, "Foot Contact", (foot_x, foot_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Place grid heatmap (all vertices)
    grid_y = foot_y + foot_panel_h + margin + 20
    grid_x = panel_x + margin
    if grid_y + grid_panel_size < h:
        vis_img[grid_y:grid_y + grid_panel_size, grid_x:grid_x + grid_panel_size] = grid_panel
        cv2.putText(vis_img, "All Vertices", (grid_x, grid_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Stats text at bottom left
    stats_y = h - 60
    cv2.rectangle(vis_img, (10, stats_y - 5), (300, h - 10), (0, 0, 0), -1)
    cv2.putText(vis_img, f"Contact Probs - Max: {contact_probs.max():.3f}, Mean: {contact_probs.mean():.3f}", 
                (15, stats_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis_img, f"Vertices > 0.5: {(contact_probs > 0.5).sum()}, > 0.8: {(contact_probs > 0.8).sum()}", 
                (15, stats_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis_img


def main():
    parser = argparse.ArgumentParser(description="Visualize contact prediction results (V2)")
    parser.add_argument("--contact-dir", type=str, required=True, help="Directory containing contact .pkl files")
    parser.add_argument("--rgb-dir", type=str, required=True, help="Directory containing RGB images")
    parser.add_argument("--output-dir", type=str, default="./contact_visualizations_v2", help="Output directory")
    parser.add_argument("--person-id", type=int, default=1, help="Person ID to visualize")
    parser.add_argument("--create-video", action="store_true", help="Create video from visualizations")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for video")
    parser.add_argument("--save-images", action="store_true", help="Save individual frame images")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load SMPL vertex segmentation
    left_foot_vert_ids, right_foot_vert_ids = load_smpl_vert_segmentation()
    
    # Get all contact files
    contact_files = sorted(glob.glob(os.path.join(args.contact_dir, '*.pkl')))
    
    if len(contact_files) == 0:
        print(f"Error: No contact files found in {args.contact_dir}")
        return
    
    print(f"Found {len(contact_files)} contact files")
    
    # Get RGB image files
    rgb_files = sorted(glob.glob(os.path.join(args.rgb_dir, '*.jpg')))
    if len(rgb_files) == 0:
        rgb_files = sorted(glob.glob(os.path.join(args.rgb_dir, '*.png')))
    
    print(f"Found {len(rgb_files)} RGB images")
    
    # Create mapping from frame name to RGB file
    frame_to_rgb = {}
    for rgb_file in rgb_files:
        frame_name = Path(rgb_file).stem
        frame_to_rgb[frame_name] = rgb_file
    
    # Initialize video writer
    video_writer = None
    first_frame_processed = False
    
    # Statistics
    all_contact_probs = []
    left_contact_frames = 0
    right_contact_frames = 0
    total_frames = 0
    
    print("\n=== Processing Frames ===")
    
    for idx, contact_file in enumerate(tqdm(contact_files, desc="Processing")):
        frame_name = Path(contact_file).stem
        
        # Load contact data
        with open(contact_file, 'rb') as f:
            contact_data = pickle.load(f)
        
        if args.person_id not in contact_data:
            print(f"Warning: Person ID {args.person_id} not found in {contact_file}")
            continue
        
        person_data = contact_data[args.person_id]
        
        # Get contact probabilities
        if 'frame_contact_vertices' not in person_data:
            print(f"Warning: No contact vertices in {contact_file}")
            continue
        
        contact_probs = person_data['frame_contact_vertices'].flatten()
        all_contact_probs.append(contact_probs)
        
        # Get foot contact info
        left_contact = person_data.get('left_foot_contact', False)
        right_contact = person_data.get('right_foot_contact', False)
        left_foot_ratio = person_data.get('left_foot_contact_ratio', 0.0)
        right_foot_ratio = person_data.get('right_foot_contact_ratio', 0.0)
        
        # Calculate foot probabilities
        if left_foot_vert_ids is not None:
            left_foot_prob = float(contact_probs[left_foot_vert_ids].mean())
        else:
            left_foot_prob = float(left_foot_ratio)
            
        if right_foot_vert_ids is not None:
            right_foot_prob = float(contact_probs[right_foot_vert_ids].mean())
        else:
            right_foot_prob = float(right_foot_ratio)
        
        total_frames += 1
        if left_contact:
            left_contact_frames += 1
        if right_contact:
            right_contact_frames += 1
        
        # Load RGB image
        if frame_name in frame_to_rgb:
            rgb_img = cv2.imread(frame_to_rgb[frame_name])
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        else:
            # Create placeholder
            rgb_img = np.zeros((720, 1280, 3), dtype=np.uint8)
            print(f"Warning: RGB image not found for {frame_name}")
        
        # Create visualization
        vis_img = visualize_frame(
            rgb_img,
            contact_probs,
            left_contact,
            right_contact,
            left_foot_prob,
            right_foot_prob,
            idx
        )
        
        h, w = vis_img.shape[:2]
        
        # Initialize video writer on first frame
        if args.create_video and not first_frame_processed:
            video_path = os.path.join(args.output_dir, "contact_visualization_v2.mp4")
            video_writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                args.fps,
                (w, h)
            )
            print(f"\nCreating video: {video_path} ({w}x{h})")
            first_frame_processed = True
        
        # Save image
        if args.save_images:
            output_path = os.path.join(args.output_dir, f"{frame_name}_contact_v2.png")
            cv2.imwrite(output_path, vis_img)
        
        # Add to video
        if video_writer is not None:
            video_writer.write(vis_img)
        
        # Debug output for first few frames
        if idx < 3:
            print(f"\n  Frame {frame_name}:")
            print(f"    Contact probs shape: {contact_probs.shape}")
            print(f"    Contact probs range: [{contact_probs.min():.4f}, {contact_probs.max():.4f}]")
            print(f"    Vertices > 0.5: {(contact_probs > 0.5).sum()}")
            print(f"    Left foot: contact={left_contact}, prob={left_foot_prob:.3f}")
            print(f"    Right foot: contact={right_contact}, prob={right_foot_prob:.3f}")
    
    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved: {os.path.join(args.output_dir, 'contact_visualization_v2.mp4')}")
    
    # Print overall statistics
    if all_contact_probs:
        all_probs = np.concatenate(all_contact_probs)
        print(f"\n=== Overall Statistics ===")
        print(f"  Total frames processed: {total_frames}")
        print(f"  Left foot contact frames: {left_contact_frames}/{total_frames} ({100*left_contact_frames/total_frames:.1f}%)")
        print(f"  Right foot contact frames: {right_contact_frames}/{total_frames} ({100*right_contact_frames/total_frames:.1f}%)")
        print(f"  Contact probability stats:")
        print(f"    Min: {all_probs.min():.4f}")
        print(f"    Max: {all_probs.max():.4f}")
        print(f"    Mean: {all_probs.mean():.4f}")
        print(f"    Std: {all_probs.std():.4f}")
        print(f"    Vertices > 0.5 (total): {(all_probs > 0.5).sum()}")
        print(f"    Vertices > 0.8 (total): {(all_probs > 0.8).sum()}")
    
    print(f"\nVisualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

