#!/usr/bin/env python3
"""
Visualize contact prediction results from BSTRO model - Final Version.
Large body heatmap panel centered on the right side of the screen.
"""

import pickle
import numpy as np
import cv2
import os
import argparse
import glob
import sys
from pathlib import Path
from tqdm import tqdm
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_preview_compat import CompatibleVideoWriter


def load_smpl_vert_segmentation():
    """Load SMPL vertex segmentation for body regions."""
    real2sim_root = Path(__file__).parent
    smpl_vert_seg_path = real2sim_root / "assets" / "body_models" / "smpl" / "smpl_vert_segmentation.json"

    if not smpl_vert_seg_path.exists():
        print(f"Warning: SMPL vertex segmentation not found at {smpl_vert_seg_path}")
        return None, None, None

    with open(smpl_vert_seg_path, 'r') as f:
        smpl_vert_seg = json.load(f)

    left_foot_vert_ids = np.array(smpl_vert_seg['leftFoot'], dtype=np.int32)
    right_foot_vert_ids = np.array(smpl_vert_seg['rightFoot'], dtype=np.int32)

    return left_foot_vert_ids, right_foot_vert_ids, smpl_vert_seg


def get_turbo_color(prob_norm):
    """Get turbo colormap color for probability (returns BGR)."""
    prob_norm = float(np.clip(prob_norm, 0, 1))
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


def create_large_body_panel(contact_probs, smpl_vert_seg, panel_width, panel_height,
                            left_contact, right_contact, left_prob, right_prob):
    """
    Create a large body-shaped heatmap panel with detailed body parts.

    Args:
        contact_probs: (6890,) contact probabilities
        smpl_vert_seg: SMPL vertex segmentation dictionary
        panel_width: Width of the panel
        panel_height: Height of the panel
        left_contact: Boolean for left foot contact
        right_contact: Boolean for right foot contact
        left_prob: Left foot probability
        right_prob: Right foot probability

    Returns:
        panel: (panel_height, panel_width, 3) BGR image
    """
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)

    # Dark background
    cv2.rectangle(panel, (0, 0), (panel_width, panel_height), (25, 25, 25), -1)

    # Calculate scale factors
    base_width, base_height = 300, 600  # Base reference size
    scale_x = panel_width / base_width
    scale_y = panel_height / base_height
    scale = min(scale_x, scale_y) * 0.85  # Leave some margin

    # Center offset
    cx = panel_width // 2
    cy = int(panel_height * 0.45)  # Slightly above center

    # Body part definitions using SMPL segmentation
    # Format: (name, center_x_offset, center_y_offset, width, height, indices_key or range)
    body_parts = [
        # Head
        ('head', 0, -220, 70, 90, 'head'),
        # Neck
        ('neck', 0, -165, 35, 30, None),
        # Torso (split into upper and lower)
        ('upper_torso', 0, -110, 100, 80, 'spine1'),
        ('lower_torso', 0, -30, 90, 80, 'spine'),
        ('hips', 0, 40, 100, 50, 'hips'),
        # Arms
        ('leftUpperArm', -85, -110, 35, 70, 'leftArm'),
        ('rightUpperArm', 85, -110, 35, 70, 'rightArm'),
        ('leftForeArm', -100, -35, 30, 70, 'leftForeArm'),
        ('rightForeArm', 100, -35, 30, 70, 'rightForeArm'),
        ('leftHand', -110, 40, 30, 50, 'leftHand'),
        ('rightHand', 110, 40, 30, 50, 'rightHand'),
        # Legs
        ('leftUpLeg', -40, 95, 50, 100, 'leftUpLeg'),
        ('rightUpLeg', 40, 95, 50, 100, 'rightUpLeg'),
        ('leftLeg', -45, 200, 45, 100, 'leftLeg'),
        ('rightLeg', 45, 200, 45, 100, 'rightLeg'),
        # Feet
        ('leftFoot', -50, 280, 45, 40, 'leftFoot'),
        ('rightFoot', 50, 280, 45, 40, 'rightFoot'),
        ('leftToeBase', -55, 310, 40, 25, 'leftToeBase'),
        ('rightToeBase', 55, 310, 40, 25, 'rightToeBase'),
    ]

    # Draw each body part
    for part_name, ox, oy, w, h, seg_key in body_parts:
        # Scale positions and sizes
        x = int(cx + ox * scale)
        y = int(cy + oy * scale)
        w_scaled = int(w * scale)
        h_scaled = int(h * scale)

        # Get vertex indices for this part
        if seg_key and smpl_vert_seg and seg_key in smpl_vert_seg:
            indices = smpl_vert_seg[seg_key]
            valid_indices = [i for i in indices if i < len(contact_probs)]
            if valid_indices:
                part_probs = contact_probs[valid_indices]
                max_prob = float(np.max(part_probs))
                avg_prob = float(np.mean(part_probs))
                # Use weighted combination of max and mean for color
                prob_for_color = 0.7 * max_prob + 0.3 * avg_prob
            else:
                prob_for_color = 0.0
        else:
            # For parts without direct mapping, use nearby regions
            prob_for_color = 0.1  # Default low

        color = get_turbo_color(prob_for_color)

        # Draw rounded rectangle (approximate with filled rect)
        x1, y1 = x - w_scaled // 2, y - h_scaled // 2
        x2, y2 = x + w_scaled // 2, y + h_scaled // 2

        # Ensure within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(panel_width, x2), min(panel_height, y2)

        cv2.rectangle(panel, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (80, 80, 80), 1)

    # Draw connections between body parts (skeleton lines)
    skeleton_color = (60, 60, 60)
    connections = [
        (0, -220, 0, -165),  # Head to neck
        (0, -165, 0, -110),  # Neck to upper torso
        (0, -110, 0, -30),   # Upper to lower torso
        (0, -30, 0, 40),     # Lower torso to hips
        (0, -110, -85, -110), # Torso to left arm
        (0, -110, 85, -110),  # Torso to right arm
        (-85, -110, -100, -35), # Left upper to forearm
        (85, -110, 100, -35),   # Right upper to forearm
        (-100, -35, -110, 40),  # Left forearm to hand
        (100, -35, 110, 40),    # Right forearm to hand
        (0, 40, -40, 95),    # Hips to left leg
        (0, 40, 40, 95),     # Hips to right leg
        (-40, 95, -45, 200), # Left upper to lower leg
        (40, 95, 45, 200),   # Right upper to lower leg
        (-45, 200, -50, 280), # Left lower leg to foot
        (45, 200, 50, 280),   # Right lower leg to foot
    ]

    for x1, y1, x2, y2 in connections:
        pt1 = (int(cx + x1 * scale), int(cy + y1 * scale))
        pt2 = (int(cx + x2 * scale), int(cy + y2 * scale))
        cv2.line(panel, pt1, pt2, skeleton_color, 2)

    # Add title
    title_y = 30
    cv2.putText(panel, "BSTRO Contact Prediction", (10, title_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Add colorbar on the right side
    colorbar_x = panel_width - 50
    colorbar_y = 60
    colorbar_h = 200
    colorbar_w = 20

    for i in range(colorbar_h):
        prob = 1.0 - (i / colorbar_h)
        color = get_turbo_color(prob)
        cv2.line(panel, (colorbar_x, colorbar_y + i),
                 (colorbar_x + colorbar_w, colorbar_y + i), color, 1)

    # Colorbar border and labels
    cv2.rectangle(panel, (colorbar_x, colorbar_y),
                  (colorbar_x + colorbar_w, colorbar_y + colorbar_h), (150, 150, 150), 1)
    cv2.putText(panel, "1.0", (colorbar_x - 5, colorbar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(panel, "0.0", (colorbar_x - 5, colorbar_y + colorbar_h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Foot contact indicators at the bottom
    foot_y = panel_height - 80
    indicator_size = 35

    # Left foot indicator
    left_color = (0, 255, 0) if left_contact else (80, 80, 80)
    left_x = panel_width // 2 - 70
    cv2.circle(panel, (left_x, foot_y), indicator_size, left_color, -1)
    cv2.circle(panel, (left_x, foot_y), indicator_size, (200, 200, 200), 2)
    cv2.putText(panel, "L", (left_x - 12, foot_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(panel, f"{left_prob:.2f}", (left_x - 25, foot_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Right foot indicator
    right_color = (0, 255, 0) if right_contact else (80, 80, 80)
    right_x = panel_width // 2 + 70
    cv2.circle(panel, (right_x, foot_y), indicator_size, right_color, -1)
    cv2.circle(panel, (right_x, foot_y), indicator_size, (200, 200, 200), 2)
    cv2.putText(panel, "R", (right_x - 12, foot_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(panel, f"{right_prob:.2f}", (right_x - 25, foot_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Contact label
    cv2.putText(panel, "Foot Contact", (panel_width // 2 - 55, foot_y - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Stats at the bottom
    stats_text = f"Max: {contact_probs.max():.2f} | Mean: {contact_probs.mean():.2f}"
    cv2.putText(panel, stats_text, (10, panel_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return panel


def visualize_frame(
    rgb_img: np.ndarray,
    contact_probs: np.ndarray,
    smpl_vert_seg: dict,
    left_contact: bool,
    right_contact: bool,
    left_foot_prob: float,
    right_foot_prob: float,
    frame_idx: int = 0
) -> np.ndarray:
    """
    Create visualization with large body heatmap panel on the right.

    Args:
        rgb_img: RGB image (H, W, 3)
        contact_probs: (6890,) contact probabilities
        smpl_vert_seg: SMPL vertex segmentation dictionary
        left_contact: Boolean for left foot contact
        right_contact: Boolean for right foot contact
        left_foot_prob: Average left foot probability
        right_foot_prob: Average right foot probability
        frame_idx: Frame index for display

    Returns:
        Visualization image (H, W, 3) BGR
    """
    h, w = rgb_img.shape[:2]

    # Panel takes up moderate portion of screen height (reduced from 0.85 to 0.70)
    panel_height = int(h * 0.40)
    panel_width = int(panel_height * 0.250)  # Aspect ratio for body shape

    # Ensure minimum size
    panel_width = max(panel_width, 200)
    panel_height = max(panel_height, 350)

    # Create body panel
    body_panel = create_large_body_panel(
        contact_probs, smpl_vert_seg, panel_width, panel_height,
        left_contact, right_contact, left_foot_prob, right_foot_prob
    )

    # Convert RGB to BGR
    vis_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    # Position: center-right of the screen
    margin = 20
    panel_x = w - panel_width - margin
    panel_y = (h - panel_height) // 2  # Vertically centered

    # Ensure panel fits
    panel_y = max(margin, panel_y)
    if panel_y + panel_height > h - margin:
        panel_height = h - panel_y - margin
        body_panel = cv2.resize(body_panel, (panel_width, panel_height))

    # Create semi-transparent overlay region
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (panel_x - 5, panel_y - 5),
                  (panel_x + panel_width + 5, panel_y + panel_height + 5),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)

    # Place panel
    vis_img[panel_y:panel_y + panel_height, panel_x:panel_x + panel_width] = body_panel

    # Draw border
    cv2.rectangle(vis_img, (panel_x - 2, panel_y - 2),
                  (panel_x + panel_width + 2, panel_y + panel_height + 2),
                  (100, 100, 100), 2)

    # Frame number at top-left
    cv2.rectangle(vis_img, (10, 10), (150, 45), (0, 0, 0), -1)
    cv2.putText(vis_img, f"Frame: {frame_idx}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return vis_img


def main():
    parser = argparse.ArgumentParser(description="Visualize contact prediction results (Final Version)")
    parser.add_argument("--contact-dir", type=str, required=True, help="Directory containing contact .pkl files")
    parser.add_argument("--rgb-dir", type=str, required=True, help="Directory containing RGB images")
    parser.add_argument("--output-dir", type=str, default="./contact_visualizations_final", help="Output directory")
    parser.add_argument("--person-id", type=int, default=1, help="Person ID to visualize")
    parser.add_argument("--create-video", action="store_true", help="Create video from visualizations")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for video")
    parser.add_argument("--save-images", action="store_true", help="Save individual frame images")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load SMPL vertex segmentation
    left_foot_vert_ids, right_foot_vert_ids, smpl_vert_seg = load_smpl_vert_segmentation()

    if smpl_vert_seg is None:
        print("Error: Could not load SMPL vertex segmentation")
        return

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
            continue

        person_data = contact_data[args.person_id]

        # Get contact probabilities
        if 'frame_contact_vertices' not in person_data:
            continue

        contact_probs = person_data['frame_contact_vertices'].flatten()
        all_contact_probs.append(contact_probs)

        # Get foot contact info
        left_contact = person_data.get('left_foot_contact', False)
        right_contact = person_data.get('right_foot_contact', False)

        # Calculate foot probabilities
        if left_foot_vert_ids is not None:
            left_foot_prob = float(contact_probs[left_foot_vert_ids].mean())
        else:
            left_foot_prob = 0.0

        if right_foot_vert_ids is not None:
            right_foot_prob = float(contact_probs[right_foot_vert_ids].mean())
        else:
            right_foot_prob = 0.0

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
            rgb_img = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Create visualization
        vis_img = visualize_frame(
            rgb_img,
            contact_probs,
            smpl_vert_seg,
            left_contact,
            right_contact,
            left_foot_prob,
            right_foot_prob,
            idx
        )

        h, w = vis_img.shape[:2]

        # Initialize video writer on first frame
        if args.create_video and not first_frame_processed:
            video_path = os.path.join(args.output_dir, "contact_visualization_final.mp4")
            video_writer = CompatibleVideoWriter(video_path, args.fps, (w, h), input_color="bgr")
            print(f"\nCreating video: {video_path} ({w}x{h})")
            first_frame_processed = True

        # Save image
        if args.save_images:
            output_path = os.path.join(args.output_dir, f"{frame_name}_contact.png")
            cv2.imwrite(output_path, vis_img)

        # Add to video
        if video_writer is not None:
            video_writer.write(vis_img)

        # Debug output for first few frames
        if idx < 3:
            print(f"\n  Frame {frame_name}:")
            print(f"    Contact probs range: [{contact_probs.min():.4f}, {contact_probs.max():.4f}]")
            print(f"    Left foot: contact={left_contact}, prob={left_foot_prob:.3f}")
            print(f"    Right foot: contact={right_contact}, prob={right_foot_prob:.3f}")

    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f"\nVideo saved: {os.path.join(args.output_dir, 'contact_visualization_final.mp4')}")

    # Print overall statistics
    if all_contact_probs:
        all_probs = np.concatenate(all_contact_probs)
        print(f"\n=== Overall Statistics ===")
        print(f"  Total frames processed: {total_frames}")
        print(f"  Left foot contact: {left_contact_frames}/{total_frames} ({100*left_contact_frames/total_frames:.1f}%)")
        print(f"  Right foot contact: {right_contact_frames}/{total_frames} ({100*right_contact_frames/total_frames:.1f}%)")
        print(f"  Contact probability - Min: {all_probs.min():.4f}, Max: {all_probs.max():.4f}, Mean: {all_probs.mean():.4f}")

    print(f"\nVisualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
