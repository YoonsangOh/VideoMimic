#!/usr/bin/env python3
"""
Visualize contact prediction results from BSTRO model.
This script creates visualizations showing contact predictions overlaid on RGB images.
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
    real2sim_root = Path(__file__).parent.parent
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

def project_vertices_to_image(vertices_3d, cam2world, intrinsics, img_h, img_w):
    """
    Project 3D vertices to 2D image coordinates.
    
    Args:
        vertices_3d: (N, 3) 3D vertices in world coordinates
        cam2world: (4, 4) camera to world transformation matrix
        intrinsics: (3, 3) camera intrinsics matrix
        img_h: Image height
        img_w: Image width
        
    Returns:
        projected_2d: (N, 2) 2D image coordinates, valid_mask: (N,) boolean mask
    """
    # Convert to camera coordinates
    world2cam = np.linalg.inv(cam2world)
    
    # Transform vertices to homogeneous coordinates
    vertices_homo = np.concatenate([vertices_3d, np.ones((vertices_3d.shape[0], 1))], axis=1)  # (N, 4)
    
    # Transform to camera space
    vertices_cam = (world2cam @ vertices_homo.T).T[:, :3]  # (N, 3)
    
    # Check which vertices are in front of camera
    valid_mask = vertices_cam[:, 2] > 0
    
    # Project to image space
    vertices_cam_valid = vertices_cam[valid_mask]
    if len(vertices_cam_valid) == 0:
        return np.zeros((vertices_3d.shape[0], 2)), np.zeros(vertices_3d.shape[0], dtype=bool)
    
    projected = (intrinsics @ vertices_cam_valid.T).T  # (N, 3)
    projected_2d = projected[:, :2] / (projected[:, 2:3] + 1e-8)  # (N, 2)
    
    # Create full array
    projected_full = np.zeros((vertices_3d.shape[0], 2))
    projected_full[valid_mask] = projected_2d
    
    # Check if projected points are within image bounds
    in_bounds = (projected_full[:, 0] >= 0) & (projected_full[:, 0] < img_w) & \
                (projected_full[:, 1] >= 0) & (projected_full[:, 1] < img_h)
    
    final_valid_mask = valid_mask & in_bounds
    
    return projected_full, final_valid_mask

def visualize_contact_on_image(
    rgb_img: np.ndarray,
    contact_data: dict,
    person_id: int = 1,
    left_foot_vert_ids: np.ndarray = None,
    right_foot_vert_ids: np.ndarray = None,
    smpl_vertices: np.ndarray = None,
    cam2world: np.ndarray = None,
    intrinsics: np.ndarray = None,
    show_heatmap: bool = True
) -> np.ndarray:
    """
    Visualize contact predictions on RGB image.
    
    Args:
        rgb_img: RGB image (H, W, 3)
        contact_data: Contact prediction data dict
        person_id: Person ID to visualize
        left_foot_vert_ids: Left foot vertex IDs
        right_foot_vert_ids: Right foot vertex IDs
        smpl_vertices: (6890, 3) SMPL vertices in world coordinates
        cam2world: (4, 4) camera to world transformation
        intrinsics: (3, 3) camera intrinsics
        show_heatmap: Whether to show contact heatmap overlay
        
    Returns:
        Visualization image with contact overlay
    """
    vis_img = rgb_img.copy()
    h, w = rgb_img.shape[:2]
    
    if person_id not in contact_data:
        return vis_img
    
    person_data = contact_data[person_id]
    left_contact = person_data.get('left_foot_contact', False)
    right_contact = person_data.get('right_foot_contact', False)
    
    # Create overlay
    overlay = vis_img.copy()
    
    # Draw contact indicators - CENTERED and LARGER
    center_x, center_y = w // 2, h // 2
    
    # Left foot indicator (center left)
    left_color = (0, 255, 0) if left_contact else (128, 128, 128)
    left_radius = 40  # Larger radius
    cv2.circle(overlay, (center_x - 150, center_y), left_radius, left_color, -1)
    cv2.circle(overlay, (center_x - 150, center_y), left_radius, (255, 255, 255), 3)  # White border
    cv2.putText(overlay, 'L', (center_x - 165, center_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Right foot indicator (center right)
    right_color = (0, 255, 0) if right_contact else (128, 128, 128)
    right_radius = 40  # Larger radius
    cv2.circle(overlay, (center_x + 150, center_y), right_radius, right_color, -1)
    cv2.circle(overlay, (center_x + 150, center_y), right_radius, (255, 255, 255), 3)  # White border
    cv2.putText(overlay, 'R', (center_x + 135, center_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Add text labels below indicators
    left_text = f"Left: {'CONTACT' if left_contact else 'NO CONTACT'}"
    right_text = f"Right: {'CONTACT' if right_contact else 'NO CONTACT'}"
    
    cv2.putText(overlay, left_text, (center_x - 200, center_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, left_color, 2)
    cv2.putText(overlay, right_text, (center_x + 50, center_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, right_color, 2)
    
    # Contact heatmap overlay - Enhanced with multiple visualization methods
    if show_heatmap and 'frame_contact_vertices' in person_data and smpl_vertices is not None and cam2world is not None and intrinsics is not None:
        contact_vertices = person_data['frame_contact_vertices']  # (6890, 1)
        contact_probs = contact_vertices.flatten()
        
        # Project SMPL vertices to image
        projected_2d, valid_mask = project_vertices_to_image(smpl_vertices, cam2world, intrinsics, h, w)
        
        # Create enhanced heatmap overlay
        heatmap_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        valid_indices = np.where(valid_mask)[0]
        
        # Method 1: Draw larger, brighter points with better color mapping
        for i in valid_indices:
            prob = float(contact_probs[i])
            x, y = int(projected_2d[i, 0]), int(projected_2d[i, 1])
            
            if 0 <= x < w and 0 <= y < h:
                # Lower threshold to show more contact areas
                if prob > 0.15:  # Show even low probabilities
                    # Normalize probability for better visualization
                    prob_norm = np.clip((prob - 0.15) / 0.85, 0, 1)
                    
                    # Turbo colormap approximation: blue -> cyan -> green -> yellow -> red
                    if prob_norm > 0.8:
                        # Very high: bright red
                        color = (0, 0, 255)
                        intensity = 255
                    elif prob_norm > 0.6:
                        # High: yellow to red
                        t = (prob_norm - 0.6) / 0.2
                        color = (0, int(255 * (1 - t)), 255)
                        intensity = 255
                    elif prob_norm > 0.4:
                        # Medium-high: green to yellow
                        t = (prob_norm - 0.4) / 0.2
                        color = (0, 255, int(255 * t))
                        intensity = 255
                    elif prob_norm > 0.2:
                        # Medium: cyan to green
                        t = (prob_norm - 0.2) / 0.2
                        color = (int(255 * (1 - t)), 255, 0)
                        intensity = int(200 + 55 * t)
                    else:
                        # Low: blue to cyan
                        t = prob_norm / 0.2
                        color = (int(255 * (1 - t)), int(255 * t), 0)
                        intensity = int(150 * prob_norm)
                    
                    # Much larger point sizes for visibility
                    point_size = max(4, int(10 * prob_norm + 4))
                    cv2.circle(heatmap_overlay, (x, y), point_size, color, -1)
                    # Add a brighter center for high probabilities
                    if prob_norm > 0.5:
                        cv2.circle(heatmap_overlay, (x, y), max(2, point_size // 2), (255, 255, 255), -1)
        
        # Apply strong Gaussian blur for smoother, more visible heatmap
        heatmap_overlay = cv2.GaussianBlur(heatmap_overlay, (21, 21), 0)
        
        # Create a second overlay with even stronger colors
        heatmap_overlay2 = np.zeros((h, w, 3), dtype=np.uint8)
        for i in valid_indices:
            prob = float(contact_probs[i])
            x, y = int(projected_2d[i, 0]), int(projected_2d[i, 1])
            
            if 0 <= x < w and 0 <= y < h and prob > 0.3:
                # Only show high contact areas in this overlay
                prob_norm = np.clip((prob - 0.3) / 0.7, 0, 1)
                # Bright red/yellow for high contact
                if prob_norm > 0.5:
                    color = (0, int(255 * (1 - prob_norm) * 2), 255)  # Yellow to red
                else:
                    color = (0, 255, int(255 * prob_norm * 2))  # Green to yellow
                
                point_size = max(6, int(15 * prob_norm + 6))
                cv2.circle(heatmap_overlay2, (x, y), point_size, color, -1)
        
        heatmap_overlay2 = cv2.GaussianBlur(heatmap_overlay2, (25, 25), 0)
        
        # Blend both overlays with stronger alpha
        alpha1 = 0.7
        vis_img = cv2.addWeighted(vis_img, 1 - alpha1, heatmap_overlay, alpha1, 0)
        alpha2 = 0.5
        vis_img = cv2.addWeighted(vis_img, 1 - alpha2, heatmap_overlay2, alpha2, 0)
        
        # Get foot contact probabilities for display
        if left_foot_vert_ids is not None and right_foot_vert_ids is not None:
            left_foot_prob = contact_probs[left_foot_vert_ids].mean()
            right_foot_prob = contact_probs[right_foot_vert_ids].mean()
            
            # Display probabilities
            left_prob_text = f"L: {left_foot_prob:.2f}"
            right_prob_text = f"R: {right_foot_prob:.2f}"
            cv2.putText(overlay, left_prob_text, (center_x - 200, center_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlay, right_prob_text, (center_x + 50, center_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Blend overlay
    alpha = 0.8
    vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0)
    
    return vis_img

def create_contact_visualization(
    contact_dir: str,
    rgb_dir: str,
    output_dir: str,
    person_id: int = 1,
    create_video: bool = False,
    fps: int = 30,
    megahunter_path: str = None,
    show_heatmap: bool = True
):
    """
    Create contact visualization for all frames.
    
    Args:
        contact_dir: Directory containing contact .pkl files
        rgb_dir: Directory containing RGB images
        output_dir: Output directory for visualizations
        person_id: Person ID to visualize
        create_video: Whether to create video
        fps: Frames per second for video
        megahunter_path: Path to megahunter H5 file for SMPL vertices and camera params
        show_heatmap: Whether to show contact heatmap overlay
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load SMPL vertex segmentation
    left_foot_vert_ids, right_foot_vert_ids = load_smpl_vert_segmentation()
    
    # Load megahunter data if available
    smpl_vertices_dict = {}
    cam_params_dict = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if megahunter_path and os.path.exists(megahunter_path) and show_heatmap:
        print(f"Loading megahunter data from {megahunter_path}...")
        try:
            with h5py.File(megahunter_path, 'r') as f:
                megahunter_data = load_dict_from_hdf5(f)
            
            world_env = megahunter_data['our_pred_world_cameras_and_structure']
            human_params = megahunter_data['our_pred_humans_smplx_params']
            
            # Get person ID from data
            available_person_ids = list(human_params.keys())
            if str(person_id) in available_person_ids:
                person_key = str(person_id)
            elif len(available_person_ids) > 0:
                person_key = available_person_ids[0]
                print(f"Person ID {person_id} not found, using {person_key}")
            else:
                print("No person data found in megahunter file")
                megahunter_path = None
            
            if megahunter_path:
                # Load SMPL model
                real2sim_root = Path(__file__).parent.parent
                model_path = real2sim_root / 'assets' / 'body_models'
                
                num_frames = human_params[person_key]['body_pose'].shape[0]
                smpl_model = smplx.create(
                    model_path=str(model_path),
                    model_type='smpl',
                    gender='male',
                    num_betas=10,
                    batch_size=num_frames
                ).to(device)
                
                # Generate SMPL vertices
                smpl_betas = torch.from_numpy(human_params[person_key]['betas'].astype(np.float32)).to(device)
                if smpl_betas.ndim == 1:
                    smpl_betas = smpl_betas.repeat(num_frames, 1)
                
                smpl_output = smpl_model(
                    body_pose=torch.from_numpy(human_params[person_key]['body_pose'].astype(np.float32)).to(device),
                    betas=smpl_betas,
                    global_orient=torch.from_numpy(human_params[person_key]['global_orient'].astype(np.float32)).to(device),
                    pose2rot=False
                )
                
                smpl_joints = smpl_output['joints']
                smpl_root_joint = smpl_joints[:, 0:1, :]
                smpl_verts = smpl_output['vertices'] - smpl_root_joint + torch.from_numpy(human_params[person_key]['root_transl']).to(device)
                smpl_verts = smpl_verts.detach().cpu().numpy()
                
                # Get frame names
                frame_names = megahunter_data['person_frame_info_list'][person_key].astype(str)
                
                # Store SMPL vertices and camera params for each frame
                for i, frame_name in enumerate(frame_names):
                    frame_name = frame_name.item()
                    smpl_vertices_dict[frame_name] = smpl_verts[i]
                    
                    if frame_name in world_env:
                        cam2world = world_env[frame_name]['cam2world']
                        intrinsics = world_env[frame_name].get('intrinsic', world_env[frame_name].get('K', None))
                        if intrinsics is None:
                            # Estimate intrinsics from image size
                            rgb_img = world_env[frame_name]['rgbimg']
                            img_h, img_w = rgb_img.shape[:2]
                            f = max(img_h, img_w) * 1.2
                            intrinsics = np.array([[f, 0, img_w/2], [0, f, img_h/2], [0, 0, 1]], dtype=np.float32)
                        
                        cam_params_dict[frame_name] = {
                            'cam2world': cam2world,
                            'intrinsics': intrinsics
                        }
                
                print(f"Loaded SMPL data for {len(smpl_vertices_dict)} frames")
        except Exception as e:
            print(f"Warning: Could not load megahunter data: {e}")
            print("Continuing without heatmap visualization...")
            show_heatmap = False
    
    # Get all contact files
    contact_files = sorted(glob.glob(os.path.join(contact_dir, '*.pkl')))
    
    if len(contact_files) == 0:
        print(f"Error: No contact files found in {contact_dir}")
        return
    
    print(f"Found {len(contact_files)} contact files")
    
    # Get RGB image files
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.jpg')))
    if len(rgb_files) == 0:
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
    
    if len(rgb_files) == 0:
        print(f"Warning: No RGB images found in {rgb_dir}")
        print("Creating visualizations without RGB images...")
        rgb_files = [None] * len(contact_files)
    
    # Create mapping from frame name to RGB file
    frame_to_rgb = {}
    for rgb_file in rgb_files:
        if rgb_file is None:
            continue
        frame_name = Path(rgb_file).stem
        frame_to_rgb[frame_name] = rgb_file
    
    # Initialize video writers if needed
    video_writer = None
    heatmap_video_writer = None
    combined_video_writer = None
    if create_video:
        # Get image dimensions from first frame
        first_rgb = None
        for rgb_file in rgb_files:
            if rgb_file is not None:
                first_rgb = cv2.imread(rgb_file)
                break
        
        if first_rgb is not None:
            h, w = first_rgb.shape[:2]
            video_path = os.path.join(output_dir, "contact_visualization.mp4")
            video_writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (w, h)
            )
            print(f"Creating video: {video_path}")
            
            # Create separate heatmap video
            if show_heatmap:
                heatmap_video_path = os.path.join(output_dir, "contact_heatmap_only.mp4")
                heatmap_video_writer = cv2.VideoWriter(
                    heatmap_video_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (w, h)
                )
                print(f"Creating heatmap video: {heatmap_video_path}")
                
                # Create combined side-by-side video
                combined_video_path = os.path.join(output_dir, "contact_combined.mp4")
                combined_video_writer = cv2.VideoWriter(
                    combined_video_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (w * 2, h)
                )
                print(f"Creating combined video: {combined_video_path}")
    
    # Statistics
    left_contact_frames = 0
    right_contact_frames = 0
    total_frames = 0
    
    # Process each contact file
    for contact_file in tqdm(contact_files, desc="Processing frames"):
        frame_name = Path(contact_file).stem
        
        # Load contact data
        with open(contact_file, 'rb') as f:
            contact_data = pickle.load(f)
        
        if person_id not in contact_data:
            continue
        
        total_frames += 1
        person_data = contact_data[person_id]
        
        left_contact = person_data.get('left_foot_contact', False)
        right_contact = person_data.get('right_foot_contact', False)
        
        if left_contact:
            left_contact_frames += 1
        if right_contact:
            right_contact_frames += 1
        
        # Load RGB image if available
        rgb_img = None
        if frame_name in frame_to_rgb:
            rgb_img = cv2.imread(frame_to_rgb[frame_name])
            if rgb_img is not None:
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        else:
            # Create blank image if RGB not available
            rgb_img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Get SMPL vertices and camera params for this frame
        smpl_verts = smpl_vertices_dict.get(frame_name)
        cam_params = cam_params_dict.get(frame_name)
        
        # Create visualization
        vis_img = visualize_contact_on_image(
            rgb_img,
            contact_data,
            person_id,
            left_foot_vert_ids,
            right_foot_vert_ids,
            smpl_vertices=smpl_verts,
            cam2world=cam_params['cam2world'] if cam_params else None,
            intrinsics=cam_params['intrinsics'] if cam_params else None,
            show_heatmap=show_heatmap and smpl_verts is not None and cam_params is not None
        )
        
        # Create side-by-side visualization with separate heatmap panel
        if show_heatmap and smpl_verts is not None and cam_params is not None and 'frame_contact_vertices' in contact_data[person_id]:
            # Create separate heatmap visualization
            contact_probs = contact_data[person_id]['frame_contact_vertices'].flatten()
            projected_2d, valid_mask = project_vertices_to_image(
                smpl_verts, 
                cam_params['cam2world'], 
                cam_params['intrinsics'], 
                h, w
            )
            
            # Create standalone heatmap image
            heatmap_standalone = np.zeros((h, w, 3), dtype=np.uint8)
            valid_indices = np.where(valid_mask)[0]
            
            for i in valid_indices:
                prob = float(contact_probs[i])
                x, y = int(projected_2d[i, 0]), int(projected_2d[i, 1])
                
                if 0 <= x < w and 0 <= y < h and prob > 0.1:
                    prob_norm = np.clip((prob - 0.1) / 0.9, 0, 1)
                    
                    # Turbo colormap: blue -> cyan -> green -> yellow -> red
                    if prob_norm > 0.8:
                        color = (0, 0, 255)  # Red
                    elif prob_norm > 0.6:
                        color = (0, int(255 * (1 - (prob_norm - 0.6) / 0.2)), 255)  # Yellow to red
                    elif prob_norm > 0.4:
                        color = (0, 255, int(255 * (prob_norm - 0.4) / 0.2))  # Green to yellow
                    elif prob_norm > 0.2:
                        color = (int(255 * (1 - (prob_norm - 0.2) / 0.2)), 255, 0)  # Cyan to green
                    else:
                        color = (255, int(255 * prob_norm / 0.2), 0)  # Blue to cyan
                    
                    # Much larger points for better visibility
                    point_size = max(8, int(20 * prob_norm + 8))
                    cv2.circle(heatmap_standalone, (x, y), point_size, color, -1)
                    # Add bright white center for high probabilities
                    if prob_norm > 0.5:
                        cv2.circle(heatmap_standalone, (x, y), max(4, point_size // 3), (255, 255, 255), -1)
            
            # Stronger blur for smoother heatmap
            heatmap_standalone = cv2.GaussianBlur(heatmap_standalone, (35, 35), 0)
            
            # Create side-by-side comparison
            combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
            combined[:, :w] = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            combined[:, w:] = heatmap_standalone
            
            # Add labels with background for readability
            cv2.rectangle(combined, (5, 5), (200, 40), (0, 0, 0), -1)
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.rectangle(combined, (w + 5, 5), (w + 350, 40), (0, 0, 0), -1)
            cv2.putText(combined, "Contact Heatmap", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Add colorbar legend at bottom
            legend_height = 40
            legend_y = h - legend_height - 20
            legend_width = w - 40
            legend_x = w + 20
            for i in range(legend_width):
                prob_val = i / legend_width
                if prob_val > 0.8:
                    color = (0, 0, 255)  # Red
                elif prob_val > 0.6:
                    t = (prob_val - 0.6) / 0.2
                    color = (0, int(255 * (1 - t)), 255)  # Yellow to red
                elif prob_val > 0.4:
                    t = (prob_val - 0.4) / 0.2
                    color = (0, 255, int(255 * t))  # Green to yellow
                elif prob_val > 0.2:
                    t = (prob_val - 0.2) / 0.2
                    color = (int(255 * (1 - t)), 255, 0)  # Cyan to green
                else:
                    t = prob_val / 0.2
                    color = (int(255 * (1 - t)), int(255 * t), 0)  # Blue to cyan
                cv2.line(combined, (legend_x + i, legend_y), (legend_x + i, legend_y + legend_height), color, 2)
            
            # Add legend text
            cv2.rectangle(combined, (legend_x - 5, legend_y + legend_height + 5), (legend_x + legend_width + 5, legend_y + legend_height + 30), (0, 0, 0), -1)
            cv2.putText(combined, "Low Contact", (legend_x, legend_y + legend_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "High Contact", (legend_x + legend_width - 150, legend_y + legend_height + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save combined image
            output_path_combined = os.path.join(output_dir, f"{frame_name}_contact_combined.png")
            cv2.imwrite(output_path_combined, combined)
            
            # Also save standalone heatmap
            output_path_heatmap = os.path.join(output_dir, f"{frame_name}_heatmap_only.png")
            cv2.imwrite(output_path_heatmap, heatmap_standalone)
            
            # Add to videos
            if create_video:
                if heatmap_video_writer is not None:
                    heatmap_video_writer.write(heatmap_standalone)
                if combined_video_writer is not None:
                    combined_video_writer.write(combined)
        
        # Convert back to BGR for saving
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        
        # Save image
        output_path = os.path.join(output_dir, f"{frame_name}_contact.png")
        cv2.imwrite(output_path, vis_img_bgr)
        
        # Add to video
        if create_video and video_writer is not None:
            video_writer.write(vis_img_bgr)
    
    # Release video writers
    if create_video:
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved: {os.path.join(output_dir, 'contact_visualization.mp4')}")
        if heatmap_video_writer is not None:
            heatmap_video_writer.release()
            print(f"Heatmap video saved: {os.path.join(output_dir, 'contact_heatmap_only.mp4')}")
        if combined_video_writer is not None:
            combined_video_writer.release()
            print(f"Combined video saved: {os.path.join(output_dir, 'contact_combined.mp4')}")
    
    # Print statistics
    print(f"\nContact Statistics:")
    print(f"  Total frames: {total_frames}")
    print(f"  Left foot contact frames: {left_contact_frames}/{total_frames} ({100*left_contact_frames/total_frames:.1f}%)")
    print(f"  Right foot contact frames: {right_contact_frames}/{total_frames} ({100*right_contact_frames/total_frames:.1f}%)")
    print(f"\nVisualizations saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Visualize contact prediction results")
    parser.add_argument("--contact-dir", type=str, required=True, help="Directory containing contact .pkl files")
    parser.add_argument("--rgb-dir", type=str, required=True, help="Directory containing RGB images")
    parser.add_argument("--output-dir", type=str, default="./contact_visualizations", help="Output directory")
    parser.add_argument("--person-id", type=int, default=1, help="Person ID to visualize")
    parser.add_argument("--create-video", action="store_true", help="Create video from visualizations")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for video")
    parser.add_argument("--megahunter-path", type=str, default=None, help="Path to megahunter H5 file for SMPL vertices (enables heatmap)")
    parser.add_argument("--no-heatmap", action="store_true", help="Disable contact heatmap overlay")
    
    args = parser.parse_args()
    
    create_contact_visualization(
        args.contact_dir,
        args.rgb_dir,
        args.output_dir,
        args.person_id,
        args.create_video,
        args.fps,
        args.megahunter_path,
        show_heatmap=not args.no_heatmap
    )

if __name__ == "__main__":
    main()

