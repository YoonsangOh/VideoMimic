"""
Stage 3 visualization with scene mesh + point cloud + SMPL in a single viser view.

This script is built on top of optimization_results_visualization.py.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import smplx
import torch
import trimesh
import tyro

try:
    # Works when executed as: python -m visualization.stage3_scene_mesh_visualization
    from visualization.optimization_results_visualization import (
        OptimizationResultsVisualizer,
        load_dict_from_hdf5,
    )
except ModuleNotFoundError:
    # Works when executed as: python visualization/stage3_scene_mesh_visualization.py
    from optimization_results_visualization import (  # type: ignore
        OptimizationResultsVisualizer,
        load_dict_from_hdf5,
    )


def main(
    world_env_path: str,
    mesh_path: str,
    bg_pc_downsample_factor: int = 4,
    camera_frustum_scale: float = 0.1,
    apply_rot_180: bool = True,
    gender: str = "male",
) -> None:
    """
    Args:
        world_env_path: Path to gravity_calibrated_megahunter.h5 (Stage 3 output)
        mesh_path: Path to background_mesh.obj (Stage 3 output)
        bg_pc_downsample_factor: Background point cloud downsample factor
        camera_frustum_scale: Camera frustum scale in viser
        apply_rot_180: Match optimization viewer coordinate convention
        gender: SMPL gender for mesh reconstruction
    """
    world_env_path = str(Path(world_env_path))
    mesh_path = str(Path(mesh_path))

    with h5py.File(world_env_path, "r") as f:
        world_env_and_human = load_dict_from_hdf5(f)

    world_env = world_env_and_human["our_pred_world_cameras_and_structure"]
    human_params_in_world = world_env_and_human.get("our_pred_humans_smplx_params", {})
    person_frame_info_list = world_env_and_human.get("person_frame_info_list", {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    smpl_batch_layer_dict: Dict[str, Any] = {}
    human_verts_dict: Dict[str, np.ndarray] = {}
    human_joints_dict: Dict[str, np.ndarray] = {}

    for person_id in human_params_in_world.keys():
        num_frames = len(human_params_in_world[person_id]["body_pose"])
        smpl_batch_layer_dict[person_id] = smplx.create(
            model_path="./assets/body_models",
            model_type="smpl",
            gender=gender,
            num_betas=10,
            batch_size=num_frames,
        ).to(device)

        smpl_betas = torch.from_numpy(human_params_in_world[person_id]["betas"]).float().to(device)
        if smpl_betas.ndim == 1:
            smpl_betas = smpl_betas.repeat(num_frames, 1)

        smpl_output = smpl_batch_layer_dict[person_id](
            body_pose=torch.from_numpy(human_params_in_world[person_id]["body_pose"]).float().to(device),
            betas=smpl_betas,
            global_orient=torch.from_numpy(human_params_in_world[person_id]["global_orient"]).float().to(device),
            pose2rot=False,
        )

        smpl_joints = smpl_output["joints"]
        smpl_root_joint = smpl_joints[:, 0:1, :]
        root_transl = torch.from_numpy(human_params_in_world[person_id]["root_transl"]).float().to(device)

        smpl_verts = smpl_output["vertices"] - smpl_root_joint + root_transl
        smpl_joints = smpl_joints - smpl_root_joint + root_transl

        human_verts_dict[person_id] = smpl_verts.detach().cpu().numpy()
        human_joints_dict[person_id] = smpl_joints.detach().cpu().numpy()

    human_verts_by_frame: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    human_joints_by_frame: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)

    for person_id in human_params_in_world.keys():
        frame_info = person_frame_info_list.get(person_id, None)
        if frame_info is None:
            continue
        person_frame_names = frame_info.astype(str)
        for idx, frame_name in enumerate(person_frame_names):
            frame_name = frame_name.item() if hasattr(frame_name, "item") else str(frame_name)
            human_verts_by_frame[frame_name][person_id] = human_verts_dict[person_id][idx]
            human_joints_by_frame[frame_name][person_id] = human_joints_dict[person_id][idx]

    visualizer = OptimizationResultsVisualizer(
        world_env=world_env,
        world_scale_factor=1.0,
        bg_pc_downsample_factor=bg_pc_downsample_factor,
        camera_frustum_scale=camera_frustum_scale,
        apply_rot_180=apply_rot_180,
    )

    smpl_faces = None
    if human_params_in_world:
        first_person = next(iter(human_params_in_world.keys()))
        smpl_faces = smpl_batch_layer_dict[first_person].faces

    visualizer.create_visualization(
        smpl_joints_3d_in_world=human_joints_by_frame if human_joints_by_frame else None,
        smplx_vertices_dict=human_verts_by_frame if human_verts_by_frame else None,
        smplx_faces=smpl_faces,
        contact_estimation=None,
    )

    bg_mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(bg_mesh, trimesh.Trimesh) or len(bg_mesh.faces) == 0:
        raise ValueError(f"Invalid mesh file: {mesh_path}")

    bg_vertices = np.asarray(bg_mesh.vertices).copy()
    if apply_rot_180:
        bg_vertices = bg_vertices @ visualizer.rot_180

    bg_mesh_handle = visualizer.server.scene.add_mesh_simple(
        name="/bg_mesh",
        vertices=bg_vertices,
        faces=np.asarray(bg_mesh.faces),
        color=(200, 200, 200),
        opacity=1.0,
        material="standard",
        flat_shading=False,
        side="double",
        visible=True,
    )

    with visualizer.server.gui.add_folder("Scene Mesh"):
        gui_show_bg_mesh = visualizer.server.gui.add_checkbox("Show Bg Mesh", True)

    @gui_show_bg_mesh.on_update
    def _(_) -> None:
        bg_mesh_handle.visible = gui_show_bg_mesh.value

    visualizer.run_playback_loop()


if __name__ == "__main__":
    tyro.cli(main)
