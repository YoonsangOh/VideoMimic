#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions_videomimic")
SIM_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SIM_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_preview_compat import CompatibleVideoWriter


def build_args(num_envs: int) -> SimpleNamespace:
    from isaacgym import gymapi

    return SimpleNamespace(
        physics_engine=gymapi.SIM_PHYSX,
        use_gpu=True,
        subscenes=0,
        use_gpu_pipeline=True,
        num_threads=0,
        sim_device="cuda:0",
        compute_device_id=0,
        sim_device_id=0,
        rl_device="cuda:0",
        headless=True,
        multi_gpu=False,
        no_use_wandb=True,
        wandb_note=None,
        max_iterations=None,
        resume=False,
        experiment_name=None,
        run_name=None,
        load_run=None,
        checkpoint=None,
        seed=1,
        num_envs=num_envs,
    )


def configure_env(env_cfg, clip_name: str, data_root: Path) -> None:
    env_cfg.env.num_envs = 1
    env_cfg.env.test = False
    env_cfg.env.offscreen_rendering = True
    env_cfg.viser.enable = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.n_rows = 1
    env_cfg.terrain.cast_mesh_to_heightfield = False
    env_cfg.deepmimic.use_amass = False
    env_cfg.deepmimic.use_human_videos = True
    env_cfg.deepmimic.human_motion_source = clip_name
    env_cfg.deepmimic.human_video_data_pattern = "retarget_poses_g1.h5"
    env_cfg.deepmimic.human_video_terrain_pattern = "background_mesh.obj"
    env_cfg.deepmimic.data_root = str(data_root)
    env_cfg.deepmimic.alt_data_root = ""
    env_cfg.deepmimic.randomize_start_offset = False
    env_cfg.deepmimic.viz_replay = False
    env_cfg.deepmimic.viz_replay_sync_robot = False
    env_cfg.asset.disable_gravity = True
    env_cfg.asset.use_alt_files = False
    env_cfg.control.stiffness = {k: 0.0 for k in env_cfg.control.stiffness}
    env_cfg.control.damping = {k: 0.0 for k in env_cfg.control.damping}


def decode_object_array(array: np.ndarray) -> list[str]:
    values = []
    for item in array.tolist():
        if isinstance(item, bytes):
            values.append(item.decode("utf-8"))
        else:
            values.append(str(item))
    return values


def write_obj(obj_path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with obj_path.open("w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            handle.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def extract_npz_clip(npz_path: Path, extracted_root: Path) -> str:
    clip_name = npz_path.stem
    clip_dir = extracted_root / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    h5_path = clip_dir / "retarget_poses_g1.h5"
    obj_path = clip_dir / "background_mesh.obj"

    with h5py.File(h5_path, "w") as handle:
        handle.attrs["fps"] = float(np.asarray(data["fps"]).reshape(-1)[0])
        handle.attrs["joint_names"] = np.asarray(decode_object_array(data["joint_names"]), dtype=object)
        handle.attrs["link_names"] = np.asarray(decode_object_array(data["link_names"]), dtype=object)

        handle.create_dataset("root_pos", data=np.asarray(data["root_pos"], dtype=np.float32))
        handle.create_dataset("root_quat", data=np.asarray(data["root_quat"], dtype=np.float32))
        handle.create_dataset("joints", data=np.asarray(data["joints"], dtype=np.float32))
        handle.create_dataset("link_pos", data=np.asarray(data["link_pos"], dtype=np.float32))
        handle.create_dataset("link_quat", data=np.asarray(data["link_quat"], dtype=np.float32))

        contacts = handle.create_group("contacts")
        for contact_key in ("contacts_left_foot", "contacts_right_foot"):
            if contact_key in data:
                contacts.create_dataset(
                    contact_key.replace("contacts_", ""),
                    data=np.asarray(data[contact_key], dtype=np.float32),
                )

    write_obj(
        obj_path,
        vertices=np.asarray(data["mesh_vertices"], dtype=np.float32),
        faces=np.asarray(data["mesh_faces"], dtype=np.int32),
    )
    return clip_name


def create_camera(env, width: int, height: int):
    from isaacgym import gymapi

    camera_props = gymapi.CameraProperties()
    camera_props.width = width
    camera_props.height = height
    camera_props.enable_tensors = False
    camera_props.use_collision_geometry = False
    camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    return camera_handle


def set_actor_color(env, color) -> None:
    from isaacgym import gymapi

    vec = gymapi.Vec3(*color)
    for body_idx in range(env.num_bodies):
        env.gym.set_rigid_body_color(
            env.envs[0],
            env.actor_handles[0],
            body_idx,
            gymapi.MESH_VISUAL,
            vec,
        )


def apply_replay_state(env, state) -> None:
    import torch
    from isaacgym import gymtorch

    env_ids = torch.tensor([0], device=env.device, dtype=torch.int32)
    world_root = env.env_frame_to_world_frame(state.root_pos[:1], env_ids).squeeze(0)

    env.root_states[0, 0:3] = world_root
    env.root_states[0, 3:7] = state.root_quat[0]
    env.root_states[0, 7:13] = 0.0
    env.dof_pos[0] = state.dofs[0]
    env.dof_vel[0] = 0.0

    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids),
        len(env_ids),
    )
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids),
        len(env_ids),
    )


def flush_pose_to_renderer(env) -> None:
    # Isaac Gym only updates rigid-body poses from the new root/dof state after
    # a sim step. Gravity is disabled in this render env so this step performs
    # articulation FK without the replay pose collapsing before rendering.
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)


def render_frame(env, camera_handle: int, width: int, height: int, camera_offset, lookat_height: float):
    from isaacgym import gymapi

    root = env.root_states[0, 0:3].detach().cpu().numpy()
    camera_pos = root + camera_offset
    lookat = root + np.array([0.0, 0.0, lookat_height], dtype=np.float32)
    env.gym.set_camera_location(
        camera_handle,
        env.envs[0],
        gymapi.Vec3(*camera_pos.tolist()),
        gymapi.Vec3(*lookat.tolist()),
    )
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    image = env.gym.get_camera_image(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
    frame = np.asarray(image, dtype=np.uint8).reshape(height, width, 4)[..., :3]
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def make_writer(path: Path, width: int, height: int, fps: float):
    return CompatibleVideoWriter(path, fps, (width, height), input_color="bgr")


def destroy_env(env) -> None:
    import torch

    if getattr(env, "viewer", None):
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)
    del env
    gc.collect()
    torch.cuda.empty_cache()


def render_clip(clip_name: str, data_root: Path, output_path: Path, width: int, height: int, camera_offset, lookat_height: float) -> None:
    import isaacgym  # noqa: F401
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils.task_registry import task_registry

    args = build_args(num_envs=1)
    env_cfg, _ = task_registry.get_cfgs(name="g1_deepmimic")
    configure_env(env_cfg, clip_name, data_root=data_root)
    env, _ = task_registry.make_env(name="g1_deepmimic", args=args, env_cfg=env_cfg)

    try:
        camera_handle = create_camera(env, width, height)
        set_actor_color(env, color=(0.65, 0.92, 1.0))

        fps = round(1.0 / env.dt)
        num_frames = int(env.ep_lengths[0].item())
        writer = make_writer(output_path, width, height, fps)
        try:
            state = env.replay_data_loader.get_current_data()
            apply_replay_state(env, state)
            flush_pose_to_renderer(env)
            for frame_idx in range(num_frames):
                frame = render_frame(env, camera_handle, width, height, camera_offset, lookat_height)
                writer.write(frame)
                if frame_idx + 1 < num_frames:
                    env.replay_data_loader.increment_indices()
                    state = env.replay_data_loader.get_current_data()
                    apply_replay_state(env, state)
                    flush_pose_to_renderer(env)
        finally:
            writer.release()
    finally:
        destroy_env(env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render VideoMimic NPZ captures to mp4 with offscreen Isaac Gym cameras.")
    parser.add_argument("--data-root", type=Path, default=Path("simulation/data/videomimic_captures_npz"))
    parser.add_argument("--extracted-root", type=Path, default=Path("/tmp/videomimic_captures_npz_unpacked"))
    parser.add_argument("--output-root", type=Path, default=Path("simulation/data/videomimic_captures_mp4"))
    parser.add_argument(
        "--input-format",
        choices=("auto", "npz", "capture_dirs"),
        default="auto",
        help="Whether data-root contains .npz files or capture directories with retarget_poses_g1.h5/background_mesh.obj.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--clip", action="append", default=[], help="Specific clip folder names to render. Can be passed multiple times.")
    parser.add_argument("--limit", type=int, default=0, help="Render only the first N clips after filtering. 0 means all.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in the sorted clip list.")
    parser.add_argument("--end-index", type=int, default=-1, help="Exclusive end index in the sorted clip list. -1 means until the end.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    extracted_root = args.extracted_root.resolve()
    extracted_root.mkdir(parents=True, exist_ok=True)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.input_format == "auto":
        has_npz = any(path.is_file() and path.suffix == ".npz" for path in data_root.iterdir())
        input_format = "npz" if has_npz else "capture_dirs"
    else:
        input_format = args.input_format

    if args.clip:
        clip_names = args.clip
    elif input_format == "npz":
        clip_names = sorted(path.name for path in data_root.iterdir() if path.is_file() and path.suffix == ".npz")
    else:
        clip_names = sorted(path.name for path in data_root.iterdir() if path.is_dir())

    if args.limit > 0:
        clip_names = clip_names[:args.limit]
    if args.start_index > 0 or args.end_index >= 0:
        end_index = None if args.end_index < 0 else args.end_index
        clip_names = clip_names[args.start_index:end_index]

    camera_offset = np.array([2.8, -2.4, 1.6], dtype=np.float32)
    lookat_height = 0.9

    total = len(clip_names)
    for idx, clip_entry in enumerate(clip_names, start=1):
        if input_format == "npz":
            npz_path = data_root / clip_entry
            if not npz_path.exists():
                print(f"[{idx}/{total}] skipping {clip_entry}: missing npz file")
                continue
            clip_name = npz_path.stem
            render_data_root = extracted_root
            extracted_clip_name = None
        else:
            clip_dir = data_root / clip_entry
            if not clip_dir.is_dir():
                print(f"[{idx}/{total}] skipping {clip_entry}: missing capture directory")
                continue
            clip_name = clip_dir.name
            render_data_root = data_root
            extracted_clip_name = clip_name

        output_path = output_root / f"{clip_name}.mp4"
        if output_path.exists() and not args.overwrite:
            print(f"[{idx}/{total}] skipping {clip_name}: output exists")
            continue

        print(f"[{idx}/{total}] rendering {clip_name} -> {output_path.name}")
        if input_format == "npz":
            extracted_clip_name = extract_npz_clip(npz_path, extracted_root)
        render_clip(
            clip_name=extracted_clip_name,
            data_root=render_data_root,
            output_path=output_path,
            width=args.width,
            height=args.height,
            camera_offset=camera_offset,
            lookat_height=lookat_height,
        )


if __name__ == "__main__":
    main()
