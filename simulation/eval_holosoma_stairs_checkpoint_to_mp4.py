#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions_videomimic_eval")
SIM_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SIM_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_preview_compat import CompatibleVideoWriter

from terrain_saliency_utils import (
    TERRAIN_SALIENCY_MODES,
    append_panel_below_frame,
    compute_terrain_saliency,
    estimate_rollout_kinematics,
    infer_run_checkpoint_from_rollout_dir,
    make_terrain_saliency_panel,
    refresh_render_observations,
    rollout_state_at_frame,
    set_reference_frame,
)


def build_args(seed: int, num_envs: int = 1) -> SimpleNamespace:
    from isaacgym import gymapi

    return SimpleNamespace(
        task=None,
        physics_engine=gymapi.SIM_PHYSX,
        use_gpu=True,
        subscenes=0,
        use_gpu_pipeline=True,
        num_threads=0,
        sim_device="cuda:0",
        compute_device_id=0,
        sim_device_id=0,
        rl_device="cuda:0",
        graphics_device_id=0,
        headless=True,
        horovod=False,
        multi_gpu=False,
        no_use_wandb=True,
        wandb_note=None,
        max_iterations=None,
        resume=True,
        experiment_name=None,
        run_name=None,
        load_run=None,
        checkpoint=None,
        seed=seed,
        num_envs=num_envs,
        slices=0,
        pipeline="gpu",
        flex=False,
        physx=False,
        nographics=False,
    )


def configure_eval_env(env_cfg, motion_source: str) -> None:
    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.env.export_trajectory = False
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
    env_cfg.deepmimic.human_motion_source = motion_source
    env_cfg.deepmimic.upsample_data = True
    env_cfg.deepmimic.link_pos_error_threshold = 0.5
    env_cfg.deepmimic.respawn_z_offset = 0.1
    env_cfg.deepmimic.randomize_terrain_offset = False
    env_cfg.deepmimic.randomize_start_offset = False
    env_cfg.deepmimic.truncate_rollout_length = 500
    env_cfg.deepmimic.amass_terrain_difficulty = 1


def configure_render_env(
    env_cfg,
    clip_dir_name: str,
    data_root: Path,
    data_pattern: str,
    terrain_pattern: str = "background_mesh.obj",
) -> None:
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
    env_cfg.deepmimic.use_amass = False
    env_cfg.deepmimic.use_human_videos = True
    env_cfg.deepmimic.human_motion_source = clip_dir_name
    env_cfg.deepmimic.human_video_data_pattern = data_pattern
    env_cfg.deepmimic.human_video_terrain_pattern = terrain_pattern
    env_cfg.deepmimic.data_root = str(data_root)
    env_cfg.deepmimic.alt_data_root = ""
    env_cfg.deepmimic.randomize_start_offset = False
    env_cfg.deepmimic.viz_replay = False
    env_cfg.deepmimic.viz_replay_sync_robot = False
    env_cfg.asset.use_alt_files = False
    env_cfg.control.stiffness = {k: 0.0 for k in env_cfg.control.stiffness}
    env_cfg.control.damping = {k: 0.0 for k in env_cfg.control.damping}


def ensure_symlink(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)


def set_episode_start(env, episode_idx: int, start_offset: int) -> None:
    import torch

    env.selected_episode_idx = episode_idx
    env.selected_start_offset = start_offset
    env.reset_idx(torch.tensor([0], device=env.device), already_reset_replay_data=True)
    total_length = int(env.replay_data_loader.sequence_lengths[episode_idx].item())
    remaining = max(total_length - start_offset, 1)
    if env.cfg.deepmimic.truncate_rollout_length > 0:
        remaining = min(remaining, int(env.cfg.deepmimic.truncate_rollout_length))
    env.max_episode_length = torch.tensor([remaining], device=env.device, dtype=torch.long)
    env.episode_length_buf[0] = 0


def run_policy_episode(env, policy, start_offset: int, rollout_dir: Path) -> dict:
    import torch

    rollout_dir.mkdir(parents=True, exist_ok=True)

    clip_index = int(env.replay_data_loader.episode_indices[0].item())
    set_episode_start(env, clip_index, start_offset)
    obs = env.get_observations()

    source_motion_path = Path(env.replay_data_loader.get_pkl_paths()[clip_index]).resolve()
    source_mesh_path = Path(env.terrain_paths[clip_index]).resolve()
    trajectory_name = source_motion_path.stem

    data = {
        "joint_names": list(env.dof_names),
        "joints": [],
        "joint_targets": [],
        "root_quat": [],
        "root_pos": [],
        "link_names": list(env.body_names),
        "link_pos": [],
        "link_quat": [],
        "actions": [],
        "contacts": {"left_foot": [], "right_foot": []},
        "stiffness": {name: float(env.p_gains[i].cpu().numpy()) for i, name in enumerate(env.dof_names)},
        "damping": {name: float(env.d_gains[i].cpu().numpy()) for i, name in enumerate(env.dof_names)},
        "trajectory_name": trajectory_name,
        "fps": float(round(1.0 / env.dt)),
    }

    done = False
    steps = 0
    while not done:
        try:
            actions = policy({k: v.detach() for k, v in obs.items()}, monitor_activations=False)
        except TypeError:
            actions = policy({k: v.detach() for k, v in obs.items()})

        joint_targets = env._compute_dof_pos_targets(actions[0:1])
        if joint_targets is not None:
            data["joint_targets"].append(joint_targets[0].detach().cpu().numpy())

        data["joints"].append(env.dof_pos[0].detach().cpu().numpy())
        data["root_quat"].append(env.base_quat[0].detach().cpu().numpy())
        data["root_pos"].append(env.env_root_pos[0].detach().cpu().numpy())
        data["actions"].append(actions[0].detach().cpu().numpy())
        data["link_pos"].append(env.env_rigid_body_pos[0].detach().cpu().numpy())
        data["link_quat"].append(env.rigid_body_quat[0].detach().cpu().numpy())

        foot_names = [env.body_names[idx] for idx in env.feet_indices]
        left_idx = foot_names.index("left_ankle_roll_link")
        right_idx = foot_names.index("right_ankle_roll_link")
        left_contact = bool(torch.norm(env.contact_forces[0, env.feet_indices[left_idx], :]) > 1.0)
        right_contact = bool(torch.norm(env.contact_forces[0, env.feet_indices[right_idx], :]) > 1.0)
        data["contacts"]["left_foot"].append(left_contact)
        data["contacts"]["right_foot"].append(right_contact)

        obs, _, dones, _ = env.step(actions.detach())
        done = bool(dones[0].item())
        steps += 1

    export = {
        "joint_names": data["joint_names"],
        "joints": np.asarray(data["joints"], dtype=np.float32),
        "joint_targets": np.asarray(data["joint_targets"], dtype=np.float32) if data["joint_targets"] else None,
        "root_quat": np.asarray(data["root_quat"], dtype=np.float32),
        "root_pos": np.asarray(data["root_pos"], dtype=np.float32),
        "link_names": data["link_names"],
        "link_pos": np.asarray(data["link_pos"], dtype=np.float32),
        "link_quat": np.asarray(data["link_quat"], dtype=np.float32),
        "actions": np.asarray(data["actions"], dtype=np.float32),
        "contacts": {
            name: np.asarray(values, dtype=np.bool_) for name, values in data["contacts"].items()
        },
        "stiffness": data["stiffness"],
        "damping": data["damping"],
        "trajectory_name": trajectory_name,
        "fps": data["fps"],
    }

    rollout_path = rollout_dir / "policy_rollout.pkl"
    with rollout_path.open("wb") as handle:
        pickle.dump(export, handle)

    ensure_symlink(source_mesh_path, rollout_dir / "background_mesh.obj")

    success = bool(env.time_out_buf[0].item())
    metrics = {
        "trajectory_name": trajectory_name,
        "source_motion_path": str(source_motion_path),
        "source_mesh_path": str(source_mesh_path),
        "start_offset": int(start_offset),
        "steps": int(steps),
        "success_by_timeout": success,
    }
    with (rollout_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return {
        "rollout_path": rollout_path,
        "metrics": metrics,
    }


def create_camera(env, width: int, height: int):
    from isaacgym import gymapi

    camera_props = gymapi.CameraProperties()
    camera_props.width = width
    camera_props.height = height
    camera_props.enable_tensors = False
    camera_props.use_collision_geometry = False
    return env.gym.create_camera_sensor(env.envs[0], camera_props)


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
    root_pos = torch.as_tensor(state.root_pos[:1], device=env.device, dtype=env.root_states.dtype)
    root_quat = torch.as_tensor(state.root_quat[0], device=env.device, dtype=env.root_states.dtype)
    dofs = torch.as_tensor(state.dofs[0], device=env.device, dtype=env.dof_pos.dtype)
    world_root = env.env_frame_to_world_frame(root_pos, env_ids).squeeze(0)
    env.root_states[0, 0:3] = world_root
    env.root_states[0, 3:7] = root_quat
    if hasattr(state, "root_vel"):
        env.root_states[0, 7:10] = torch.as_tensor(state.root_vel[0], device=env.device, dtype=env.root_states.dtype)
    else:
        env.root_states[0, 7:10] = 0.0
    if hasattr(state, "root_ang_vel"):
        env.root_states[0, 10:13] = torch.as_tensor(state.root_ang_vel[0], device=env.device, dtype=env.root_states.dtype)
    else:
        env.root_states[0, 10:13] = 0.0
    env.dof_pos[0] = dofs
    if hasattr(state, "dof_vel"):
        env.dof_vel[0] = torch.as_tensor(state.dof_vel[0], device=env.device, dtype=env.dof_pos.dtype)
    else:
        env.dof_vel[0] = 0.0
    if hasattr(state, "prev_action") and state.prev_action.shape[-1] == env.actions.shape[-1]:
        env.actions[0] = torch.as_tensor(state.prev_action[0], device=env.device, dtype=env.actions.dtype)
    else:
        env.actions[0] = 0.0

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


def load_obj_bounds(mesh_path: Path):
    vertices = []
    with mesh_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        return None
    verts = np.asarray(vertices, dtype=np.float32)
    return verts.min(axis=0), verts.max(axis=0)


def compute_fixed_camera_pose(rollout_dir: Path):
    with (rollout_dir / "policy_rollout.pkl").open("rb") as handle:
        rollout = pickle.load(handle)

    root_pos = np.asarray(rollout["root_pos"], dtype=np.float32)
    mins = root_pos.min(axis=0)
    maxs = root_pos.max(axis=0)

    mesh_path = rollout_dir / "background_mesh.obj"
    if mesh_path.exists():
        mesh_bounds = load_obj_bounds(mesh_path)
        if mesh_bounds is not None:
            mesh_mins, mesh_maxs = mesh_bounds
            mins = np.minimum(mins, mesh_mins)
            maxs = np.maximum(maxs, mesh_maxs)

    center = 0.5 * (mins + maxs)
    extent = np.maximum(maxs - mins, np.array([1.0, 1.0, 0.5], dtype=np.float32))
    planar_span = float(max(extent[0], extent[1]))
    vertical_span = float(extent[2])

    camera_pos = center + np.array(
        [
            0.85 * planar_span,
            -1.75 * planar_span,
            0.9 * planar_span + 0.6 * vertical_span + 0.8,
        ],
        dtype=np.float32,
    )
    lookat = center + np.array([0.0, 0.0, 0.35 * vertical_span + 0.4], dtype=np.float32)
    return camera_pos, lookat


def render_frame_fixed(env, camera_handle: int, width: int, height: int, camera_pos, lookat):
    from isaacgym import gymapi

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


def render_rollout_mp4(
    rollout_dir: Path,
    output_path: Path,
    width: int,
    height: int,
    camera_mode: str,
    task_name: str,
    load_run: str | None,
    checkpoint: int | None,
    show_terrain_saliency: bool,
    terrain_saliency_mode: str,
) -> None:
    import isaacgym  # noqa: F401
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils.task_registry import task_registry

    metrics = json.loads((rollout_dir / "metrics.json").read_text(encoding="utf-8"))
    source_motion_path = Path(metrics["source_motion_path"]).resolve()
    source_clip_dir = source_motion_path.parent
    source_mesh_path = Path(metrics.get("source_mesh_path", source_clip_dir / "background_mesh.obj")).resolve()
    with (rollout_dir / "policy_rollout.pkl").open("rb") as handle:
        policy_rollout = pickle.load(handle)
    rollout_dt = 1.0 / float(policy_rollout.get("fps", 50.0))
    rollout_kinematics = estimate_rollout_kinematics(policy_rollout, rollout_dt)
    num_frames = int(len(policy_rollout["root_pos"]))

    if show_terrain_saliency and (load_run is None or checkpoint is None):
        load_run, checkpoint = infer_run_checkpoint_from_rollout_dir(rollout_dir)

    args = build_args(seed=1, num_envs=1)
    args.task = task_name
    args.load_run = load_run
    args.checkpoint = checkpoint
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    configure_render_env(
        env_cfg,
        clip_dir_name=source_clip_dir.name,
        data_root=source_clip_dir.parent,
        data_pattern=source_motion_path.name,
        terrain_pattern=source_mesh_path.name,
    )
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    camera_offset = np.array([2.8, -2.4, 1.6], dtype=np.float32)
    lookat_height = 0.9
    fixed_camera_pos, fixed_lookat = compute_fixed_camera_pose(rollout_dir)
    actor_critic = None

    try:
        if show_terrain_saliency:
            train_cfg.runner.resume = True
            train_cfg.runner.load_run = load_run
            train_cfg.runner.checkpoint = checkpoint
            runner, _ = task_registry.make_alg_runner(
                env=env,
                name=task_name,
                args=args,
                train_cfg=train_cfg,
            )
            actor_critic = runner.alg.actor_critic
            actor_critic.eval()
            if env.history_handler is not None:
                import torch

                env.history_handler.reset(torch.tensor([0], device=env.device))

        camera_handle = create_camera(env, width, height)
        set_actor_color(env, color=(0.75, 0.95, 1.0))
        fps = float(policy_rollout.get("fps", round(1.0 / env.dt)))
        writer = None
        try:
            for frame_idx in range(num_frames):
                reference_frame_idx = int(metrics["start_offset"]) + frame_idx
                set_reference_frame(env, reference_frame_idx, episode_idx=0)
                state = rollout_state_at_frame(policy_rollout, rollout_kinematics, frame_idx)
                apply_replay_state(env, state)
                flush_pose_to_renderer(env)
                obs = refresh_render_observations(env, frame_idx)
                if camera_mode == "fixed":
                    frame = render_frame_fixed(env, camera_handle, width, height, fixed_camera_pos, fixed_lookat)
                else:
                    frame = render_frame(env, camera_handle, width, height, camera_offset, lookat_height)
                if show_terrain_saliency and actor_critic is not None:
                    terrain_grid, saliency_grid = compute_terrain_saliency(
                        actor_critic,
                        {k: v.detach() for k, v in obs.items()},
                        mode=terrain_saliency_mode,
                    )
                    panel = make_terrain_saliency_panel(
                        terrain_grid,
                        saliency_grid,
                        saliency_mode=terrain_saliency_mode,
                    )
                    frame = append_panel_below_frame(frame, panel, saliency_mode=terrain_saliency_mode)
                if writer is None:
                    writer = make_writer(output_path, frame.shape[1], frame.shape[0], fps)
                writer.write(frame)
        finally:
            if writer is not None:
                writer.release()
    finally:
        destroy_env(env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the latest holosoma-stairs checkpoint and render rollout mp4 files.")
    parser.add_argument("--mode", choices=("eval", "render"), default="eval")
    parser.add_argument("--task", default="g1_deepmimic_proj_heightfield")
    parser.add_argument("--load-run", default="20260316_034146_holosoma_stairs_heightfield_single_gpu_30k_20260316")
    parser.add_argument("--checkpoint", type=int, default=314000)
    parser.add_argument("--motion-source", default="resources/data_config/holosoma_stairs_motion.yaml")
    parser.add_argument("--num-evals", type=int, default=5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("simulation/data/holosoma_stairs_policy_eval"),
    )
    parser.add_argument("--rollout-dir", type=Path)
    parser.add_argument("--mp4-path", type=Path)
    parser.add_argument("--camera-mode", choices=("fixed", "follow"), default="fixed")
    parser.add_argument("--show-terrain-saliency", action="store_true", default=False)
    parser.add_argument("--terrain-saliency-mode", choices=TERRAIN_SALIENCY_MODES, default="global_mean")
    return parser.parse_args()


def run_eval_phase(args: argparse.Namespace) -> None:
    import isaacgym  # noqa: F401
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils.task_registry import task_registry

    output_root = args.output_root.resolve() / f"{args.load_run}_ckpt{args.checkpoint}"
    rollouts_root = output_root / "rollouts"
    rollouts_root.mkdir(parents=True, exist_ok=True)

    sim_args = build_args(seed=1, num_envs=1)
    sim_args.task = args.task
    sim_args.load_run = args.load_run
    sim_args.checkpoint = args.checkpoint

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    configure_eval_env(env_cfg, motion_source=args.motion_source)
    env, _ = task_registry.make_env(name=args.task, args=sim_args, env_cfg=env_cfg)

    try:
        train_cfg.runner.resume = True
        train_cfg.runner.load_run = args.load_run
        train_cfg.runner.checkpoint = args.checkpoint
        runner, _ = task_registry.make_alg_runner(
            env=env,
            name=args.task,
            args=sim_args,
            train_cfg=train_cfg,
        )
        policy = runner.get_inference_policy(device=env.device)

        total_length = int(env.replay_data_loader.sequence_lengths[0].item())
        max_offset = max(total_length - 30, 0)
        start_offsets = np.linspace(0, max_offset, num=args.num_evals, dtype=int)

        summary = []
        for eval_idx, start_offset in enumerate(start_offsets, start=1):
            rollout_dir = rollouts_root / f"eval_{eval_idx:02d}"
            result = run_policy_episode(env, policy, int(start_offset), rollout_dir)
            entry = {
                "eval_index": eval_idx,
                **result["metrics"],
            }
            summary.append(entry)
            print(json.dumps(entry, ensure_ascii=True))

        with (output_root / "summary_eval.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    finally:
        destroy_env(env)


def run_render_phase(args: argparse.Namespace) -> None:
    if args.rollout_dir is None:
        raise ValueError("--rollout-dir is required in render mode")
    rollout_dir = args.rollout_dir.resolve()
    if args.mp4_path is None:
        output_path = rollout_dir.with_suffix(".mp4")
    else:
        output_path = args.mp4_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_rollout_mp4(
        rollout_dir,
        output_path,
        width=args.width,
        height=args.height,
        camera_mode=args.camera_mode,
        task_name=args.task,
        load_run=args.load_run,
        checkpoint=args.checkpoint,
        show_terrain_saliency=args.show_terrain_saliency,
        terrain_saliency_mode=args.terrain_saliency_mode,
    )
    print(json.dumps({"rollout_dir": str(rollout_dir), "mp4_path": str(output_path)}, ensure_ascii=True))


def main() -> None:
    args = parse_args()
    if args.mode == "eval":
        run_eval_phase(args)
    else:
        run_render_phase(args)


if __name__ == "__main__":
    main()
