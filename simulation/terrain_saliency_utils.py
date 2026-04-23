#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

TERRAIN_SALIENCY_MODES = ("global_mean", "local_mean", "gradient")


def infer_run_checkpoint_from_rollout_dir(rollout_dir: Path) -> tuple[str, int]:
    if rollout_dir.parent.name == "rollouts":
        candidate = rollout_dir.parent.parent.name
    else:
        candidate = rollout_dir.parent.name
    match = re.match(r"^(?P<run>.+)_ckpt(?P<ckpt>\d+)$", candidate)
    if match is None:
        raise ValueError(f"Could not infer load_run/checkpoint from rollout dir: {rollout_dir}")
    return match.group("run"), int(match.group("ckpt"))


def _quat_conjugate_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.asarray([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float32)


def _quat_multiply_xyzw(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    return np.asarray(
        [
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        ],
        dtype=np.float32,
    )


def _quat_to_ang_vel_xyzw(q_prev: np.ndarray, q_next: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0.0:
        return np.zeros(3, dtype=np.float32)
    dq = _quat_multiply_xyzw(q_next, _quat_conjugate_xyzw(q_prev))
    if dq[3] < 0.0:
        dq = -dq
    axis_norm = float(np.linalg.norm(dq[:3]))
    if axis_norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(axis_norm, float(np.clip(dq[3], -1.0, 1.0)))
    axis = dq[:3] / axis_norm
    return (axis * (angle / dt)).astype(np.float32)


def estimate_rollout_kinematics(rollout: dict, dt: float) -> dict:
    root_pos = np.asarray(rollout["root_pos"], dtype=np.float32)
    root_quat = np.asarray(rollout["root_quat"], dtype=np.float32)
    joints = np.asarray(rollout["joints"], dtype=np.float32)
    actions = np.asarray(rollout.get("actions", np.zeros((len(joints), 0), dtype=np.float32)), dtype=np.float32)

    root_vel = np.gradient(root_pos, dt, axis=0).astype(np.float32)
    dof_vel = np.gradient(joints, dt, axis=0).astype(np.float32)
    root_ang_vel = np.zeros_like(root_pos, dtype=np.float32)

    num_frames = root_quat.shape[0]
    for idx in range(num_frames):
        prev_idx = max(idx - 1, 0)
        next_idx = min(idx + 1, num_frames - 1)
        local_dt = max((next_idx - prev_idx) * dt, dt)
        root_ang_vel[idx] = _quat_to_ang_vel_xyzw(root_quat[prev_idx], root_quat[next_idx], local_dt)

    prev_actions = np.zeros_like(actions, dtype=np.float32)
    if actions.shape[0] > 1:
        prev_actions[1:] = actions[:-1]

    return {
        "root_vel": root_vel,
        "root_ang_vel": root_ang_vel,
        "dof_vel": dof_vel,
        "prev_actions": prev_actions,
    }


def rollout_state_at_frame(rollout: dict, kinematics: dict, frame_idx: int) -> SimpleNamespace:
    return SimpleNamespace(
        root_pos=np.asarray(rollout["root_pos"][frame_idx : frame_idx + 1], dtype=np.float32),
        root_quat=np.asarray(rollout["root_quat"][frame_idx : frame_idx + 1], dtype=np.float32),
        dofs=np.asarray(rollout["joints"][frame_idx : frame_idx + 1], dtype=np.float32),
        root_vel=np.asarray(kinematics["root_vel"][frame_idx : frame_idx + 1], dtype=np.float32),
        root_ang_vel=np.asarray(kinematics["root_ang_vel"][frame_idx : frame_idx + 1], dtype=np.float32),
        dof_vel=np.asarray(kinematics["dof_vel"][frame_idx : frame_idx + 1], dtype=np.float32),
        prev_action=np.asarray(kinematics["prev_actions"][frame_idx : frame_idx + 1], dtype=np.float32),
    )


def set_reference_frame(env, frame_idx: int, episode_idx: int = 0) -> None:
    env.replay_data_loader.set_env_data(0, episode_idx, int(frame_idx))
    env.update_replay_data()


def refresh_render_observations(env, frame_idx: int) -> dict:
    from legged_gym.tensor_utils.torch_jit_utils import quat_rotate_inverse
    from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor

    env.episode_length_buf[0] = int(frame_idx)

    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_net_contact_force_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)

    env.base_pos[:] = env.root_states[:, 0:3]
    env.base_quat[:] = env.root_states[:, 3:7]
    env.rpy[:] = get_euler_xyz_in_tensor(env.base_quat[:])
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])
    env.projected_gravity[:] = quat_rotate_inverse(env.base_quat, env.gravity_vec)

    for sensor in env.sensors.values():
        sensor.update_buffers(episode_step=env.episode_length_buf, env_ids=...)

    env.compute_observations()
    return env.obs_dict


def resolve_terrain_obs_key(actor_critic, obs_dict: dict) -> str | None:
    candidate_keys = []
    candidate_keys.extend(actor_critic.actor_input_net.heads.keys())
    candidate_keys.extend(actor_critic.actor_input_net.extra_proj_heads.keys())
    for key in candidate_keys:
        if key in ("terrain_height", "terrain_height_noisy") and key in obs_dict:
            return key
    for key in ("terrain_height", "terrain_height_noisy"):
        if key in obs_dict:
            return key
    return None


def _terrain_saliency_mode_title(mode: str) -> str:
    if mode == "global_mean":
        return "Terrain sensitivity (global mean)"
    if mode == "local_mean":
        return "Terrain sensitivity (local mean)"
    if mode == "gradient":
        return "Terrain sensitivity (gradient)"
    raise ValueError(f"Unsupported terrain saliency mode: {mode}")


def _terrain_saliency_mode_footer(mode: str, saliency_max: float) -> str:
    if mode == "global_mean":
        return f"global mean mask -> |delta action|, max {saliency_max:.3f}"
    if mode == "local_mean":
        return f"local mean mask -> |delta action|, max {saliency_max:.3f}"
    if mode == "gradient":
        return f"jacobian L2 per cell, max {saliency_max:.3f}"
    raise ValueError(f"Unsupported terrain saliency mode: {mode}")


def _compute_local_mean_fill_values(terrain_single):
    import torch
    import torch.nn.functional as F

    kernel = torch.ones((1, 1, 3, 3), device=terrain_single.device, dtype=terrain_single.dtype)
    kernel[0, 0, 1, 1] = 0.0
    terrain_4d = terrain_single.unsqueeze(1)
    sums = F.conv2d(terrain_4d, kernel, padding=1)
    counts = F.conv2d(torch.ones_like(terrain_4d), kernel, padding=1)
    global_mean = terrain_single.mean()
    fill_values = torch.where(counts > 0, sums / counts.clamp_min(1.0), global_mean)
    return fill_values.squeeze(0).squeeze(0)


def _compute_occlusion_saliency(actor_critic, obs_dict: dict, obs_key: str, mode: str):
    import torch

    terrain_obs = obs_dict[obs_key]
    terrain_single = terrain_obs[:1]
    height, width = terrain_single.shape[-2:]
    num_cells = height * width

    with torch.inference_mode():
        base_obs = {key: value[:1] for key, value in obs_dict.items()}
        base_action = actor_critic.act_inference(base_obs)

        batched_obs = {}
        for key, value in obs_dict.items():
            repeats = [num_cells] + [1] * (value.dim() - 1)
            batched_obs[key] = value[:1].repeat(*repeats)

        terrain_batch = batched_obs[obs_key].clone().view(num_cells, -1)
        diag_idx = torch.arange(num_cells, device=terrain_batch.device)
        if mode == "global_mean":
            fill_values = terrain_single.mean().expand(num_cells)
        elif mode == "local_mean":
            fill_values = _compute_local_mean_fill_values(terrain_single).reshape(-1)
        else:
            raise ValueError(f"Unsupported occlusion saliency mode: {mode}")
        terrain_batch[diag_idx, diag_idx] = fill_values
        batched_obs[obs_key] = terrain_batch.view(num_cells, height, width)

        masked_actions = actor_critic.act_inference(batched_obs)
        delta = masked_actions - base_action.repeat(num_cells, 1)
        saliency = torch.linalg.norm(delta, dim=1).view(height, width)

    return (
        terrain_single[0].detach().cpu().numpy(),
        saliency.detach().cpu().numpy(),
        )


def _compute_gradient_saliency(actor_critic, obs_dict: dict, obs_key: str):
    import torch

    base_obs = {key: value[:1].detach() for key, value in obs_dict.items()}
    terrain_input = base_obs[obs_key].clone().detach().requires_grad_(True)
    base_obs[obs_key] = terrain_input

    actor_critic.zero_grad(set_to_none=True)
    action = actor_critic.act_inference(base_obs)
    saliency_sq = torch.zeros_like(terrain_input[0])
    num_actions = int(action.shape[-1])
    for action_idx in range(num_actions):
        retain = action_idx + 1 < num_actions
        grad = torch.autograd.grad(
            action[0, action_idx],
            terrain_input,
            retain_graph=retain,
            create_graph=False,
            allow_unused=False,
        )[0]
        saliency_sq = saliency_sq + grad[0] ** 2
    actor_critic.zero_grad(set_to_none=True)
    saliency = torch.sqrt(saliency_sq)
    return (
        terrain_input[0].detach().cpu().numpy(),
        saliency.detach().cpu().numpy(),
    )


def compute_terrain_saliency(
    actor_critic,
    obs_dict: dict,
    obs_key: str | None = None,
    mode: str = "global_mean",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if mode not in TERRAIN_SALIENCY_MODES:
        raise ValueError(f"Unsupported terrain saliency mode: {mode}")

    if obs_key is None:
        obs_key = resolve_terrain_obs_key(actor_critic, obs_dict)
    if obs_key is None:
        return None, None

    terrain_obs = obs_dict[obs_key]
    if terrain_obs.ndim != 3 or terrain_obs.shape[0] < 1:
        return None, None

    if mode in ("global_mean", "local_mean"):
        return _compute_occlusion_saliency(actor_critic, obs_dict, obs_key, mode)
    if mode == "gradient":
        return _compute_gradient_saliency(actor_critic, obs_dict, obs_key)
    raise ValueError(f"Unsupported terrain saliency mode: {mode}")


def _normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    lo = float(np.percentile(values[finite], 5.0))
    hi = float(np.percentile(values[finite], 95.0))
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    clipped = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return np.round(clipped * 255.0).astype(np.uint8)


def _orient_grid_for_display(values: np.ndarray) -> np.ndarray:
    """Rotate the local torso-centric height grid to a top-down map view.

    Original sensor convention in the rendered panel was:
    - right  = robot front
    - left   = robot back
    - top    = robot right
    - bottom = robot left

    For presentation we prefer the more intuitive map convention:
    - top    = robot front
    - bottom = robot back
    - left   = robot left
    - right  = robot right
    """
    return np.fliplr(np.rot90(values, k=1))


def make_terrain_saliency_panel(
    terrain_grid: np.ndarray | None,
    saliency_grid: np.ndarray | None,
    panel_size: int = 280,
    saliency_mode: str = "global_mean",
) -> np.ndarray:
    if terrain_grid is None or saliency_grid is None:
        canvas = np.full((panel_size + 70, panel_size + 40, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, "No terrain saliency", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)
        return canvas

    terrain_grid = _orient_grid_for_display(np.asarray(terrain_grid))
    saliency_grid = _orient_grid_for_display(np.asarray(saliency_grid))

    terrain_vis = _normalize_to_uint8(terrain_grid)
    terrain_vis = cv2.cvtColor(terrain_vis, cv2.COLOR_GRAY2BGR)

    saliency_max = float(np.max(saliency_grid)) if saliency_grid.size > 0 else 0.0
    if saliency_max < 1e-8:
        saliency_vis = np.zeros_like(terrain_vis)
    else:
        saliency_vis = cv2.applyColorMap(_normalize_to_uint8(saliency_grid), cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(terrain_vis, 0.35, saliency_vis, 0.65, 0.0)

    height, width = saliency_grid.shape
    overlay = cv2.resize(overlay, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((panel_size + 116, panel_size + 128, 3), 245, dtype=np.uint8)
    y0, x0 = 52, 64
    canvas[y0 : y0 + panel_size, x0 : x0 + panel_size] = overlay

    for row in range(height + 1):
        y = y0 + int(round(row * panel_size / height))
        cv2.line(canvas, (x0, y), (x0 + panel_size, y), (90, 90, 90), 1, cv2.LINE_AA)
    for col in range(width + 1):
        x = x0 + int(round(col * panel_size / width))
        cv2.line(canvas, (x, y0), (x, y0 + panel_size), (90, 90, 90), 1, cv2.LINE_AA)

    cv2.putText(canvas, _terrain_saliency_mode_title(saliency_mode), (18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (24, 24, 24), 2, cv2.LINE_AA)
    cv2.putText(canvas, "front", (x0 + panel_size // 2 - 22, y0 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "back", (x0 + panel_size // 2 - 18, y0 + panel_size + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "left", (x0 - 40, y0 + panel_size // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "right", (x0 + panel_size + 12, y0 + panel_size // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        _terrain_saliency_mode_footer(saliency_mode, saliency_max),
        (18, panel_size + 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    return canvas


def append_panel_below_frame(frame: np.ndarray, panel: np.ndarray, saliency_mode: str = "global_mean") -> np.ndarray:
    footer_h = max(panel.shape[0] + 20, 140)
    canvas = np.full((frame.shape[0] + footer_h, frame.shape[1], 3), 242, dtype=np.uint8)
    canvas[: frame.shape[0]] = frame

    y0 = frame.shape[0] + (footer_h - panel.shape[0]) // 2
    x0 = 20
    canvas[y0 : y0 + panel.shape[0], x0 : x0 + panel.shape[1]] = panel

    explainer_x = x0 + panel.shape[1] + 24
    explainer_y = frame.shape[0] + 48
    if saliency_mode == "gradient":
        line1 = "Cold: small local derivative of the policy action wrt that cell"
        line2 = "Hot: large local derivative of the policy action wrt that cell"
    elif saliency_mode == "local_mean":
        line1 = "Cold: replacing that cell with local mean barely changes the action"
        line2 = "Hot: replacing that cell with local mean changes the action more"
    else:
        line1 = "Cold: replacing that cell with global mean barely changes the action"
        line2 = "Hot: replacing that cell with global mean changes the action more"
    cv2.putText(canvas, line1, (explainer_x, explainer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (explainer_x, explainer_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Base gray image is the actor terrain height map around the torso", (explainer_x, explainer_y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
    return canvas
