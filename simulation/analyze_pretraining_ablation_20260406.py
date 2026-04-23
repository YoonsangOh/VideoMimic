from __future__ import annotations

import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import h5py
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R, Slerp


REPO_ROOT = Path("/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic")
SIM_ROOT = REPO_ROOT / "simulation"
REPORT_DIR = SIM_ROOT / "docs" / "reports"
REPORT_DIR.mkdir(exist_ok=True)

MCPT_CHECKPOINT = SIM_ROOT / "data/checkpoints/20250410_063030_g1_deepmimic/model_300000.pt"


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    display_name: str
    raw_mp4_path: str | None
    ghost_mp4_path: str
    source_motion_path: str
    source_mesh_path: str
    capture_dir: str
    pretrain_eval_root: str
    scratch_eval_root: str
    pretrain_run_name: str
    scratch_run_name: str
    pretrain_model_path: str
    scratch_model_path: str
    extra_notes: str = ""


CLIPS: List[ClipSpec] = [
    ClipSpec(
        clip_id="5568",
        display_name="anthony_apr21_IMG_5568",
        raw_mp4_path=str(
            SIM_ROOT
            / "data/videomimic_raw_videos_mp4/videomimic_release_video_data__anthony_apr21_IMG_5568__cam01.mp4"
        ),
        ghost_mp4_path=str(
            SIM_ROOT
            / "data/videomimic_captures_mp4/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5568_cam01_frame_0_350_subsample_1.mp4"
        ),
        source_motion_path=str(
            SIM_ROOT
            / "data/videomimic_captures/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5568_cam01_frame_0_350_subsample_1/retarget_poses_g1.h5"
        ),
        source_mesh_path=str(
            SIM_ROOT
            / "data/videomimic_captures/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5568_cam01_frame_0_350_subsample_1/background_mesh.obj"
        ),
        capture_dir=str(
            SIM_ROOT
            / "data/videomimic_captures/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5568_cam01_frame_0_350_subsample_1"
        ),
        pretrain_eval_root=str(
            SIM_ROOT / "data/single_clip_policy_eval/20260331_040113_0331_ft_5568_ckpt330000"
        ),
        scratch_eval_root=str(
            SIM_ROOT / "data/single_clip_policy_eval/20260402_012652_0401_sc_5568_ckpt30000"
        ),
        pretrain_run_name="20260331_040113_0331_ft_5568",
        scratch_run_name="20260402_012652_0401_sc_5568",
        pretrain_model_path=str(
            SIM_ROOT / "videomimic_gym/logs/g1_deepmimic/20260331_040113_0331_ft_5568/model_330000.pt"
        ),
        scratch_model_path=str(
            SIM_ROOT / "videomimic_gym/logs/g1_deepmimic/20260402_012652_0401_sc_5568/model_30000.pt"
        ),
    ),
    ClipSpec(
        clip_id="5585",
        display_name="anthony_apr21_IMG_5585",
        raw_mp4_path=str(
            SIM_ROOT
            / "data/videomimic_raw_videos_mp4/videomimic_release_video_data__anthony_apr21_IMG_5585__cam01.mp4"
        ),
        ghost_mp4_path=str(
            SIM_ROOT
            / "data/videomimic_captures_mp4/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5585_cam01_frame_0_350_subsample_1.mp4"
        ),
        source_motion_path=str(
            SIM_ROOT
            / "data/videomimic_captures/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5585_cam01_frame_0_350_subsample_1/retarget_poses_g1.h5"
        ),
        source_mesh_path=str(
            SIM_ROOT
            / "data/videomimic_captures/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5585_cam01_frame_0_350_subsample_1/background_mesh.obj"
        ),
        capture_dir=str(
            SIM_ROOT
            / "data/videomimic_captures/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5585_cam01_frame_0_350_subsample_1"
        ),
        pretrain_eval_root=str(
            SIM_ROOT / "data/single_clip_policy_eval/20260331_040113_0331_ft_5585_ckpt330000"
        ),
        scratch_eval_root=str(
            SIM_ROOT / "data/single_clip_policy_eval/20260402_012652_0401_sc_5585_ckpt30000"
        ),
        pretrain_run_name="20260331_040113_0331_ft_5585",
        scratch_run_name="20260402_012652_0401_sc_5585",
        pretrain_model_path=str(
            SIM_ROOT / "videomimic_gym/logs/g1_deepmimic/20260331_040113_0331_ft_5585/model_330000.pt"
        ),
        scratch_model_path=str(
            SIM_ROOT / "videomimic_gym/logs/g1_deepmimic/20260402_012652_0401_sc_5585/model_30000.pt"
        ),
    ),
    ClipSpec(
        clip_id="7276seg2",
        display_name="apr21_IMG_7276 seg2",
        raw_mp4_path=str(
            SIM_ROOT
            / "data/videomimic_raw_videos_mp4/videomimic_release_video_data__apr21_IMG_7276-00.02.11.110-00.02.16.628-seg2__cam01.mp4"
        ),
        ghost_mp4_path=str(
            SIM_ROOT
            / "data/videomimic_captures_mp4/megahunter_align3r_reconstruction_results_apr21_IMG_7276-00.02.11.110-00.02.16.628-seg2_cam01_frame_0_300_subsample_2.pkl.mp4"
        ),
        source_motion_path=str(
            SIM_ROOT
            / "data/videomimic_captures/megahunter_align3r_reconstruction_results_apr21_IMG_7276-00.02.11.110-00.02.16.628-seg2_cam01_frame_0_300_subsample_2.pkl/retarget_poses_g1.h5"
        ),
        source_mesh_path=str(
            SIM_ROOT
            / "data/videomimic_captures/megahunter_align3r_reconstruction_results_apr21_IMG_7276-00.02.11.110-00.02.16.628-seg2_cam01_frame_0_300_subsample_2.pkl/background_mesh.obj"
        ),
        capture_dir=str(
            SIM_ROOT
            / "data/videomimic_captures/megahunter_align3r_reconstruction_results_apr21_IMG_7276-00.02.11.110-00.02.16.628-seg2_cam01_frame_0_300_subsample_2.pkl"
        ),
        pretrain_eval_root=str(
            SIM_ROOT / "data/single_clip_policy_eval/20260331_040157_0331_ft_7276seg2_ckpt330000"
        ),
        scratch_eval_root=str(
            SIM_ROOT / "data/single_clip_policy_eval/20260402_012720_0401_sc_7276seg2_ckpt30000"
        ),
        pretrain_run_name="20260331_040157_0331_ft_7276seg2",
        scratch_run_name="20260402_012720_0401_sc_7276seg2",
        pretrain_model_path=str(
            SIM_ROOT / "videomimic_gym/logs/g1_deepmimic/20260331_040157_0331_ft_7276seg2/model_330000.pt"
        ),
        scratch_model_path=str(
            SIM_ROOT / "videomimic_gym/logs/g1_deepmimic/20260402_012720_0401_sc_7276seg2/model_30000.pt"
        ),
    ),
    ClipSpec(
        clip_id="holosoma_stairs",
        display_name="holosoma_stairs",
        raw_mp4_path=None,
        ghost_mp4_path=str(
            SIM_ROOT
            / "data/holosoma_stairs_reference_motion/holosoma_stairs_cam01_frame_0_140_subsample_1_reference_follow.mp4"
        ),
        source_motion_path=str(
            Path("/home/nas4_user/kyungminlee/work/yoonsangoh/holosoma/data/stairs/retarget_poses_g1.h5")
        ),
        source_mesh_path=str(
            Path("/home/nas4_user/kyungminlee/work/yoonsangoh/holosoma/data/stairs/scene_mesh_gravity_aligned.obj")
        ),
        capture_dir=str(SIM_ROOT / "data/videomimic_captures/holosoma_stairs_cam01_frame_0_140_subsample_1"),
        pretrain_eval_root=str(
            SIM_ROOT
            / "data/holosoma_stairs_policy_eval/20260316_034146_holosoma_stairs_heightfield_single_gpu_30k_20260316_ckpt330000"
        ),
        scratch_eval_root=str(
            SIM_ROOT
            / "data/holosoma_stairs_policy_eval/20260321_072255_holosoma_stairs_heightfield_single_gpu_30k_scratch_20260321_ckpt30000"
        ),
        pretrain_run_name="20260316_034146_holosoma_stairs_heightfield_single_gpu_30k_20260316",
        scratch_run_name="20260321_072255_holosoma_stairs_heightfield_single_gpu_30k_scratch_20260321",
        pretrain_model_path=str(
            SIM_ROOT
            / "videomimic_gym/logs/g1_deepmimic/20260316_034146_holosoma_stairs_heightfield_single_gpu_30k_20260316/model_330000.pt"
        ),
        scratch_model_path=str(
            SIM_ROOT
            / "videomimic_gym/logs/g1_deepmimic/20260321_072255_holosoma_stairs_heightfield_single_gpu_30k_scratch_20260321/model_30000.pt"
        ),
        extra_notes="raw mp4 is not present under videomimic_raw_videos_mp4; capture assets inside videomimic point to the holosoma repo",
    ),
]


def decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [decode_attr(v) for v in value.tolist()]
    if isinstance(value, list):
        return [decode_attr(v) for v in value]
    return value


def read_h5_reference(path: Path, target_fps: float = 50.0) -> Dict[str, Any]:
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        fps_attr = attrs.get("fps", attrs.get("/fps"))
        joint_names_attr = attrs.get("joint_names", attrs.get("/joint_names"))
        link_names_attr = attrs.get("link_names", attrs.get("/link_names"))
        data = {
            "root_pos": np.asarray(f["root_pos"][:], dtype=np.float64),
            "root_quat": np.asarray(f["root_quat"][:], dtype=np.float64),
            "joints": np.asarray(f["joints"][:], dtype=np.float64),
            "link_pos": np.asarray(f["link_pos"][:], dtype=np.float64),
            "link_quat": np.asarray(f["link_quat"][:], dtype=np.float64),
            "contacts": {
                key: np.asarray(f["contacts"][key][:], dtype=bool) for key in f["contacts"].keys()
            },
            "fps": float(fps_attr),
            "joint_names": decode_attr(joint_names_attr),
            "link_names": decode_attr(link_names_attr),
        }
    return upsample_reference(data, data["fps"], target_fps=target_fps)


def upsample_reference(data: Dict[str, Any], source_fps: float, target_fps: float = 50.0) -> Dict[str, Any]:
    factor = target_fps / source_fps
    if abs(factor - 1.0) < 0.01:
        return data

    num_frames = data["root_pos"].shape[0]
    up_frames = int(math.ceil(num_frames * factor))
    original_time = np.linspace(0.0, num_frames / source_fps, num_frames)
    upsampled_time = np.linspace(0.0, num_frames / source_fps, up_frames)

    out: Dict[str, Any] = {
        "fps": target_fps,
        "joint_names": data["joint_names"],
        "link_names": data["link_names"],
    }

    out["root_pos"] = np.zeros((up_frames, data["root_pos"].shape[1]), dtype=np.float64)
    out["joints"] = np.zeros((up_frames, data["joints"].shape[1]), dtype=np.float64)
    for i in range(data["root_pos"].shape[1]):
        out["root_pos"][:, i] = np.interp(upsampled_time, original_time, data["root_pos"][:, i])
    for i in range(data["joints"].shape[1]):
        out["joints"][:, i] = np.interp(upsampled_time, original_time, data["joints"][:, i])

    rot_times = np.arange(num_frames)
    interp_times = np.interp(upsampled_time, original_time, rot_times)

    out["root_quat"] = Slerp(rot_times, R.from_quat(data["root_quat"]))(interp_times).as_quat()

    num_links = data["link_pos"].shape[1]
    out["link_pos"] = np.zeros((up_frames, num_links, 3), dtype=np.float64)
    out["link_quat"] = np.zeros((up_frames, num_links, 4), dtype=np.float64)
    for link_idx in range(num_links):
        for axis in range(3):
            out["link_pos"][:, link_idx, axis] = np.interp(
                upsampled_time, original_time, data["link_pos"][:, link_idx, axis]
            )
        out["link_quat"][:, link_idx] = Slerp(
            rot_times, R.from_quat(data["link_quat"][:, link_idx])
        )(interp_times).as_quat()

    out["contacts"] = {}
    for key, values in data["contacts"].items():
        indices = np.round(np.linspace(0, len(values) - 1, up_frames)).astype(int)
        out["contacts"][key] = values[indices]
    return out


def quat_geodesic_deg(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    px, pw = pred[..., :3], pred[..., 3:4]
    rx, rw = ref[..., :3], ref[..., 3:4]
    xyz = pw * rx - rw * px - np.cross(px, rx)
    w = pw * rw + np.sum(px * rx, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(np.linalg.norm(xyz, axis=-1), np.clip(np.abs(w[..., 0]), 1e-8, None))
    angle = np.where(angle > np.pi, 2.0 * np.pi - angle, angle)
    return np.degrees(angle)


def prefix_slice(arr: np.ndarray, n: int = 50) -> np.ndarray:
    return arr[: min(len(arr), n)]


def rollout_metrics(summary_entry: Dict[str, Any], rollout_path: Path, ref_cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    with open(rollout_path, "rb") as f:
        rollout = pickle.load(f)

    source_motion_path = summary_entry["source_motion_path"]
    if source_motion_path not in ref_cache:
        ref_cache[source_motion_path] = read_h5_reference(Path(source_motion_path), target_fps=rollout["fps"])
    ref = ref_cache[source_motion_path]

    start_offset = int(summary_entry["start_offset"])
    steps = int(rollout["joints"].shape[0])
    remaining = max(len(ref["joints"]) - start_offset, 1)
    aligned_steps = min(steps, remaining)

    ref_slice = {
        "joints": ref["joints"][start_offset : start_offset + aligned_steps],
        "root_pos": ref["root_pos"][start_offset : start_offset + aligned_steps],
        "root_quat": ref["root_quat"][start_offset : start_offset + aligned_steps],
        "link_pos": ref["link_pos"][start_offset : start_offset + aligned_steps],
        "contacts": {
            key: values[start_offset : start_offset + aligned_steps]
            for key, values in ref["contacts"].items()
        },
    }

    rollout_joint = np.asarray(rollout["joints"], dtype=np.float64)[:aligned_steps]
    rollout_root_pos = np.asarray(rollout["root_pos"], dtype=np.float64)[:aligned_steps]
    rollout_root_quat = np.asarray(rollout["root_quat"], dtype=np.float64)[:aligned_steps]

    joint_err_deg = np.degrees(rollout_joint - ref_slice["joints"])
    root_pos_l2_cm = np.linalg.norm(rollout_root_pos - ref_slice["root_pos"], axis=-1) * 100.0
    root_xy_l2_cm = np.linalg.norm(rollout_root_pos[:, :2] - ref_slice["root_pos"][:, :2], axis=-1) * 100.0
    root_quat_deg = quat_geodesic_deg(rollout_root_quat, ref_slice["root_quat"])

    rollout_link_idx = {name: idx for idx, name in enumerate(rollout["link_names"])}
    ref_link_idx = {name: idx for idx, name in enumerate(ref["link_names"])}
    common_links = [name for name in ref["link_names"] if name in rollout_link_idx]
    r_idx = [rollout_link_idx[name] for name in common_links]
    q_idx = [ref_link_idx[name] for name in common_links]
    link_err_cm = (
        np.linalg.norm(
            np.asarray(rollout["link_pos"], dtype=np.float64)[:, r_idx]
            [:aligned_steps]
            - ref_slice["link_pos"][:, q_idx],
            axis=-1,
        )
        * 100.0
    )

    contact_acc = {}
    for key in ["left_foot", "right_foot"]:
        contact_acc[key] = float(
            (
                np.asarray(rollout["contacts"][key], dtype=bool)
                [:aligned_steps]
                == np.asarray(ref_slice["contacts"][key], dtype=bool)
            ).mean()
        )

    joint_prefix = prefix_slice(joint_err_deg)
    link_prefix = prefix_slice(link_err_cm)
    root_prefix = prefix_slice(root_pos_l2_cm)

    return {
        "eval_index": int(summary_entry["eval_index"]),
        "start_offset": start_offset,
        "steps": steps,
        "remaining_ref_steps": remaining,
        "success_by_timeout": bool(summary_entry["success_by_timeout"]),
        "completion_ratio": float(steps / remaining),
        "joint_rmse_deg": float(np.sqrt(np.mean(np.square(joint_err_deg)))),
        "joint_mae_deg": float(np.mean(np.abs(joint_err_deg))),
        "joint_rmse_deg_first50": float(np.sqrt(np.mean(np.square(joint_prefix)))),
        "root_pos_mean_cm": float(np.mean(root_pos_l2_cm)),
        "root_pos_p95_cm": float(np.percentile(root_pos_l2_cm, 95)),
        "root_xy_mean_cm": float(np.mean(root_xy_l2_cm)),
        "root_quat_mean_deg": float(np.mean(root_quat_deg)),
        "link_pos_mean_cm": float(np.mean(link_err_cm)),
        "link_pos_p95_cm": float(np.percentile(link_err_cm, 95)),
        "link_pos_mean_cm_first50": float(np.mean(link_prefix)),
        "root_pos_mean_cm_first50": float(np.mean(root_prefix)),
        "contact_acc_left": contact_acc["left_foot"],
        "contact_acc_right": contact_acc["right_foot"],
        "contact_acc_mean": float((contact_acc["left_foot"] + contact_acc["right_foot"]) / 2.0),
        "common_link_count": int(len(common_links)),
        "_aligned_steps": aligned_steps,
        "_joint_err_deg": joint_err_deg,
        "_root_pos_l2_cm": root_pos_l2_cm,
        "_link_err_cm": link_err_cm,
    }


def aggregate_eval(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_steps = sum(item["steps"] for item in metrics)
    success_count = sum(int(item["success_by_timeout"]) for item in metrics)

    def weighted_mean(key: str) -> float:
        return float(sum(item[key] * item["steps"] for item in metrics) / max(total_steps, 1))

    def mean(key: str) -> float:
        return float(np.mean([item[key] for item in metrics]))

    return {
        "num_evals": len(metrics),
        "success_rate_5": success_count / max(len(metrics), 1),
        "mean_completion_ratio": mean("completion_ratio"),
        "mean_steps": mean("steps"),
        "weighted_joint_rmse_deg": weighted_mean("joint_rmse_deg"),
        "weighted_joint_mae_deg": weighted_mean("joint_mae_deg"),
        "weighted_root_pos_mean_cm": weighted_mean("root_pos_mean_cm"),
        "weighted_root_xy_mean_cm": weighted_mean("root_xy_mean_cm"),
        "weighted_root_quat_mean_deg": weighted_mean("root_quat_mean_deg"),
        "weighted_link_pos_mean_cm": weighted_mean("link_pos_mean_cm"),
        "weighted_link_pos_p95_cm": weighted_mean("link_pos_p95_cm"),
        "weighted_contact_acc_mean": weighted_mean("contact_acc_mean"),
        "mean_joint_rmse_deg_first50": mean("joint_rmse_deg_first50"),
        "mean_link_pos_mean_cm_first50": mean("link_pos_mean_cm_first50"),
        "mean_root_pos_mean_cm_first50": mean("root_pos_mean_cm_first50"),
    }


def parse_obj_stats(path: Path) -> Dict[str, Any]:
    num_vertices = 0
    num_faces = 0
    z_min = float("inf")
    z_max = float("-inf")
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                _, _, _, z = line.split()[:4]
                z_value = float(z)
                num_vertices += 1
                z_min = min(z_min, z_value)
                z_max = max(z_max, z_value)
            elif line.startswith("f "):
                num_faces += 1
    return {
        "mesh_vertices": num_vertices,
        "mesh_faces": num_faces,
        "mesh_z_range_m": float(z_max - z_min),
    }


def clip_stats(spec: ClipSpec) -> Dict[str, Any]:
    with h5py.File(spec.source_motion_path, "r") as f:
        fps = float(f.attrs.get("fps", f.attrs.get("/fps")))
        root_pos = np.asarray(f["root_pos"][:], dtype=np.float64)
        contacts = {key: np.asarray(f["contacts"][key][:], dtype=bool) for key in f["contacts"].keys()}

    duration_s = len(root_pos) / fps
    step = np.diff(root_pos, axis=0)
    path_xy_m = float(np.linalg.norm(step[:, :2], axis=-1).sum())
    path_3d_m = float(np.linalg.norm(step, axis=-1).sum())
    net_xy_m = float(np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2]))
    net_z_m = float(root_pos[-1, 2] - root_pos[0, 2])
    z_range_m = float(root_pos[:, 2].max() - root_pos[:, 2].min())
    total_switches = int(sum(np.count_nonzero(v[1:] != v[:-1]) for v in contacts.values()))
    mesh = parse_obj_stats(Path(spec.source_mesh_path))
    return {
        "source_fps": fps,
        "source_frames": int(len(root_pos)),
        "duration_s": duration_s,
        "upsampled_frames_50hz": int(math.ceil(len(root_pos) * (50.0 / fps))),
        "path_xy_m": path_xy_m,
        "path_3d_m": path_3d_m,
        "mean_xy_speed_mps": path_xy_m / duration_s,
        "net_xy_m": net_xy_m,
        "net_z_m": net_z_m,
        "z_range_m": z_range_m,
        "grade_ratio_abs": abs(net_z_m) / max(net_xy_m, 1e-6),
        "contact_switches": total_switches,
        "contact_switches_per_s": total_switches / duration_s,
        **mesh,
    }


def read_wandb_summaries() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for config_path in Path("/tmp/wandb").glob("run-*/files/config.yaml"):
        summary_path = config_path.parent / "wandb-summary.json"
        if not summary_path.exists():
            continue
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        run_name = config.get("run_name", {}).get("value")
        if not run_name:
            continue
        with open(summary_path, "r") as f:
            summary = json.load(f)
        out[run_name] = summary
    return out


def select_training_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "train_success_overall": float(summary["success/overall"]),
        "train_mean_episode_length": float(summary["Train/mean_episode_length"]),
        "train_mean_reward": float(summary["Train/mean_reward"]),
        "collection_time_s": float(summary["Perf/collection time"]),
        "learning_time_s": float(summary["Perf/learning_time"]),
        "total_fps": float(summary["Perf/total_fps"]),
        "actor_terrain_attention": float(summary["Network/attention_terrain_height_actor"]),
        "critic_terrain_attention": float(summary["Network/attention_terrain_height_critic"]),
        "actor_max_terrain_attention": float(summary["Network/max_attention_terrain_height_actor"]),
        "critic_max_terrain_attention": float(summary["Network/max_attention_terrain_height_critic"]),
    }


def load_checkpoint_state(path: Path) -> Dict[str, np.ndarray]:
    import torch

    ckpt = torch.load(path, map_location="cpu")
    model = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    out = {}
    for key, value in model.items():
        out[key] = value.detach().cpu().numpy().astype(np.float64)
    return out


def checkpoint_similarity(mcpt: Dict[str, np.ndarray], final_state: Dict[str, np.ndarray]) -> Dict[str, Any]:
    common_keys = [key for key in mcpt if key in final_state and (key.startswith("actor.") or key.startswith("critic.") or key == "std")]
    ref = np.concatenate([mcpt[key].reshape(-1) for key in common_keys])
    cur = np.concatenate([final_state[key].reshape(-1) for key in common_keys])
    dot = float(np.dot(ref, cur))
    denom = float(np.linalg.norm(ref) * np.linalg.norm(cur))
    cosine = dot / denom if denom > 0 else 0.0
    rel_l2 = float(np.linalg.norm(cur - ref) / max(np.linalg.norm(ref), 1e-8))
    return {
        "shared_param_count": int(ref.size),
        "cosine_to_mcpt": cosine,
        "relative_l2_to_mcpt": rel_l2,
    }


def build_report(
    eval_summary_rows: List[Dict[str, Any]],
    clip_rows: List[Dict[str, Any]],
    train_rows: List[Dict[str, Any]],
    pair_rows: List[Dict[str, Any]],
    holosoma_notes: Dict[str, Any],
) -> str:
    def fmt(x: Any, digits: int = 2) -> str:
        if x is None:
            return "-"
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.{digits}f}"
        return str(x)

    lines: List[str] = []
    lines.append("# Pretraining Ablation Report (2026-04-06)")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Four clips were compared as pretrained-vs-scratch pairs using the same stage-2 scene-aware tracking setup.")
    lines.append("- The three VideoMimic clips use raw mp4 + processed `retarget_poses_g1.h5` + `background_mesh.obj` inside this repo.")
    lines.append("- `holosoma_stairs` has processed assets inside `videomimic_captures`, but its underlying `h5/obj` are symlinks to the separate `holosoma` repo and there is no raw holosoma mp4 under `simulation/data/videomimic_raw_videos_mp4`.")
    lines.append("- Metrics below go beyond success rate: joint RMSE, link-position error, root error, contact accuracy, and completion ratio were computed from saved rollout pickles aligned to the source reference motion at 50 Hz.")
    lines.append("")
    lines.append("## Key Conclusions")
    lines.append("")
    lines.append("1. Pretraining helps strongly on `5568` and `5585` in both completion and matched-horizon tracking error. On `7276seg2`, pretraining still helps but the gap is small because scratch already learns the clip well.")
    lines.append("2. `holosoma_stairs` is the hardest clip under equal compute. Pretraining clearly improves completion there, but matched-horizon tracking error does not improve much. The main gain is staying alive longer, not making the very early steps cleaner.")
    lines.append("3. `holosoma_stairs` is the longest clip, has the largest uphill displacement, the largest motion z-range, and the largest scene z-range. These factors align with its weaker equal-budget scratch result.")
    lines.append("4. Success rate alone is misleading. In particular, `holosoma_stairs` pretrained can survive many offsets while still showing materially larger tracking error than the easy clips.")
    lines.append("")
    lines.append("## Clip Assets")
    lines.append("")
    lines.append("| Clip | Raw video in repo | Ghost/reference replay | Processed motion | Processed scene |")
    lines.append("| --- | --- | --- | --- | --- |")
    for spec in CLIPS:
        raw = spec.raw_mp4_path or "not present under videomimic raw mp4 folder"
        lines.append(
            f"| {spec.clip_id} | `{raw}` | `{spec.ghost_mp4_path}` | `{spec.source_motion_path}` | `{spec.source_mesh_path}` |"
        )
    lines.append("")
    lines.append("## Quantitative Tracking Comparison")
    lines.append("")
    lines.append("`success_rate_5` is over the 5 canonical eval offsets already used for the mp4 renders. `completion` is the fraction of the remaining clip completed before termination. `first50` metrics isolate short-horizon tracking quality so that very early failures do not hide initial tracking behavior.")
    lines.append("")
    lines.append("| Clip | Init | success_rate_5 | completion | first50 link err (cm) | first50 joint RMSE (deg) | full link err (cm) | full joint RMSE (deg) | root pos err (cm) | contact acc |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in eval_summary_rows:
        lines.append(
            f"| {row['clip_id']} | {row['init']} | {fmt(row['success_rate_5'])} | {fmt(row['mean_completion_ratio'])} | "
            f"{fmt(row['mean_link_pos_mean_cm_first50'])} | {fmt(row['mean_joint_rmse_deg_first50'])} | "
            f"{fmt(row['weighted_link_pos_mean_cm'])} | {fmt(row['weighted_joint_rmse_deg'])} | "
            f"{fmt(row['weighted_root_pos_mean_cm'])} | {fmt(row['weighted_contact_acc_mean'])} |"
        )
    lines.append("")
    lines.append("## What Pretraining Changed")
    lines.append("")
    lines.append("| Clip | success delta | completion delta | matched-horizon link err delta (cm) | matched-horizon joint RMSE delta (deg) | full link err delta (cm) | cosine to MCPT: pretrained | cosine to MCPT: scratch |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in pair_rows:
        lines.append(
            f"| {row['clip_id']} | {fmt(row['success_delta'])} | {fmt(row['completion_delta'])} | "
            f"{fmt(row['matched_link_delta_cm'])} | {fmt(row['matched_joint_delta_deg'])} | "
            f"{fmt(row['full_link_delta_cm'])} | {fmt(row['pretrain_cosine_to_mcpt'], 4)} | {fmt(row['scratch_cosine_to_mcpt'], 4)} |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- Negative error deltas above mean the pretrained model tracks better than scratch.")
    lines.append("- `matched-horizon` compares both models over the same number of steps for each eval offset, which removes the survivorship bias that otherwise makes very short scratch rollouts look deceptively clean.")
    lines.append("- The MCPT cosine similarity is a network-level check: pretrained stage-2 policies stay much closer to the stage-1 motion prior, while scratch models must discover a workable motion prior from random initialization.")
    lines.append("")
    lines.append("## Difficulty Ranking Under Equal Compute")
    lines.append("")
    lines.append("Difficulty is defined here by equal-budget evidence: lower final train success, lower final train episode length, lower 5-offset eval completion, and larger tracking error after the same 30k stage-2 iterations.")
    lines.append("")
    lines.append("| Clip | source dur (s) | 50 Hz frames | path_xy (m) | net_z (m) | motion z-range (m) | scene z-range (m) | switches/s | train success scratch | train ep len scratch | collection time scratch (s) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for clip in clip_rows:
        scratch_train = next(
            row for row in train_rows if row["clip_id"] == clip["clip_id"] and row["init"] == "scratch"
        )
        lines.append(
            f"| {clip['clip_id']} | {fmt(clip['duration_s'])} | {fmt(clip['upsampled_frames_50hz'], 0)} | {fmt(clip['path_xy_m'])} | "
            f"{fmt(clip['net_z_m'])} | {fmt(clip['z_range_m'])} | {fmt(clip['mesh_z_range_m'])} | {fmt(clip['contact_switches_per_s'])} | "
            f"{fmt(scratch_train['train_success_overall'])} | {fmt(scratch_train['train_mean_episode_length'])} | {fmt(scratch_train['collection_time_s'])} |"
        )
    lines.append("")
    lines.append("Suggested ranking from easiest to hardest at 30k stage-2 steps:")
    lines.append("")
    lines.append("1. `7276seg2`: scratch already reaches 5/5 success on the canonical 5-offset evals.")
    lines.append("2. `5568`: pretraining helps a lot, but scratch still learns a partially workable tracker.")
    lines.append("3. `5585`: short clip, but scratch remains unstable and never succeeds on the 5 canonical evals.")
    lines.append("4. `holosoma_stairs`: longest clip, largest climb, largest terrain relief, and the weakest equal-budget scratch outcome.")
    lines.append("")
    lines.append("## Why `holosoma_stairs` Is Harder")
    lines.append("")
    lines.append("Evidence-backed reasons:")
    lines.append("")
    lines.append("- It is the longest sequence: 467 policy steps at 50 Hz, versus 407 / 290 / 304 for the other three clips.")
    lines.append("- It covers by far the largest spatial excursion: 6.29 m of xy path length and +2.37 m net vertical gain.")
    lines.append("- Its scene relief is also the largest: mesh z-range 5.18 m, versus 1.44 / 0.71 / 3.18 m for the other clips.")
    lines.append("- The scratch policy remains weak after 30k steps despite similar network structure and hyperparameters.")
    lines.append("- The pretrained model improves survival a lot, but its tracking error remains materially above the easy `7276seg2` case, which shows that success under a 0.5 m termination rule does not imply high-fidelity tracking.")
    lines.append("- On a matched horizon, holosoma shows much smaller error gains from pretraining than `5568` or `5585`, which suggests the main benefit is a better motion prior for long-horizon stability rather than an immediate low-level correction.")
    lines.append("")
    lines.append("A likely mechanism is that stairs require coordinated foothold placement over a long uphill trajectory. The policy has to preserve the whole-body motion prior and learn terrain-conditioned corrections at the same time. In the easier clips, motion can remain closer to flat-ground whole-body imitation, so scratch optimization is less brittle.")
    lines.append("")
    lines.append("## Network-Level Observations")
    lines.append("")
    lines.append("- The stage-1 MCPT checkpoint has no terrain-specific parameter keys. The stage-2 scene-aware policies add `actor_input_net.extra_proj_heads.terrain_height.*` and `critic_input_net.extra_proj_heads.terrain_height.*` on top of the shared actor/critic trunk.")
    lines.append("- This means scene awareness is added as a terrain projection head during stage-2 finetuning, not as a completely different policy architecture.")
    lines.append("- Final terrain-attention magnitudes are clip-dependent. The easiest clip, `7276seg2`, ends with the strongest terrain-attention signal. `holosoma_stairs` remains comparatively weak here, which is consistent with the optimizer struggling to exploit terrain observations effectively on the hardest uphill clip.")
    lines.append("")
    lines.append("| Clip | Init | train success | train ep len | actor terrain attn | critic terrain attn | actor max terrain attn | critic max terrain attn |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in train_rows:
        lines.append(
            f"| {row['clip_id']} | {row['init']} | {fmt(row['train_success_overall'])} | {fmt(row['train_mean_episode_length'])} | "
            f"{fmt(row['actor_terrain_attention'])} | {fmt(row['critic_terrain_attention'])} | "
            f"{fmt(row['actor_max_terrain_attention'])} | {fmt(row['critic_max_terrain_attention'])} |"
        )
    lines.append("")
    lines.append("## Success-Rate Caveat for `holosoma_stairs`")
    lines.append("")
    lines.append(
        f"- Existing 100-episode evals already show that success is protocol-sensitive: `random-start 100eps` success is {holosoma_notes['random_success_rate']:.2f}, but `first-frame 100eps` success is {holosoma_notes['first_frame_success_rate']:.2f}."
    )
    lines.append(
        f"- On the `0..50`-offset window with the train-time threshold (0.5 m), pretrained reaches {holosoma_notes['window_success_rate']:.2f}; with a stricter 0.3 m eval threshold, it drops to {holosoma_notes['window_thr03_success_rate']:.2f}."
    )
    lines.append("- This is why the report above emphasizes tracking error and completion ratio, not success rate alone.")
    lines.append("")
    lines.append("## Grounding in the Official VideoMimic Pipeline")
    lines.append("")
    lines.append("- The root repo states that the sim pipeline has four stages: motion-capture pretraining, scene-conditioned tracking, distillation, and RL finetuning.")
    lines.append(f"- See `{REPO_ROOT / 'README.md'}` and `{SIM_ROOT / 'README.md'}`.")
    lines.append("- The official stage-2 script uses `human_motion_list_123_motions.yaml` in a single run, which is direct evidence that stage-2 trains one multi-clip teacher rather than 123 independent clip policies.")
    lines.append(
        f"- See `{SIM_ROOT / 'videomimic_gym/legged_gym/scripts/train_stage_2_terrain_rl.sh'}` and `{SIM_ROOT / 'videomimic_gym/legged_gym/scripts/train_stage_3_distillation.sh'}`."
    )
    lines.append("- Stage-3 then takes a single `LOAD_RUN=stage_2_run_name`, again indicating one teacher policy is distilled, not 123 separate teachers.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- Eval metrics CSV: `{REPORT_DIR / 'pretraining_ablation_20260406_eval_metrics.csv'}`")
    lines.append(f"- Training summary CSV: `{REPORT_DIR / 'pretraining_ablation_20260406_training_summary.csv'}`")
    lines.append(f"- Clip stats CSV: `{REPORT_DIR / 'pretraining_ablation_20260406_clip_stats.csv'}`")
    lines.append(f"- Pairwise delta CSV: `{REPORT_DIR / 'pretraining_ablation_20260406_pairwise_delta.csv'}`")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    filtered_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(filtered_rows[0].keys()))
        writer.writeheader()
        writer.writerows(filtered_rows)


def main() -> None:
    ref_cache: Dict[str, Dict[str, Any]] = {}
    wandb_summaries = read_wandb_summaries()

    import torch

    mcpt_state = load_checkpoint_state(MCPT_CHECKPOINT)

    eval_summary_rows: List[Dict[str, Any]] = []
    eval_detail_rows: List[Dict[str, Any]] = []
    clip_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    clip_eval_lookup: Dict[tuple[str, str], Dict[str, Any]] = {}
    train_lookup: Dict[tuple[str, str], Dict[str, Any]] = {}

    for spec in CLIPS:
        clip_rows.append({"clip_id": spec.clip_id, **clip_stats(spec)})

        per_eval_lookup: Dict[str, List[Dict[str, Any]]] = {}
        for init, eval_root_str, model_path_str, run_name in [
            ("pretrained", spec.pretrain_eval_root, spec.pretrain_model_path, spec.pretrain_run_name),
            ("scratch", spec.scratch_eval_root, spec.scratch_model_path, spec.scratch_run_name),
        ]:
            eval_root = Path(eval_root_str)
            with open(eval_root / "summary_eval.json", "r") as f:
                summary = json.load(f)
            per_eval = []
            for entry in summary:
                eval_index = int(entry["eval_index"])
                rollout_path = eval_root / "rollouts" / f"eval_{eval_index:02d}" / "policy_rollout.pkl"
                metrics = rollout_metrics(entry, rollout_path, ref_cache)
                metrics["clip_id"] = spec.clip_id
                metrics["init"] = init
                per_eval.append(metrics)
                eval_detail_rows.append(metrics)
            per_eval_lookup[init] = per_eval
            aggregate = {"clip_id": spec.clip_id, "init": init, **aggregate_eval(per_eval)}
            eval_summary_rows.append(aggregate)
            clip_eval_lookup[(spec.clip_id, init)] = aggregate

            run_summary = select_training_summary(wandb_summaries[run_name])
            similarity = checkpoint_similarity(mcpt_state, load_checkpoint_state(Path(model_path_str)))
            train_row = {"clip_id": spec.clip_id, "init": init, **run_summary, **similarity}
            train_rows.append(train_row)
            train_lookup[(spec.clip_id, init)] = train_row

        pre = clip_eval_lookup[(spec.clip_id, "pretrained")]
        sc = clip_eval_lookup[(spec.clip_id, "scratch")]
        train_pre = train_lookup[(spec.clip_id, "pretrained")]
        train_sc = train_lookup[(spec.clip_id, "scratch")]

        matched_link_deltas = []
        matched_joint_deltas = []
        for pre_eval, sc_eval in zip(per_eval_lookup["pretrained"], per_eval_lookup["scratch"]):
            matched = min(int(pre_eval["_aligned_steps"]), int(sc_eval["_aligned_steps"]))
            matched_link_deltas.append(
                float(np.mean(pre_eval["_link_err_cm"][:matched]) - np.mean(sc_eval["_link_err_cm"][:matched]))
            )
            matched_joint_deltas.append(
                float(np.sqrt(np.mean(np.square(pre_eval["_joint_err_deg"][:matched])))
                      - np.sqrt(np.mean(np.square(sc_eval["_joint_err_deg"][:matched]))))
            )
        pair_rows.append(
            {
                "clip_id": spec.clip_id,
                "success_delta": pre["success_rate_5"] - sc["success_rate_5"],
                "completion_delta": pre["mean_completion_ratio"] - sc["mean_completion_ratio"],
                "matched_link_delta_cm": float(np.mean(matched_link_deltas)),
                "matched_joint_delta_deg": float(np.mean(matched_joint_deltas)),
                "full_link_delta_cm": pre["weighted_link_pos_mean_cm"] - sc["weighted_link_pos_mean_cm"],
                "pretrain_cosine_to_mcpt": train_pre["cosine_to_mcpt"],
                "scratch_cosine_to_mcpt": train_sc["cosine_to_mcpt"],
            }
        )

    holosoma_success_dir = SIM_ROOT / "data/holosoma_stairs_success_eval"
    with open(holosoma_success_dir / "pretrain_vs_scratch_100eps.json", "r") as f:
        random_100 = json.load(f)
    with open(holosoma_success_dir / "pretrain_vs_scratch_first_frame_100eps.json", "r") as f:
        first_frame_100 = json.load(f)
    with open(holosoma_success_dir / "pretrain_vs_scratch_start_window_0_50_100eps.json", "r") as f:
        window_0_50 = json.load(f)
    with open(holosoma_success_dir / "pretrain_vs_scratch_start_window_0_50_thr0p3_100eps.json", "r") as f:
        window_0_50_thr03 = json.load(f)
    def result_by_label(blob: Dict[str, Any], label: str) -> Dict[str, Any]:
        for item in blob["results"]:
            if item["label"] == label:
                return item
        raise KeyError(label)

    holosoma_notes = {
        "random_success_rate": float(result_by_label(random_100, "pretrain")["success_rate"]),
        "first_frame_success_rate": float(result_by_label(first_frame_100, "pretrain")["success_rate"]),
        "window_success_rate": float(result_by_label(window_0_50, "pretrain")["success_rate"]),
        "window_thr03_success_rate": float(result_by_label(window_0_50_thr03, "pretrain")["success_rate"]),
    }

    write_csv(REPORT_DIR / "pretraining_ablation_20260406_eval_metrics.csv", eval_summary_rows)
    write_csv(REPORT_DIR / "pretraining_ablation_20260406_eval_details.csv", eval_detail_rows)
    write_csv(REPORT_DIR / "pretraining_ablation_20260406_training_summary.csv", train_rows)
    write_csv(REPORT_DIR / "pretraining_ablation_20260406_clip_stats.csv", clip_rows)
    write_csv(REPORT_DIR / "pretraining_ablation_20260406_pairwise_delta.csv", pair_rows)

    report_text = build_report(eval_summary_rows, clip_rows, train_rows, pair_rows, holosoma_notes)
    report_path = REPORT_DIR / "pretraining_ablation_20260406.md"
    report_path.write_text(report_text)

    aux = {
        "eval_summary_rows": eval_summary_rows,
        "clip_rows": clip_rows,
        "train_rows": train_rows,
        "pair_rows": pair_rows,
        "holosoma_notes": holosoma_notes,
    }
    with open(REPORT_DIR / "pretraining_ablation_20260406_summary.json", "w") as f:
        json.dump(aux, f, indent=2)

    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
