#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np

from eval_holosoma_stairs_checkpoint_to_mp4 import (
    build_args,
    configure_eval_env,
    destroy_env,
    set_episode_start,
)


@dataclass
class RunSpec:
    label: str
    load_run: str
    checkpoint: int


def parse_run_spec(raw: str) -> RunSpec:
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --run-spec '{raw}'. Expected format label:load_run:checkpoint"
        )
    label, load_run, checkpoint = parts
    return RunSpec(label=label, load_run=load_run, checkpoint=int(checkpoint))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated success-rate evaluation for holosoma stairs checkpoints."
    )
    parser.add_argument("--task", default="g1_deepmimic_proj_heightfield")
    parser.add_argument(
        "--motion-source",
        default="resources/data_config/holosoma_stairs_motion.yaml",
    )
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--link-pos-error-threshold",
        type=float,
        default=None,
        help="Override deepmimic termination threshold used only for evaluation.",
    )
    parser.add_argument(
        "--start-mode",
        choices=["random", "first_frame", "start_window"],
        default="random",
        help="How to choose the replay start frame for each eval episode.",
    )
    parser.add_argument("--start-offset-min", type=int, default=0)
    parser.add_argument("--start-offset-max", type=int, default=50)
    parser.add_argument(
        "--run-spec",
        action="append",
        required=True,
        help="label:load_run:checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation/data/holosoma_stairs_success_eval"),
    )
    return parser.parse_args()


def reset_env_for_eval(env) -> List[Dict[str, int]]:
    import torch

    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(env_ids)
    env.compute_observations()
    return [current_episode_info(env, env_id) for env_id in range(env.num_envs)]


def current_episode_info(env, env_id: int) -> Dict[str, int]:
    clip_index = int(env.replay_data_loader.episode_indices[env_id].item())
    start_offset = int(env.replay_data_loader.index_within_episode[env_id].item())
    clip_length = int(env.replay_data_loader.episode_clip_length[env_id].item())
    max_episode_length = int(env.max_episode_length[env_id].item())
    return {
        "clip_index": clip_index,
        "start_offset": start_offset,
        "clip_length": clip_length,
        "remaining_steps": max_episode_length,
    }


def summarize_episodes(label: str, load_run: str, checkpoint: int, env_dt: float, episodes: List[Dict]) -> Dict:
    success_flags = [int(ep["success_by_timeout"]) for ep in episodes]
    steps = [ep["steps"] for ep in episodes]
    completion = [ep["completion_ratio"] for ep in episodes]
    remaining = [ep["remaining_steps"] for ep in episodes]
    start_offsets = [ep["start_offset"] for ep in episodes]
    start_times = [ep["start_time_sec"] for ep in episodes]

    first_half_eps = [ep for ep in episodes if ep["start_offset"] < ep["clip_length"] / 2.0]
    first_half_success_rate = None
    if first_half_eps:
        first_half_success_rate = mean(int(ep["success_by_timeout"]) for ep in first_half_eps)

    return {
        "label": label,
        "load_run": load_run,
        "checkpoint": checkpoint,
        "num_episodes": len(episodes),
        "env_dt": env_dt,
        "env_fps": round(1.0 / env_dt),
        "success_count": int(sum(success_flags)),
        "success_rate": float(mean(success_flags)) if success_flags else 0.0,
        "mean_steps": float(mean(steps)) if steps else 0.0,
        "mean_duration_sec": float(mean(step * env_dt for step in steps)) if steps else 0.0,
        "mean_completion_ratio": float(mean(completion)) if completion else 0.0,
        "mean_remaining_steps": float(mean(remaining)) if remaining else 0.0,
        "mean_remaining_sec": float(mean(step * env_dt for step in remaining)) if remaining else 0.0,
        "mean_start_offset": float(mean(start_offsets)) if start_offsets else 0.0,
        "mean_start_time_sec": float(mean(start_times)) if start_times else 0.0,
        "success_rate_first_half_starts": float(first_half_success_rate) if first_half_success_rate is not None else None,
        "episodes": episodes,
    }


def run_single_episode_with_fixed_start(env, policy, start_offset: int, episode_index: int) -> Dict:
    clip_index = 0
    set_episode_start(env, clip_index, start_offset)
    env.compute_observations()
    obs = env.get_observations()

    done = False
    steps = 0
    while not done:
        try:
            actions = policy({k: v.detach() for k, v in obs.items()}, monitor_activations=False)
        except TypeError:
            actions = policy({k: v.detach() for k, v in obs.items()})

        obs, _, dones, infos = env.step(actions.detach())
        done = bool(dones[0].item())
        steps += 1

    clip_length = int(env.replay_data_loader.sequence_lengths[clip_index].item())
    remaining_steps = int(env.max_episode_length[0].item())
    success = bool(infos["time_outs"][0].item())
    return {
        "episode_index": episode_index,
        "env_id": 0,
        "clip_index": clip_index,
        "start_offset": int(start_offset),
        "start_time_sec": float(start_offset * env.dt),
        "clip_length": clip_length,
        "remaining_steps": remaining_steps,
        "remaining_sec": float(remaining_steps * env.dt),
        "steps": int(steps),
        "duration_sec": float(steps * env.dt),
        "success_by_timeout": success,
        "completion_ratio": float(steps / max(remaining_steps, 1)),
    }


def evaluate_run(args: argparse.Namespace, run_spec: RunSpec) -> Dict:
    import isaacgym  # noqa: F401
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils.task_registry import task_registry

    sim_num_envs = 1 if args.start_mode == "start_window" else args.num_envs
    sim_args = build_args(seed=args.seed, num_envs=sim_num_envs)
    sim_args.task = args.task
    sim_args.load_run = run_spec.load_run
    sim_args.checkpoint = run_spec.checkpoint

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    configure_eval_env(env_cfg, motion_source=args.motion_source)
    if args.link_pos_error_threshold is not None:
        env_cfg.deepmimic.link_pos_error_threshold = args.link_pos_error_threshold
    if args.start_mode == "start_window":
        env_cfg.env.num_envs = 1
    else:
        env_cfg.env.num_envs = args.num_envs

    env_cfg.env.test = False
    if args.start_mode == "random":
        env_cfg.deepmimic.randomize_start_offset = True
        env_cfg.deepmimic.weighting_strategy = "uniform"
    elif args.start_mode == "first_frame":
        env_cfg.deepmimic.randomize_start_offset = False
    elif args.start_mode == "start_window":
        env_cfg.deepmimic.randomize_start_offset = False
    else:
        raise ValueError(f"Unsupported start mode: {args.start_mode}")

    env, _ = task_registry.make_env(name=args.task, args=sim_args, env_cfg=env_cfg)

    try:
        train_cfg.runner.resume = True
        train_cfg.runner.load_run = run_spec.load_run
        train_cfg.runner.checkpoint = run_spec.checkpoint

        runner, _ = task_registry.make_alg_runner(
            env=env,
            name=args.task,
            args=sim_args,
            train_cfg=train_cfg,
        )
        policy = runner.get_inference_policy(device=env.device)

        if args.start_mode == "start_window":
            rng = np.random.default_rng(args.seed)
            clip_length = int(env.replay_data_loader.sequence_lengths[0].item())
            min_offset = max(0, args.start_offset_min)
            max_offset = min(args.start_offset_max, clip_length - 1)
            if min_offset > max_offset:
                raise ValueError(
                    f"Invalid start window [{args.start_offset_min}, {args.start_offset_max}] for clip length {clip_length}"
                )

            episodes: List[Dict] = []
            for episode_index in range(1, args.num_episodes + 1):
                sampled_offset = int(rng.integers(min_offset, max_offset + 1))
                episode_record = run_single_episode_with_fixed_start(
                    env=env,
                    policy=policy,
                    start_offset=sampled_offset,
                    episode_index=episode_index,
                )
                episodes.append(episode_record)
                print(json.dumps({"label": run_spec.label, **episode_record}))

            return summarize_episodes(
                label=run_spec.label,
                load_run=run_spec.load_run,
                checkpoint=run_spec.checkpoint,
                env_dt=env.dt,
                episodes=episodes,
            )

        current_infos = reset_env_for_eval(env)
        obs = env.get_observations()
        episodes: List[Dict] = []
        episode_steps = np.zeros(env.num_envs, dtype=np.int64)

        while len(episodes) < args.num_episodes:
            try:
                actions = policy({k: v.detach() for k, v in obs.items()}, monitor_activations=False)
            except TypeError:
                actions = policy({k: v.detach() for k, v in obs.items()})

            obs, _, dones, infos = env.step(actions.detach())
            episode_steps += 1

            done_ids = np.nonzero(dones.detach().cpu().numpy())[0]
            for env_id in done_ids:
                if len(episodes) >= args.num_episodes:
                    break

                current_info = current_infos[env_id]
                success = bool(infos["time_outs"][env_id].item())
                steps = int(episode_steps[env_id])
                remaining_steps = max(current_info["remaining_steps"], 1)
                episode_record = {
                    "episode_index": len(episodes) + 1,
                    "env_id": int(env_id),
                    "clip_index": current_info["clip_index"],
                    "start_offset": current_info["start_offset"],
                    "start_time_sec": current_info["start_offset"] * env.dt,
                    "clip_length": current_info["clip_length"],
                    "remaining_steps": current_info["remaining_steps"],
                    "remaining_sec": current_info["remaining_steps"] * env.dt,
                    "steps": steps,
                    "duration_sec": steps * env.dt,
                    "success_by_timeout": success,
                    "completion_ratio": float(steps / remaining_steps),
                }
                episodes.append(episode_record)
                print(json.dumps({"label": run_spec.label, **episode_record}))

            if len(episodes) >= args.num_episodes:
                break

            for env_id in done_ids:
                current_infos[env_id] = current_episode_info(env, int(env_id))
                episode_steps[env_id] = 0

        return summarize_episodes(
            label=run_spec.label,
            load_run=run_spec.load_run,
            checkpoint=run_spec.checkpoint,
            env_dt=env.dt,
            episodes=episodes,
        )
    finally:
        destroy_env(env)


def main() -> None:
    args = parse_args()
    run_specs = [parse_run_spec(raw) for raw in args.run_spec]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for run_spec in run_specs:
        results.append(evaluate_run(args, run_spec))

    comparison = {
        "task": args.task,
        "motion_source": args.motion_source,
        "num_episodes": args.num_episodes,
        "link_pos_error_threshold": args.link_pos_error_threshold,
        "start_mode": args.start_mode,
        "start_offset_min": args.start_offset_min,
        "start_offset_max": args.start_offset_max,
        "results": results,
    }

    safe_labels = "_vs_".join(spec.label for spec in run_specs)
    if args.start_mode == "start_window":
        threshold_str = (
            f"_thr{str(args.link_pos_error_threshold).replace('.', 'p')}"
            if args.link_pos_error_threshold is not None
            else ""
        )
        suffix = f"{args.start_mode}_{args.start_offset_min}_{args.start_offset_max}{threshold_str}_{args.num_episodes}eps"
    else:
        threshold_str = (
            f"_thr{str(args.link_pos_error_threshold).replace('.', 'p')}"
            if args.link_pos_error_threshold is not None
            else ""
        )
        suffix = f"{args.start_mode}{threshold_str}_{args.num_episodes}eps"
    output_path = output_dir / f"{safe_labels}_{suffix}.json"
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
