#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "simulation" / "videomimic_gym" / "resources" / "data_config" / "human_motion_list_123_motions.yaml"
DEFAULT_CAPTURE_ROOT = REPO_ROOT / "simulation" / "data" / "videomimic_captures"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "simulation" / "data" / "videomimic_captures_mp4"
DEFAULT_LOG_ROOT = REPO_ROOT / "simulation" / "logs" / "render_videomimic_captures_123"
DEFAULT_RENDER_SCRIPT = REPO_ROOT / "simulation" / "render_videomimic_captures_mp4.py"
DEFAULT_CONDA_PREFIX = Path("/home/nas4_user/kyungminlee/anaconda3/envs/OpenHL")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust batch renderer for the official 123 VideoMimic capture clips.")
    parser.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--render-script", type=Path, default=DEFAULT_RENDER_SCRIPT)
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU index. The child process will see it as cuda:0 via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry", type=int, default=1, help="Extra attempts after the first try if the output is invalid.")
    return parser.parse_args()


def resolve_capture_dir_name(requested_name: str, available_dirs: list[str]) -> str:
    if requested_name in available_dirs:
        return requested_name

    matches = [name for name in available_dirs if requested_name in name]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(f"No capture directory matched YAML entry: {requested_name}")
    raise RuntimeError(f"Ambiguous capture directory for {requested_name}: {matches[:5]}")


def load_official_clip_names(yaml_path: Path, capture_root: Path) -> list[str]:
    rows = yaml.safe_load(yaml_path.read_text())
    available_dirs = sorted(path.name for path in capture_root.iterdir() if path.is_dir())
    clip_names = [
        resolve_capture_dir_name(Path(row["folder_path"]).name, available_dirs)
        for row in rows
    ]
    if len(clip_names) != len(set(clip_names)):
        raise RuntimeError("Resolved capture directory names are not unique.")
    return clip_names


def shard_clip_names(clip_names: list[str], shard_index: int, num_shards: int) -> list[str]:
    return [clip for idx, clip in enumerate(clip_names) if idx % num_shards == shard_index]


def is_valid_mp4(path: Path) -> tuple[bool, dict]:
    if not path.exists():
        return False, {"reason": "missing"}
    if path.stat().st_size <= 0:
        return False, {"reason": "empty"}

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False, {"reason": "opencv_open_failed"}
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        ok, _ = capture.read()
        if not ok or frames <= 0 or width <= 0 or height <= 0:
            return False, {
                "reason": "invalid_stream",
                "frames": frames,
                "width": width,
                "height": height,
                "fps": fps,
            }
        return True, {
            "reason": "ok",
            "frames": frames,
            "width": width,
            "height": height,
            "fps": fps,
            "size_bytes": path.stat().st_size,
        }
    finally:
        capture.release()


def build_child_env(gpu: int, conda_prefix: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TORCH_EXTENSIONS_DIR"] = f"/tmp/torch_extensions_videomimic_render_g{gpu}"
    env["PYTHONUNBUFFERED"] = "1"

    conda_bin = conda_prefix / "bin"
    conda_lib = conda_prefix / "lib"
    env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{conda_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def run_one_clip(
    clip_name: str,
    args: argparse.Namespace,
    env: dict[str, str],
    clip_log_dir: Path,
) -> dict:
    output_path = args.output_root / f"{clip_name}.mp4"
    clip_log_path = clip_log_dir / f"{clip_name}.log"
    clip_log_dir.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        valid, meta = is_valid_mp4(output_path)
        if valid:
            return {
                "clip": clip_name,
                "status": "skipped_existing",
                "output_path": str(output_path),
                "log_path": str(clip_log_path),
                "returncode": 0,
                "validation": meta,
            }
        output_path.unlink()

    result = None
    max_attempts = 1 + max(0, args.retry)
    for attempt in range(1, max_attempts + 1):
        if output_path.exists():
            output_path.unlink()
        cmd = [
            sys.executable,
            str(args.render_script),
            "--data-root",
            str(args.capture_root),
            "--output-root",
            str(args.output_root),
            "--input-format",
            "capture_dirs",
            "--clip",
            clip_name,
            "--overwrite",
        ]
        started = time.time()
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        clip_log_path.write_text(proc.stdout)
        valid, meta = is_valid_mp4(output_path)
        result = {
            "clip": clip_name,
            "status": "success" if valid else "failed",
            "output_path": str(output_path),
            "log_path": str(clip_log_path),
            "returncode": proc.returncode,
            "validation": meta,
            "attempt": attempt,
            "elapsed_sec": round(time.time() - started, 2),
        }
        if valid:
            break
    return result


def main() -> None:
    args = parse_args()
    args.yaml = args.yaml.resolve()
    args.capture_root = args.capture_root.resolve()
    args.output_root = args.output_root.resolve()
    args.log_root = args.log_root.resolve()
    args.render_script = args.render_script.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)

    clip_names = load_official_clip_names(args.yaml, args.capture_root)
    shard_clips = shard_clip_names(clip_names, args.shard_index, args.num_shards)

    shard_log_dir = args.log_root / f"gpu{args.gpu}_shard{args.shard_index:02d}"
    shard_log_dir.mkdir(parents=True, exist_ok=True)
    env = build_child_env(gpu=args.gpu, conda_prefix=DEFAULT_CONDA_PREFIX)

    summary: list[dict] = []
    total = len(shard_clips)
    for idx, clip_name in enumerate(shard_clips, start=1):
        print(f"[gpu {args.gpu}] [{idx}/{total}] {clip_name}")
        sys.stdout.flush()
        result = run_one_clip(clip_name=clip_name, args=args, env=env, clip_log_dir=shard_log_dir)
        summary.append(result)
        print(json.dumps(result, ensure_ascii=True))
        sys.stdout.flush()
        summary_path = shard_log_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

    success_count = sum(1 for row in summary if row["status"] in {"success", "skipped_existing"})
    failed = [row["clip"] for row in summary if row["status"] == "failed"]
    final = {
        "gpu": args.gpu,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "total": total,
        "successful_or_existing": success_count,
        "failed_count": len(failed),
        "failed_clips": failed,
    }
    print(json.dumps(final, indent=2))
    (shard_log_dir / "final_status.json").write_text(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
