#!/usr/bin/env python3
"""Re-encode MP4 files to preview-friendly H.264/yuv420p."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from video_preview_compat import resolve_ffmpeg_executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="MP4 files or directories to process")
    parser.add_argument("--no-recursive", action="store_true", help="Do not recurse into directories")
    parser.add_argument("--replace", action="store_true", help="Replace original files in place")
    parser.add_argument("--suffix", default="_h264", help="Suffix for non-replaced outputs")
    parser.add_argument("--force", action="store_true", help="Re-encode even if already H.264")
    return parser.parse_args()


def iter_mp4_paths(targets: list[Path], recursive: bool) -> list[Path]:
    found: list[Path] = []
    pattern = "**/*.mp4" if recursive else "*.mp4"
    for target in targets:
        if target.is_file():
            if target.suffix.lower() == ".mp4":
                found.append(target.resolve())
            continue
        if target.is_dir():
            found.extend(path.resolve() for path in sorted(target.glob(pattern)))
    return found


def probe_video_codec(ffmpeg_exe: str, path: Path) -> str | None:
    result = subprocess.run(
        [ffmpeg_exe, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
    )
    match = re.search(r"Video:\s*([^ ,]+)", result.stderr)
    if match is None:
        return None
    return match.group(1)


def build_output_path(path: Path, replace: bool, suffix: str) -> Path:
    if replace:
        return path.with_name(f"{path.stem}.compat_tmp.mp4")
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def transcode_to_h264(ffmpeg_exe: str, src: Path, dst: Path) -> None:
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        "-preset",
        "medium",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg re-encode failed")


def main() -> None:
    args = parse_args()
    ffmpeg_exe = resolve_ffmpeg_executable()
    if ffmpeg_exe is None:
        raise RuntimeError("No ffmpeg executable available. Install ffmpeg or imageio-ffmpeg.")

    mp4_paths = iter_mp4_paths(args.paths, recursive=not args.no_recursive)
    if not mp4_paths:
        raise FileNotFoundError("No MP4 files matched the requested paths.")

    converted = 0
    skipped = 0
    for path in mp4_paths:
        codec = probe_video_codec(ffmpeg_exe, path)
        if codec == "h264" and not args.force:
            print(f"skip  {path}  codec={codec}")
            skipped += 1
            continue

        output_path = build_output_path(path, replace=args.replace, suffix=args.suffix)
        print(f"encode {path}  codec={codec or 'unknown'} -> {output_path}")
        transcode_to_h264(ffmpeg_exe, path, output_path)

        if args.replace:
            output_path.replace(path)
            print(f"done   {path}")
        else:
            print(f"done   {output_path}")
        converted += 1

    print(f"finished: converted={converted}, skipped={skipped}")


if __name__ == "__main__":
    main()
