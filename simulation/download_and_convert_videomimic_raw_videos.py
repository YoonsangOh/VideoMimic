from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_preview_compat import CompatibleVideoWriter


FILE_ID = "1lQWmxebQX9Yu_KBX_tpyg3CuP61hZrvP"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            # Force central directory parsing and basic member access.
            zf.infolist()
        return True
    except zipfile.BadZipFile:
        return False


def cleanup_temp_parts(download_dir: Path) -> None:
    for part in download_dir.glob("*.part"):
        try:
            part.unlink()
        except FileNotFoundError:
            pass


def download_until_valid(zip_path: Path, max_attempts: int, sleep_seconds: float) -> None:
    for attempt in range(1, max_attempts + 1):
        if is_valid_zip(zip_path):
            print(f"zip validated on attempt {attempt - 1 if attempt > 1 else 0}")
            return

        cmd = ["gdown", FILE_ID, "--output", str(zip_path)]
        if zip_path.exists():
            cmd.append("--continue")

        print(f"download attempt {attempt}/{max_attempts}: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)

        if is_valid_zip(zip_path):
            print(f"zip validated after attempt {attempt}")
            cleanup_temp_parts(zip_path.parent)
            return

        print(f"zip still invalid after attempt {attempt}")
        if attempt < max_attempts:
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to obtain a valid zip after {max_attempts} attempts: {zip_path}")


def find_frame_dirs(root: Path) -> list[Path]:
    frame_dirs: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        image_names = sorted(
            name for name in filenames if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if image_names:
            frame_dirs.append(Path(dirpath))
    return sorted(frame_dirs)


def natural_key(path: Path) -> list[object]:
    import re

    parts = re.split(r"(\d+)", path.name)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


def convert_dir(frame_dir: Path, input_root: Path, output_root: Path, fps: float) -> Path:
    image_paths = sorted(
        [p for p in frame_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=natural_key,
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {frame_dir}")

    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {image_paths[0]}")
    height, width = first.shape[:2]

    rel = frame_dir.relative_to(input_root)
    stem = "__".join(rel.parts)
    output_path = output_root / f"{stem}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = CompatibleVideoWriter(output_path, fps, (width, height), input_color="bgr")

    try:
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {image_path}")
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-dir", type=Path, default=Path("simulation/data/videomimic_raw_videos_jpeg"))
    parser.add_argument("--output-dir", type=Path, default=Path("simulation/data/videomimic_raw_videos_mp4"))
    parser.add_argument("--zip-name", default="raw_videos_jpeg.zip")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--max-download-attempts", type=int, default=10)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    args = parser.parse_args()

    download_dir = (REPO_ROOT / args.download_dir).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = download_dir / args.zip_name

    if not args.skip_download and not any(download_dir.iterdir()):
        pass

    if not args.skip_download:
        download_until_valid(
            zip_path,
            max_attempts=args.max_download_attempts,
            sleep_seconds=args.retry_sleep_seconds,
        )
        run(["unzip", "-o", str(zip_path)], cwd=download_dir)

    frame_dirs = find_frame_dirs(download_dir)
    converted = 0
    skipped = 0

    for frame_dir in frame_dirs:
        try:
            rel = frame_dir.relative_to(download_dir)
        except ValueError:
            continue
        if rel.parts and rel.parts[0].startswith("."):
            continue
        output_path = output_dir / f"{'__'.join(rel.parts)}.mp4"
        if output_path.exists() and output_path.stat().st_size > 0:
            skipped += 1
            continue
        convert_dir(frame_dir, download_dir, output_dir, args.fps)
        converted += 1
        print(f"converted: {frame_dir} -> {output_path}")

    print(f"done: converted={converted} skipped={skipped} frame_dirs={len(frame_dirs)}")


if __name__ == "__main__":
    main()
