from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


def resolve_ffmpeg_executable() -> str | None:
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is not None:
        return ffmpeg_exe

    if imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    return None


class CompatibleVideoWriter:
    """Write MP4 files in preview-friendly H.264 when ffmpeg is available."""

    def __init__(
        self,
        output_path,
        fps: float,
        frame_size: tuple[int, int],
        input_color: str = "bgr",
        prefer_h264: bool = True,
    ) -> None:
        self.output_path = str(output_path)
        self.fps = float(fps)
        self.width, self.height = frame_size
        self.input_color = input_color.lower()
        self.backend = ""
        self._mode = ""
        self._writer = None

        if self.input_color not in {"bgr", "rgb"}:
            raise ValueError(f"Unsupported input_color: {input_color}")

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        if prefer_h264:
            ffmpeg_exe = resolve_ffmpeg_executable()
            if ffmpeg_exe is not None:
                input_pix_fmt = "bgr24" if self.input_color == "bgr" else "rgb24"
                ffmpeg_cmd = [
                    ffmpeg_exe,
                    "-y",
                    "-f",
                    "rawvideo",
                    "-vcodec",
                    "rawvideo",
                    "-pix_fmt",
                    input_pix_fmt,
                    "-s",
                    f"{self.width}x{self.height}",
                    "-r",
                    str(self.fps),
                    "-i",
                    "-",
                    "-an",
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
                    self.output_path,
                ]
                try:
                    self._writer = subprocess.Popen(
                        ffmpeg_cmd,
                        stdin=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    self._mode = "ffmpeg"
                    self.backend = "ffmpeg/libx264"
                    return
                except Exception:
                    self._writer = None

        writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width, self.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {self.output_path}")
        self._writer = writer
        self._mode = "opencv"
        self.backend = "opencv/mp4v"

    def _ffmpeg_error_message(self) -> str:
        if self._writer is None or self._writer.stderr is None:
            return "Unknown ffmpeg error"
        stderr_output = self._writer.stderr.read().decode("utf-8", errors="replace").strip()
        return stderr_output or "Unknown ffmpeg error"

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("Video writer is closed")

        if self._mode == "ffmpeg":
            try:
                self._writer.stdin.write(np.ascontiguousarray(frame).tobytes())
            except BrokenPipeError as exc:
                raise RuntimeError(f"ffmpeg encoding failed: {self._ffmpeg_error_message()}") from exc
            return

        if self.input_color == "rgb":
            self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            self._writer.write(frame)

    def close(self) -> None:
        if self._writer is None:
            return

        if self._mode == "ffmpeg":
            if self._writer.stdin is not None:
                self._writer.stdin.close()
            return_code = self._writer.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg encoding failed: {self._ffmpeg_error_message()}")
        else:
            self._writer.release()

        self._writer = None

    def release(self) -> None:
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self.close()
        return False
