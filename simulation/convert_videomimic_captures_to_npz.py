#!/usr/bin/env python3
"""Bundle each VideoMimic capture folder into a single NPZ file.

Input per clip:
  - retarget_poses_g1.h5
  - background_mesh.obj

Output per clip:
  - <clip_name>.npz in simulation/data/videomimic_captures_npz/

The NPZ stores both the motion arrays and the terrain mesh so downstream code
can load a single file per clip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
CAPTURES_ROOT = PROJECT_ROOT / "data" / "videomimic_captures"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "videomimic_captures_npz"


def _decode_name(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _get_h5_attr(attrs: h5py.AttributeManager, *names: str, default: object | None = None) -> object | None:
    for name in names:
        if name in attrs:
            return attrs[name]
    return default


def load_h5_motion(h5_path: Path) -> dict[str, np.ndarray | float]:
    with h5py.File(h5_path, "r") as handle:
        joint_names_raw = _get_h5_attr(handle.attrs, "joint_names", "/joint_names", default=[])
        link_names_raw = _get_h5_attr(handle.attrs, "link_names", "/link_names", default=[])
        fps = float(_get_h5_attr(handle.attrs, "fps", "/fps", default=30.0))

        motion = {
            "fps": np.array([fps], dtype=np.float32),
            "joint_names": np.array([_decode_name(name) for name in joint_names_raw], dtype=object),
            "link_names": np.array([_decode_name(name) for name in link_names_raw], dtype=object),
            "root_pos": handle["root_pos"][:].astype(np.float32),
            "root_quat": handle["root_quat"][:].astype(np.float32),
            "joints": handle["joints"][:].astype(np.float32),
            "link_pos": handle["link_pos"][:].astype(np.float32),
            "link_quat": handle["link_quat"][:].astype(np.float32),
        }

        if "contacts" in handle:
            for contact_name, dataset in handle["contacts"].items():
                motion[f"contacts_{contact_name}"] = dataset[:].astype(np.float32)

    return motion


def load_obj_mesh(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []

    with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue

            if not line.startswith("f "):
                continue

            parts = line.strip().split()[1:]
            face: list[int] = []
            for part in parts:
                vertex_token = part.split("/")[0]
                vertex_idx = int(vertex_token)
                if vertex_idx < 0:
                    vertex_idx = len(vertices) + vertex_idx
                else:
                    vertex_idx -= 1
                face.append(vertex_idx)

            # Fan triangulation keeps the stored mesh simple and numeric.
            for idx in range(1, len(face) - 1):
                triangles.append([face[0], face[idx], face[idx + 1]])

    return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def iter_capture_folders(root: Path) -> list[Path]:
    folders: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "retarget_poses_g1.h5").is_file() and (path / "background_mesh.obj").is_file():
            folders.append(path)
    return folders


def convert_one_folder(folder: Path, output_dir: Path) -> Path:
    h5_path = folder / "retarget_poses_g1.h5"
    obj_path = folder / "background_mesh.obj"

    motion = load_h5_motion(h5_path)
    mesh_vertices, mesh_faces = load_obj_mesh(obj_path)

    output_path = output_dir / f"{folder.name}.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        clip_name=np.array([folder.name], dtype=object),
        source_folder=np.array([str(folder)], dtype=object),
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        **motion,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VideoMimic capture folders to single-file NPZ bundles.")
    parser.add_argument("--input-root", type=Path, default=CAPTURES_ROOT, help="Folder containing per-clip capture folders.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="Folder to write NPZ bundles into.")
    args = parser.parse_args()

    folders = iter_capture_folders(args.input_root)
    print(f"Found {len(folders)} capture folders in {args.input_root}")

    failures: list[tuple[str, str]] = []
    for idx, folder in enumerate(folders, start=1):
        try:
            output_path = convert_one_folder(folder, args.output_root)
            print(f"[{idx}/{len(folders)}] {folder.name} -> {output_path.name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((folder.name, str(exc)))
            print(f"[{idx}/{len(folders)}] FAILED {folder.name}: {exc}")

    print(f"Done. Wrote {len(folders) - len(failures)} NPZ files to {args.output_root}")
    if failures:
        print("Failures:")
        for name, message in failures:
            print(f"  - {name}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
