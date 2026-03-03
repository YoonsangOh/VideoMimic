# VideoMimic Agent Notes

## Scope
- Repository root: `/home/nas5/kyungminlee/VideoMimic`
- Current primary focus: `real2sim` pipeline
- Future work may include `simulation`, `sim2real`, and other top-level folders.

## Real2Sim Pipeline Understanding
- Stage 0 (preprocessing): `vm1rs` (Python 3.12, CUDA 12.4+)
- Stage 1 (MegaSam reconstruction): `vm1recon` (Python 3.10, CUDA 11.8)
- Stage 2 (MegaHunter optimization): `vm1rs` (Python 3.12, CUDA 12.4+)
- Stage 3 (GeoCalib + NKSR meshification): `vm1recon` (Python 3.10, CUDA 11.8)
- Stage 4 (robot motion retargeting): `vm1rs` (Python 3.12, CUDA 12.4+)

## Infrastructure Constraints (Important)
- Two conda environments are required due to dependency/CUDA conflicts:
  - `vm1rs` for most stages
  - `vm1recon` for MegaSam/NKSR-related stages
- CUDA compatibility differs by server/GPU, so full pipeline cannot always run end-to-end on one machine.
- Network direction is asymmetric:
  - `h200 (59.29.246.31) -> rtx3090 (143.248.159.149:8022)`: blocked
  - `rtx3090 (143.248.159.149:8022) -> h200 (59.29.246.31)`: allowed
- Operational implication:
  - Run stages on the appropriate server.
  - Move intermediate artifacts manually from `rtx3090` side by pulling from `h200` and then placing/sending as needed.
  - No strict end-to-end automation script is required; ad-hoc transfer steps are expected.

## Git Branch Snapshot (at time of recording)
- Local branches: `main`, `moge`
- Remote: `origin/main`
- `main` and `moge` point to the same HEAD commit at snapshot time.

