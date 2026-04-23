# Stage0 Result Transfer Guide (H200 -> RTX3090)

작성일: 2026-03-07
기준 로그: `/home/nas5/kyungminlee/VideoMimic/real2sim/stage0_yoon_videos_20260307_070201.log`

## 1) Stage0 처리 완료 비디오

아래 12개 비디오에 대해 stage0가 완료됨:

- `pg_b1`
- `pg_f1`
- `pg_f2`
- `platform_b1`
- `platform_b2`
- `platform_f1`
- `platform_f2`
- `stairs`
- `wall-kicking`
- `ws_d1`
- `ws_u1`
- `ws_ub1`

## 2) H200(stage0 실행 서버) 산출물 위치

루트:

- `/home/nas5/kyungminlee/VideoMimic/real2sim/demo_data`

비디오별 산출물(각 `<VIDEO>/cam01`):

- `input_images` (프레임)
- `input_masks` (SAM2 마스크/JSON)
- `input_2d_poses` (ViTPose 결과)
- `input_3d_meshes` (ViMo SMPL 결과)
- `input_contacts` (BSTRO contact 결과)

예시:

- `/home/nas5/kyungminlee/VideoMimic/real2sim/demo_data/input_images/stairs/cam01`
- `/home/nas5/kyungminlee/VideoMimic/real2sim/demo_data/input_masks/stairs/cam01`

## 3) 3090에서 Stage1을 위해 최소로 받아야 할 것

`megasam_reconstruction.py --gsam2` 기준 최소 필요:

- `input_images/<VIDEO>/cam01`
- `input_masks/<VIDEO>/cam01`

즉, Stage1만 이어서 할 때는 `input_images`, `input_masks` 두 폴더만 받아도 됨.

## 4) scp 명령 (3090 서버에서 실행)

아래 명령은 **RTX3090 서버(143.248.159.149:8022)에 로그인한 상태**에서 실행.
접속 정책상 3090 -> H200(59.29.246.31) 방향만 허용됨.

```bash
VIDEOS="pg_b1 pg_f1 pg_f2 platform_b1 platform_b2 platform_f1 platform_f2 stairs wall-kicking ws_d1 ws_u1 ws_ub1"
DEST=/home/nas4_user/kyungminlee/work/VideoMimic/real2sim/demo_data
SRC_BASE=/home/nas5/kyungminlee/VideoMimic/real2sim/demo_data

mkdir -p "$DEST/input_images" "$DEST/input_masks"

for v in $VIDEOS; do
  scp -r "kyungminlee@59.29.246.31:$SRC_BASE/input_images/$v" "$DEST/input_images/"
  scp -r "kyungminlee@59.29.246.31:$SRC_BASE/input_masks/$v" "$DEST/input_masks/"
done
```

## 5) Stage2+까지 고려해 전체 stage0 산출물을 받을 때

```bash
VIDEOS="pg_b1 pg_f1 pg_f2 platform_b1 platform_b2 platform_f1 platform_f2 stairs wall-kicking ws_d1 ws_u1 ws_ub1"
DEST=/home/nas4_user/kyungminlee/work/VideoMimic/real2sim/demo_data
SRC_BASE=/home/nas5/kyungminlee/VideoMimic/real2sim/demo_data

mkdir -p "$DEST/input_images" "$DEST/input_masks" "$DEST/input_2d_poses" "$DEST/input_3d_meshes" "$DEST/input_contacts"

for v in $VIDEOS; do
  scp -r "kyungminlee@59.29.246.31:$SRC_BASE/input_images/$v" "$DEST/input_images/"
  scp -r "kyungminlee@59.29.246.31:$SRC_BASE/input_masks/$v" "$DEST/input_masks/"
  scp -r "kyungminlee@59.29.246.31:$SRC_BASE/input_2d_poses/$v" "$DEST/input_2d_poses/"
  scp -r "kyungminlee@59.29.246.31:$SRC_BASE/input_3d_meshes/$v" "$DEST/input_3d_meshes/"
  scp -r "kyungminlee@59.29.246.31:$SRC_BASE/input_contacts/$v" "$DEST/input_contacts/"
done
```

## 6) 전송 후 간단 검증 (3090에서 실행)

```bash
for v in pg_b1 pg_f1 pg_f2 platform_b1 platform_b2 platform_f1 platform_f2 stairs wall-kicking ws_d1 ws_u1 ws_ub1; do
  i=$(find "/home/nas4_user/kyungminlee/work/VideoMimic/real2sim/demo_data/input_images/$v/cam01" -maxdepth 1 -name '*.jpg' | wc -l)
  m=$(find "/home/nas4_user/kyungminlee/work/VideoMimic/real2sim/demo_data/input_masks/$v/cam01/json_data" -maxdepth 1 -name 'mask_*.json' | wc -l)
  echo "$v images=$i masks=$m"
done
```
