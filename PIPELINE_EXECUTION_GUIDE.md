# VideoMimic Real2Sim Pipeline 실행 가이드

이 문서는 VideoMimic Real2Sim 파이프라인을 처음부터 끝까지 실행하는 방법을 단계별로 정리한 가이드입니다.

## 목차

1. [사전 준비](#사전-준비)
2. [Stage 0: 전처리 (Preprocessing)](#stage-0-전처리-preprocessing)
3. [Stage 1: 장면 재구성 (Scene Reconstruction)](#stage-1-장면-재구성-scene-reconstruction)
4. [Stage 2: 최적화 (Optimization)](#stage-2-최적화-optimization)
5. [Stage 3: 후처리 (Postprocessing)](#stage-3-후처리-postprocessing)
6. [Stage 4: 로봇 모션 리타겟팅 (Robot Motion Retargeting)](#stage-4-로봇-모션-리타겟팅-robot-motion-retargeting)
7. [시각화 (Visualization)](#시각화-visualization)
8. [서버 간 데이터 전송](#서버-간-데이터-전송)

---

## 사전 준비

### 환경 설정

VideoMimic 파이프라인은 두 개의 서로 다른 Conda 환경이 필요합니다:

| 환경 이름 | Python | CUDA | 서버 | 용도 |
|---------|--------|------|------|------|
| `vm1rs` | 3.12 | 12.4+ | **H200** | 인간 전처리, 최적화, 리타겟팅 |
| `vm1recon` | 3.10 | 11.8 | **RTX 3090** | MegaSam, NKSR 메시화, GeoCalib |

**중요**: MegaSam과 NKSR은 CUDA 11.8이 필요하므로 RTX 3090 서버에서 실행해야 합니다.

### 데이터 구조 준비

비디오 파일을 다음과 같은 구조로 준비합니다:

```
demo_data/
├── input_images/
│   └── {video_name}/
│       └── cam01/
│           ├── 00001.jpg
│           ├── 00002.jpg
│           └── ...
└── input_contacts/
    └── {video_name}/
        └── cam01/
            ├── {frame_name}.pkl
            └── ...
```

---

## Stage 0: 전처리 (Preprocessing)

### 서버 및 환경
- **서버**: H200 (CUDA 12.4+)
- **환경**: `vm1rs`

### 목적
비디오에서 프레임 추출, 인간 탐지, 2D/3D 포즈 추정, 접촉 정보 생성

### 입력 (Input)
- 비디오 파일: `{video_name}.mp4` (또는 다른 형식)

### 실행 방법

#### 0.1 프레임 추출
```bash
conda activate vm1rs
python utilities/extract_frames_from_video.py \
    --video-path {video_name}.mp4 \
    --output-dir ./demo_data/input_images/{video_name}/cam01 \
    --start-frame 0 \
    --end-frame 300
```

#### 0.2 인간 탐지 및 마스크 생성 (Grounded-SAM-2)
```bash
conda activate vm1rs
python stage0_preprocessing/grounded_sam2_masks.py \
    --video-dir ./demo_data/input_images/{video_name}/cam01 \
    --text-prompt "person" \
    --output-dir ./demo_data/input_masks/{video_name}/cam01
```

#### 0.3 2D 포즈 추정 (ViTPose)
```bash
conda activate vm1rs
python stage0_preprocessing/vitpose_2d_poses.py \
    --video-dir ./demo_data/input_images/{video_name}/cam01 \
    --output-dir ./demo_data/input_2d_poses/{video_name}/cam01
```

#### 0.4 3D 메시 추정 (VIMO)
```bash
conda activate vm1rs
python stage0_preprocessing/vimo_3d_meshes.py \
    --video-dir ./demo_data/input_images/{video_name}/cam01 \
    --output-dir ./demo_data/input_3d_meshes/{video_name}/cam01 \
    --gender female  # 또는 male
```

#### 0.5 접촉 정보 생성 (BSTRO)
```bash
conda activate vm1rs
python stage0_preprocessing/bstro_contacts.py \
    --video-dir ./demo_data/input_images/{video_name}/cam01 \
    --smpl-dir ./demo_data/input_3d_meshes/{video_name}/cam01 \
    --output-dir ./demo_data/input_contacts/{video_name}/cam01
```

### 출력 (Output)
```
demo_data/
├── input_images/{video_name}/cam01/          # 추출된 프레임
├── input_masks/{video_name}/cam01/           # 마스크 데이터
├── input_2d_poses/{video_name}/cam01/        # 2D 포즈
├── input_3d_meshes/{video_name}/cam01/       # SMPL 파라미터
└── input_contacts/{video_name}/cam01/        # 접촉 정보
```

---

## Stage 1: 장면 재구성 (Scene Reconstruction)

### 서버 및 환경
- **서버**: RTX 3090 (CUDA 11.8)
- **환경**: `vm1recon`

### 목적
MegaSam을 사용하여 단일 카메라 비디오로부터 3D 장면 포인트 클라우드 재구성

### 입력 (Input)
- Stage 0 출력: `demo_data/input_images/{video_name}/cam01/`
- Stage 0 마스크: `demo_data/input_masks/{video_name}/cam01/`

### 실행 방법

```bash
conda activate vm1recon
python stage1_reconstruction/megasam_reconstruction.py \
    --video-dir ./demo_data/input_images/{video_name}/cam01 \
    --mask-dir ./demo_data/input_masks/{video_name}/cam01 \
    --output-dir ./demo_data/input_megasam/megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample} \
    --start-frame 0 \
    --end-frame 300 \
    --stride 2 \
    --gsam2
```

**파라미터 설명:**
- `--start-frame`: 시작 프레임 번호
- `--end-frame`: 종료 프레임 번호 (-1이면 끝까지)
- `--stride`: 프레임 간격 (예: 2면 매 2번째 프레임만 사용)
- `--gsam2`: GSAM2 마스크 사용 여부

### 출력 (Output)
```
demo_data/input_megasam/
└── megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample}.h5
```

**파일 내용:**
- `monst3r_ga_output`: MegaSam 재구성 결과
  - 각 프레임의 포인트 클라우드 (`pts3d`)
  - 깊이 맵 (`depths`)
  - RGB 이미지 (`rgbimg`)
  - 카메라 포즈 (`cam2world`)
  - 동적 마스크 (`dynamic_msk`)
  - 신뢰도 (`conf`)

---

## Stage 2: 최적화 (Optimization)

### 서버 및 환경
- **서버**: H200 (CUDA 12.4+)
- **환경**: `vm1rs`

### 목적
MegaHunter를 사용하여 인간 모션과 장면을 정렬하고 최적화

### 입력 (Input)
- Stage 1 출력: `megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample}.h5`
- Stage 0 마스크: `demo_data/input_masks/{video_name}/cam01/`
- Stage 0 2D 포즈: `demo_data/input_2d_poses/{video_name}/cam01/`
- Stage 0 3D 메시: `demo_data/input_3d_meshes/{video_name}/cam01/`

### 실행 방법

```bash
conda activate vm1rs
python stage2_optimization/megahunter_optimization.py \
    --world-env-path ./demo_data/input_megasam/megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample}.h5 \
    --bbox-dir ./demo_data/input_masks/{video_name}/cam01/json_data \
    --pose2d-dir ./demo_data/input_2d_poses/{video_name}/cam01 \
    --smpl-dir ./demo_data/input_3d_meshes/{video_name}/cam01 \
    --out-dir ./demo_data/output_smpl_and_points/megahunter_megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample} \
    --use-g1-shape  # G1 로봇 크기에 맞춰 스케일 조정
```

**파라미터 설명:**
- `--use-g1-shape`: G1 로봇 크기에 맞춰 세계 재구성 결과 스케일 조정
- `--multihuman`: 다중 인간 모드 활성화
- `--num-iterations`: 최적화 반복 횟수 (기본값: 300)

### 출력 (Output)
```
demo_data/output_smpl_and_points/
└── megahunter_megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample}.h5
```

**파일 내용:**
- `our_pred_world_cameras_and_structure`: 최적화된 세계 환경
- `our_pred_humans_smplx_params`: 최적화된 인간 SMPL 파라미터
  - `body_pose`: 신체 포즈
  - `betas`: Shape 파라미터
  - `global_orient`: 전역 회전
  - `root_transl`: 루트 위치

---

## Stage 3: 후처리 (Postprocessing)

### 서버 및 환경
- **서버**: RTX 3090 (CUDA 11.8)
- **환경**: `vm1recon`

### 목적
중력 보정 및 배경 메시 생성

### 입력 (Input)
- Stage 2 출력: `megahunter_megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample}.h5`

### 실행 방법

#### Stage 3.1: 중력 보정 (Gravity Calibration)

```bash
conda activate vm1recon
python stage3_postprocessing/postprocessing_pipeline.py \
    --megahunter-path ./demo_data/output_smpl_and_points/megahunter_megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample}.h5 \
    --out-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample} \
    --gender female \
    --is-megasam
```

**파라미터 설명:**
- `--gender`: SMPL 모델 성별 (`male` 또는 `female`)
- `--is-megasam`: MegaSam 재구성 결과 사용 여부
- `--multihuman`: 다중 인간 모드

**주의**: Stage 3.1은 성공하지만, Stage 3.2에서 NKSR 이슈가 발생할 수 있습니다. NKSR이 제대로 설치되지 않은 경우 `background_mesh.obj`가 생성되지 않을 수 있습니다.

#### Stage 3.2: 메시 생성 (Mesh Generation)

Stage 3.1과 함께 자동으로 실행되지만, NKSR 문제로 실패할 수 있습니다.

**NKSR이 정상 작동할 경우 생성되는 파일:**
- `background_mesh.obj`: 배경 메시 파일
- `background_less_filtered_colored_pointcloud.ply`: 덜 필터링된 포인트 클라우드
- `background_more_filtered_colored_pointcloud.ply`: 더 필터링된 포인트 클라우드

### 출력 (Output)

```
demo_data/output_calib_mesh/
└── megahunter_megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample}/
    ├── gravity_calibrated_keypoints.h5          # 중력 보정된 키포인트
    ├── gravity_calibrated_megahunter.h5         # 중력 보정된 메가헌터 데이터
    ├── background_mesh.obj                      # 배경 메시 (NKSR 성공 시)
    ├── background_less_filtered_colored_pointcloud.ply
    └── background_more_filtered_colored_pointcloud.ply
```

---

## Stage 4: 로봇 모션 리타겟팅 (Robot Motion Retargeting)

### 서버 및 환경
- **서버**: H200 (CUDA 12.4+)
- **환경**: `vm1rs`

### 목적
인간 모션을 G1 로봇의 관절 각도 시퀀스로 변환

### 입력 (Input)
- Stage 3 출력 디렉토리: `demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample}/`
  - `gravity_calibrated_keypoints.h5`
  - `gravity_calibrated_megahunter.h5`
  - `background_mesh.obj` (선택적, 있으면 더 좋음)
- Stage 0 접촉 정보: `demo_data/input_contacts/{video_name}/cam01/`

### 실행 방법

#### 기본 실행 (단일 인간)

```bash
conda activate vm1rs
python stage4_retargeting/robot_motion_retargeting.py \
    --src-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample} \
    --contact-dir ./demo_data/input_contacts/{video_name}/cam01 \
    --vis  # 시각화 옵션
```

#### 다중 인간 모드

```bash
conda activate vm1rs
python stage4_retargeting/robot_motion_retargeting.py \
    --src-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample} \
    --contact-dir ./demo_data/input_contacts/{video_name}/cam01 \
    --multihuman \
    --top-k 3 \
    --vis
```

#### Cost 가중치 조정

리타겟팅 결과가 이상할 경우 다음 가중치를 조정할 수 있습니다:

```bash
python stage4_retargeting/robot_motion_retargeting.py \
    --src-dir ... \
    --contact-dir ... \
    --local-pose-cost-weight 8.0 \
    --end-effector-cost-weight 5.0 \
    --global-pose-cost-weight 2.0 \
    --foot-skating-cost-weight 10.0 \
    --ground-contact-cost-weight 0.5 \
    --smoothness-cost-factor-weight 10.0 \
    --self-coll-factor-weight 1.0 \
    --world-coll-factor-weight 0.01 \
    --limit-cost-factor-weight 1000.0
```

**Cost 가중치 설명:**
- `--local-pose-cost-weight` (기본값: 8.0): 상대적 관절 위치 유사성
- `--end-effector-cost-weight` (기본값: 5.0): 발 위치 정확도
- `--global-pose-cost-weight` (기본값: 2.0): 전체 포즈 유사성
- `--foot-skating-cost-weight` (기본값: 10.0): 발 미끄러짐 방지
- `--ground-contact-cost-weight` (기본값: 0.5): 지면 접촉 정렬
- `--smoothness-cost-factor-weight` (기본값: 10.0): 모션 부드러움

### 출력 (Output)

```
demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample}/
└── retarget_poses_g1.h5  # 단일 인간 모드
    # 또는
└── retarget_poses_g1_multiperson.h5  # 다중 인간 모드
```

**파일 내용:**
- `joints`: (T, N) - 각 프레임의 로봇 관절 각도
- `root_pos`: (T, 3) - 각 프레임의 로봇 루트 위치
- `root_quat`: (T, 4) - 각 프레임의 로봇 루트 회전 (xyzw)
- `link_pos`: (T, M, 3) - 각 프레임의 각 링크 위치
- `link_quat`: (T, M, 4) - 각 프레임의 각 링크 회전
- `contacts`: 접촉 정보
  - `left_foot`: (T,) - 왼발 접촉 여부
  - `right_foot`: (T,) - 오른발 접촉 여부
- `fps`: 프레임레이트

---

## 시각화 (Visualization)

### 전체 결과 시각화

```bash
conda activate vm1rs
python visualization/complete_results_egoview_visualization.py \
    --postprocessed-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample} \
    --robot-name g1 \
    --bg-pc-downsample-factor 4 \
    --is-megasam
```

**파라미터 설명:**
- `--postprocessed-dir`: Stage 3 출력 디렉토리
- `--robot-name`: 로봇 이름 (`g1`)
- `--bg-pc-downsample-factor`: 배경 포인트 클라우드 다운샘플링 팩터
- `--is-megasam`: MegaSam 결과 사용 여부

**주의**: `--robot-name g1`은 Stage 4를 실행하지 않았으면 로봇 시각화에 영향을 주지 않습니다. SMPL 인간 모델은 여전히 시각화됩니다.

### 리타겟팅 결과 시각화

```bash
conda activate vm1rs
python visualization/retargeting_visualization.py \
    --postprocessed-dir ./demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample} \
    --robot-name g1 \
    --bg-pc-downsample-factor 4 \
    --confidence-threshold 0.0
```

---

## 서버 간 데이터 전송

### H200 → RTX 3090 (Stage 1, Stage 3 실행 전)

#### Stage 1 실행을 위한 데이터 전송

```bash
# H200 서버에서
scp -r demo_data/input_images/{video_name} user@rtx3090:/path/to/VideoMimic/real2sim/demo_data/input_images/
scp -r demo_data/input_masks/{video_name} user@rtx3090:/path/to/VideoMimic/real2sim/demo_data/input_masks/
```

또는 `rsync` 사용:

```bash
rsync -avz demo_data/input_images/{video_name} user@rtx3090:/path/to/VideoMimic/real2sim/demo_data/input_images/
rsync -avz demo_data/input_masks/{video_name} user@rtx3090:/path/to/VideoMimic/real2sim/demo_data/input_masks/
```

#### Stage 3 실행을 위한 데이터 전송

```bash
# H200 서버에서
scp demo_data/output_smpl_and_points/megahunter_megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample}.h5 \
    user@rtx3090:/path/to/VideoMimic/real2sim/demo_data/output_smpl_and_points/
```

### RTX 3090 → H200 (Stage 2, Stage 4 실행 전)

#### Stage 2 실행을 위한 데이터 전송

```bash
# RTX 3090 서버에서
scp demo_data/input_megasam/megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample}.h5 \
    user@h200:/path/to/VideoMimic/real2sim/demo_data/input_megasam/
```

#### Stage 4 실행을 위한 데이터 전송

```bash
# RTX 3090 서버에서
scp -r demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_{video_name}_cam01_frame_{start}_{end}_subsample_{subsample} \
    user@h200:/path/to/VideoMimic/real2sim/demo_data/output_calib_mesh/
```

---

## 전체 파이프라인 실행 스크립트

`process_video.sh` 스크립트를 사용하여 전체 파이프라인을 한 번에 실행할 수 있습니다:

```bash
cd /home/nas5/kyungminlee/VideoMimic/real2sim
./process_video.sh {video_name} {start_frame} {end_frame} {subsample_factor} g1 {human_height}
```

**예시:**
```bash
./process_video.sh ladder2 100 700 2 g1 1.8
```

**파라미터 설명:**
- `{video_name}`: 비디오 이름
- `{start_frame}`: 시작 프레임
- `{end_frame}`: 종료 프레임
- `{subsample_factor}`: 서브샘플링 팩터
- `g1`: 로봇 이름
- `{human_height}`: 인간 키 (미터 단위, 0이면 로봇 shape 사용, -1이면 추정된 shape 사용)

**주의**: 이 스크립트는 모든 stage를 순차적으로 실행하므로, 서버 간 데이터 전송이 필요합니다.

---

## 문제 해결 (Troubleshooting)

### NKSR 설치 문제

Stage 3.2에서 NKSR 관련 에러가 발생하는 경우:
- NKSR은 CUDA 11.8 환경에서만 작동합니다
- `vm1recon` 환경에서 NKSR이 제대로 설치되었는지 확인하세요
- NKSR 없이도 Stage 3.1은 완료되지만, `background_mesh.obj`가 생성되지 않습니다
- 시각화는 `background_mesh.obj`가 없으면 실패할 수 있습니다

### 메모리 부족

- MegaSam: ~24GB+ GPU 메모리 필요 (300 프레임 기준)
- `--end-frame`를 줄이거나 `--stride`를 늘려 프레임 수를 줄이세요

### Cost 가중치 조정

리타겟팅 결과가 이상한 경우:
1. `--foot-skating-cost-weight`, `--ground-contact-cost-weight`, `--world-coll-factor-weight`를 0.0으로 설정해보세요
2. 점진적으로 가중치를 조정하세요

---

## 참고 자료

- [Setup Guide](./real2sim/docs/setup.md): 환경 설정 가이드
- [Commands Guide](./real2sim/docs/commands.md): 상세 명령어 가이드
- [VideoMimic GitHub](https://github.com/hongsukchoi/VideoMimic): 공식 레포지토리

---

**작성일**: 2025년
**작성자**: VideoMimic 사용자 가이드






