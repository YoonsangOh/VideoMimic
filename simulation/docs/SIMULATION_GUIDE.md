# VideoMimic Simulation 가이드

## 개요

VideoMimic 시뮬레이션은 Real2Sim 파이프라인에서 생성된 scene과 motion 데이터를 사용하여 humanoid robot의 locomotion을 학습합니다.

## 데이터 구조 및 형식

### 1. Motion 데이터 형식

시뮬레이션은 **PKL** 또는 **H5** 형식의 motion 데이터를 로드합니다.

#### PKL 파일 구조:
```python
{
    "joint_names": ["left_hip_pitch_joint", "left_hip_roll_joint", ...],  # 리스트
    "joints": np.array((timeseries_length, num_joints)),  # 관절 위치 (x, y, z)
    "root_quat": np.array((timeseries_length, 4)),  # 전역 좌표계 쿼터니언 (x, y, z, w)
    "root_pos": np.array((timeseries_length, 3)),  # 전역 좌표계 위치 (x, y, z)
    "link_names": ["torso_link", "left_foot", ...],  # 리스트
    "link_pos": np.array((timeseries_length, len(link_names), 3)),  # root 기준 로컬 위치
    "link_quat": np.array((timeseries_length, len(link_names), 4)),  # root 기준 로컬 쿼터니언
    "contacts": {  # 선택적
        "left_foot": np.array(timeseries_length, dtype=bool),
        "right_foot": np.array(timeseries_length, dtype=bool)
    },
    "fps": 30.0  # 프레임 레이트
}
```

#### H5 파일 구조:
H5 파일은 동일한 구조를 가집니다:
- `root_pos`, `root_quat`, `joints`, `link_pos`, `link_quat`는 데이터셋으로 저장
- `joint_names`, `link_names`, `fps`는 attributes로 저장 (`/joint_names`, `/link_names`, `/fps`)

### 2. Scene 데이터 형식

Scene 데이터는 **OBJ 형식의 메쉬 파일**입니다:
- 파일명: `background_mesh.obj` (기본값, 설정으로 변경 가능)
- 3D 메쉬로 terrain을 표현
- trimesh로 로드 가능한 표준 OBJ 형식

### 3. 데이터 폴더 구조

각 비디오/시퀀스는 하나의 폴더에 motion과 scene 파일을 함께 저장해야 합니다:

```
data/
  videomimic_captures/  # 또는 사용자 지정 경로
    video1_folder/
      retarget_poses_g1.h5  # 또는 .pkl
      background_mesh.obj
    video2_folder/
      retarget_poses_g1.h5
      background_mesh.obj
    ...
```

## 시뮬레이션 동작 방식

### 1. 데이터 로딩

시뮬레이션은 다음 순서로 데이터를 로드합니다:

1. **AMASS 데이터** (선택적)
   - `use_amass=True`일 때 로드
   - PKL 파일 경로를 glob 패턴으로 지정
   - Random terrain과 페어링

2. **Human Video 데이터** (선택적)
   - `use_human_videos=True`일 때 로드
   - YAML 파일 또는 단일 폴더명으로 지정
   - 각 폴더에서 motion 파일과 terrain mesh를 함께 로드

### 2. YAML 설정 파일

여러 motion을 사용할 때는 YAML 파일로 관리:

```yaml
- folder_path: "video_folder_name"
  teacher_checkpoint_run_name: "checkpoint_name"  # 선택적
  human_video_data_pattern: "retarget_poses_g1.h5"  # 기본값 사용 가능
  human_video_terrain_pattern: "background_mesh.obj"  # 기본값 사용 가능
  default_data_fps_override: 60  # 선택적
```

### 3. Terrain 처리

- 여러 terrain mesh를 하나의 큰 mesh로 연결
- Grid 패턴으로 배치 (n_rows 설정)
- 각 환경은 다른 terrain 위치에 배치되어 겹침 방지

### 4. 학습 단계

시뮬레이션 학습은 4단계로 진행됩니다:

1. **Stage 1: MoCap Pre-training**
   - Motion capture 데이터로 사전 학습
   - 스크립트: `train_stage_1_mcpt.sh`

2. **Stage 2: Terrain Tracking**
   - Terrain 위에서 tracking 학습
   - 스크립트: `train_stage_2_terrain_rl.sh`

3. **Stage 3: Distillation**
   - Teacher policy로부터 distillation
   - 스크립트: `train_stage_3_distillation.sh`

4. **Stage 4: RL Finetuning**
   - 최종 RL fine-tuning
   - 스크립트: `train_stage_4_rl_finetune.sh`

## 데이터 준비 방법

### Real2Sim 파이프라인에서 생성된 데이터 사용

1. **Motion 데이터 확인**
   - Real2Sim에서 생성된 retargeting 결과 확인
   - PKL 또는 H5 형식인지 확인
   - 필요한 필드가 모두 있는지 확인 (joint_names, link_names, fps 등)

2. **Scene 데이터 확인**
   - Background mesh 파일 확인
   - OBJ 형식인지 확인

3. **폴더 구조 생성**
   ```
   simulation/data/videomimic_captures/
     your_video1/
       retarget_poses_g1.h5  # 또는 .pkl
       background_mesh.obj
     your_video2/
       retarget_poses_g1.h5
       background_mesh.obj
   ```

4. **YAML 파일 생성** (선택적, 여러 motion 사용 시)
   ```yaml
   - folder_path: "your_video1"
     human_video_data_pattern: "retarget_poses_g1.h5"
     human_video_terrain_pattern: "background_mesh.obj"
   - folder_path: "your_video2"
     human_video_data_pattern: "retarget_poses_g1.h5"
     human_video_terrain_pattern: "background_mesh.obj"
   ```

## 주요 설정 파라미터

### 환경 설정 (--env.*)
- `deepmimic.use_human_videos`: Human video 데이터 사용 여부
- `deepmimic.human_video_folders`: Human video 폴더 리스트
- `deepmimic.human_motion_source`: YAML 파일 경로 또는 폴더명
- `deepmimic.data_root`: 데이터 루트 경로
- `terrain.n_rows`: Terrain grid 행 수

### 학습 설정 (--train.*)
- `algorithm.learning_rate`: 학습률
- `runner.save_interval`: 체크포인트 저장 간격

## 실행 예시

### 단일 motion 학습
```bash
torchrun --nproc-per-node 2 legged_gym/scripts/train.py \
  --multi_gpu \
  --task=g1_deepmimic \
  --headless \
  --env.deepmimic.use_amass=False \
  --env.deepmimic.use_human_videos=True \
  --env.deepmimic.human_motion_source="your_video_folder" \
  --env.terrain.n_rows=6
```

### YAML 파일로 여러 motion 학습
```bash
torchrun --nproc-per-node 2 legged_gym/scripts/train.py \
  --multi_gpu \
  --task=g1_deepmimic \
  --headless \
  --env.deepmimic.use_amass=False \
  --env.deepmimic.use_human_videos=True \
  --env.deepmimic.human_motion_source="resources/data_config/your_motions.yaml" \
  --env.terrain.n_rows=6
```

## 주의사항

1. **데이터 형식**: PKL/H5 파일의 쿼터니언 형식은 `xyzw`여야 합니다
2. **좌표계**:
   - `root_pos`, `root_quat`: 전역 좌표계
   - `link_pos`, `link_quat`: root 기준 로컬 좌표계
3. **FPS**: Motion 데이터의 FPS가 시뮬레이션 dt와 일치하거나 upsample_data 옵션 사용
4. **Terrain 정렬**: Motion의 root_pos와 terrain mesh가 같은 좌표계에 있어야 합니다
