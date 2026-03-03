# VideoMimic 시뮬레이션 구조 상세 설명

## 개요

VideoMimic의 시뮬레이션 부분은 영상에서 추출한 scene(배경 지형)과 motion(로봇 동작) 정보를 사용하여 이미 locomotion을 학습한 pretrained policy를 특정 영상의 motion과 scene에 맞게 finetuning합니다.

---

## 1. 데이터 로딩 및 초기화 과정

### 1.1 영상 데이터 로딩 (`get_replay_terrain_path`)

**위치**: `simulation/videomimic_gym/legged_gym/envs/g1/g1_deepmimic.py`

영상에서 추출한 데이터는 다음과 같이 로드됩니다:

1. **YAML 설정 파일 사용** (`resources/data_config/ladder2_motion.yaml` 등)
   - 각 영상 클립에 대한 정보:
     - `folder_path`: 데이터가 저장된 폴더 경로
     - `human_video_data_pattern`: 모션 데이터 파일 패턴 (예: `retarget_poses_g1.h5`)
     - `human_video_terrain_pattern`: 씬 데이터 파일 패턴 (예: `background_mesh.obj`)
     - `teacher_checkpoint_run_name`: 사용할 pretrained teacher checkpoint 이름

2. **데이터 파일 구조**:
   - **모션 데이터** (`retarget_poses_g1.h5` 또는 `.pkl`):
     - `root_pos`: 로봇 루트 위치 (전역 좌표계)
     - `root_quat`: 로봇 루트 방향 (쿼터니언)
     - `joints`: 관절 각도
     - `link_pos`: 각 링크의 위치 (루트 기준 로컬 좌표계)
     - `link_quat`: 각 링크의 방향
     - `contacts`: 발 접촉 정보 (left_foot, right_foot)
     - `fps`: 데이터 프레임레이트

   - **씬 데이터** (`background_mesh.obj`):
     - 3D 메쉬 파일로 저장된 배경 지형
     - 영상에서 재구성된 3D 환경

### 1.2 ReplayDataLoader 초기화

**위치**: `simulation/videomimic_gym/legged_gym/tensor_utils/replay_data.py`

```python
self.replay_data_loader = ReplayDataLoader(
    replay_data_path,  # 모션 데이터 파일 경로들
    self.num_envs,     # 환경 개수
    self.device,       # GPU/CPU
    self.dt,           # 시뮬레이션 타임스텝
    dof_names=...,     # 관절 이름들
    link_names=...,    # 링크 이름들
    contact_names=..., # 접촉 이름들
    ...
)
```

**주요 기능**:
- 여러 영상 클립을 로드하고 각 환경에 할당
- 타임스텝별로 target state 제공 (다음 프레임의 목표 자세)
- `randomize_start_offset`: 에피소드 시작 위치를 랜덤화 (옵션)
- `clip_weighting_strategy`: 클립 선택 전략 (uniform, success_rate_adaptive 등)

---

## 2. 씬(Scene) 설정: Terrain 생성

### 2.1 Terrain 로딩 (`DeepMimicTerrain`)

**위치**: `simulation/videomimic_gym/legged_gym/utils/deepmimic_terrain.py`

1. **메쉬 파일 로드**:
   ```python
   self.meshes = load_all_meshes(terrain_paths, convert_to_heightfield=...)
   ```
   - 각 영상 클립에 대응하는 `background_mesh.obj` 파일을 로드
   - 필요시 heightfield로 변환 (`cast_mesh_to_heightfield` 옵션)

2. **그리드 배치**:
   ```python
   self.vertices, self.triangles, ... = duplicate_mesh_grid_multi(
       self.meshes, 
       cfg.n_rows,  # 행 개수
       noise_config=noise_config  # 노이즈 설정 (옵션)
   )
   ```
   - 여러 환경을 위해 메쉬를 그리드 형태로 복제
   - 각 환경은 서로 다른 씬을 가질 수 있음

3. **Terrain Offset 계산**:
   ```python
   self.env_offsets = self.terrain.get_terrain_offset(
       self.replay_data_loader.episode_indices
   )
   ```
   - 각 환경의 씬이 배치된 위치를 계산
   - 로봇이 씬의 첫 프레임 위치에 정확히 생성되도록 조정

### 2.2 환경 좌표계 변환

**위치**: `simulation/videomimic_gym/legged_gym/envs/base/robot_deepmimic.py`

로봇의 위치는 두 가지 좌표계로 관리됩니다:

1. **World Frame**: 시뮬레이션의 절대 좌표계
2. **Env Frame**: 각 환경의 씬 기준 로컬 좌표계

```python
@property
def env_root_pos(self):
    return self.root_states[:, 0:3] - self.env_offsets
```

- `env_root_pos`: 씬 기준 로컬 위치 (모션 데이터와 직접 비교 가능)
- `root_states`: 시뮬레이션 절대 위치

---

## 3. 로봇 초기화: 첫 프레임 위치 설정

### 3.1 Reset 과정 (`reset_idx`)

**위치**: `simulation/videomimic_gym/legged_gym/envs/base/robot_deepmimic.py:206`

1. **Replay 데이터 리셋**:
   ```python
   self.reset_start_state = self.replay_data_loader.get_current_data()
   ```
   - 현재 에피소드의 첫 프레임 데이터를 가져옴

2. **Terrain Offset 설정**:
   ```python
   self.env_offsets[env_ids] = self.terrain.get_terrain_offset(
       self.reset_start_state.clip_index[env_ids]
   )
   ```
   - 각 환경의 씬이 배치된 위치를 계산
   - **중요**: 로봇이 씬의 첫 프레임 위치에 정확히 생성됨

3. **루트 상태 초기화** (`_reset_root_states`):
   ```python
   self.root_states[env_ids, 0:3] = self.env_frame_to_world_frame(
       self.reset_start_state.root_pos[env_ids], env_ids
   )
   self.root_states[env_ids, 3:7] = self.reset_start_state.root_quat[env_ids]
   ```
   - **첫 프레임의 root_pos와 root_quat을 사용하여 로봇 생성**
   - `env_frame_to_world_frame`: 로컬 좌표계를 절대 좌표계로 변환
   - 결과적으로 로봇은 영상의 첫 프레임 위치와 자세로 정확히 생성됨

4. **관절 초기화** (`_reset_dofs`):
   ```python
   self.dof_pos[env_ids] = self.reset_start_state.dofs[env_ids]
   self.dof_vel[env_ids] = self.reset_start_state.motor_vels[env_ids]
   ```
   - 첫 프레임의 관절 각도와 속도로 초기화

### 3.2 초기화 옵션

- `init_default_frac`: 일정 비율의 환경을 기본 자세로 초기화 (다양성 증가)
- `init_velocities`: 첫 프레임의 속도를 사용할지 여부
- `randomize_start_offset`: 에피소드 내 시작 위치를 랜덤화

---

## 4. 보상(Reward) 구조

### 4.1 보상 계산 흐름

**위치**: `simulation/videomimic_gym/legged_gym/envs/base/robot_deepmimic.py:433`

```python
def compute_reward(self):
    super().compute_reward()  # 기본 보상 계산
```

각 스텝마다 `update_replay_data()`가 호출되어 현재 타임스텝의 target state를 업데이트:

```python
def update_replay_data(self):
    state = self.replay_data_loader.get_current_data()
    self.target_root_pos = state.root_pos
    self.target_root_quat = state.root_quat
    self.target_dofs = state.dofs
    self.target_link_pos = state.link_pos
    self.target_contacts = state.contacts
    ...
```

### 4.2 주요 보상 항목

**위치**: `simulation/videomimic_gym/legged_gym/envs/g1/g1_deepmimic_config.py:165`

#### A. 모션 추적 보상 (Motion Tracking Rewards)

1. **Joint Position Tracking** (`joint_pos_tracking = 120.0`):
   ```python
   motor_pos_error = self.dof_pos - self.target_motors
   reward = torch.exp(-torch.pow(motor_pos_error, 2).sum(dim=-1) * k)
   ```
   - 관절 각도가 target과 일치할수록 높은 보상
   - `k = 2.0` (감쇠 계수)

2. **Joint Velocity Tracking** (`joint_vel_tracking = 24.0`):
   ```python
   motor_vel_error = self.dof_vel - self.target_motor_vels
   reward = torch.exp(-torch.pow(motor_vel_error, 2).sum(dim=-1) * k)
   ```
   - 관절 속도 추적

3. **Link Position Tracking** (`link_pos_tracking = 30.0`):
   ```python
   link_pos_error = self.env_rigid_body_pos[:, self.tracked_body_indices] - self.target_link_pos
   reward = torch.exp(-torch.pow(link_pos_error, 2).sum(dim=1).sum(dim=1) * k)
   ```
   - **중요**: 각 링크(손, 발, 팔 등)의 위치가 영상의 모션과 일치하도록 유도
   - 13개 주요 링크 추적

4. **Link Velocity Tracking** (`link_vel_tracking = 5.0`):
   - 링크 속도 추적

5. **Root Position Tracking** (`root_pos_tracking = 0.0`):
   ```python
   root_pos_error = self.env_root_pos - self.target_root_pos
   reward = torch.exp(-torch.pow(root_pos_error, 2).sum(dim=-1) * k)
   ```
   - 루트 위치 추적 (현재는 비활성화, scale=0.0)

6. **Root Orientation Tracking** (`root_orientation_tracking = 15.0`):
   ```python
   quat_diff = quat_mul(self.root_states[:, 3:7], quat_conjugate(self.target_root_quat))
   root_orientation_error = 2. * torch.asin(torch.clamp(torch.norm(quat_diff[:, :3], p=2, dim=-1), max=1.0))
   reward = torch.exp(-root_orientation_error * k)
   ```
   - 루트 방향 추적

7. **Torso Position/Orientation Tracking** (`torso_pos_tracking = 15.0`, `torso_orientation_tracking = 15.0`):
   - 상체 위치 및 방향 추적

#### B. 접촉 보상 (Contact Rewards)

8. **Feet Contact Matching** (`feet_contact_matching = 1.0`):
   ```python
   contact = self.contact_forces[:, self.feet_indices, 2] > 1.
   desired_contact = self.target_contacts
   reward = torch.sum((contact == desired_contact).float(), dim=1)
   ```
   - **중요**: 발 접촉 상태가 영상의 접촉 패턴과 일치해야 함
   - 왼발/오른발 각각에 대해 일치하면 +1

9. **Contact Smoothness** (`contact_smoothness = 0.0`):
   - 접촉 상태 변화의 부드러움 (현재 비활성화)

#### C. 정규화 보상 (Regularization Rewards)

10. **Action Rate** (`action_rate = -0.2`):
    - 액션 변화량 페널티 (부드러운 동작 유도)

11. **Collision** (`collision = -15.0`):
    - 충돌 페널티

12. **DOF Position Limits** (`dof_pos_limits = -50.0`):
    - 관절 한계 초과 페널티

13. **Termination** (`termination = -500.0`):
    - 에피소드 종료 시 큰 페널티

### 4.3 보상 가중치 요약

| 보상 항목 | Scale | 설명 |
|---------|-------|------|
| `joint_pos_tracking` | 120.0 | 관절 각도 추적 (가장 중요) |
| `link_pos_tracking` | 30.0 | 링크 위치 추적 |
| `joint_vel_tracking` | 24.0 | 관절 속도 추적 |
| `root_orientation_tracking` | 15.0 | 루트 방향 추적 |
| `torso_pos_tracking` | 15.0 | 상체 위치 추적 |
| `torso_orientation_tracking` | 15.0 | 상체 방향 추적 |
| `link_vel_tracking` | 5.0 | 링크 속도 추적 |
| `feet_contact_matching` | 1.0 | 발 접촉 일치 |
| `action_rate` | -0.2 | 액션 변화 페널티 |
| `collision` | -15.0 | 충돌 페널티 |
| `dof_pos_limits` | -50.0 | 관절 한계 페널티 |
| `termination` | -500.0 | 종료 페널티 |

---

## 5. Pretrained Policy Finetuning 과정

### 5.1 Teacher-Student 구조

1. **Pretrained Teacher Policy**:
   - AMASS 데이터셋으로 학습된 기본 locomotion policy
   - 다양한 지형에서 걷기, 뛰기 등의 기본 동작 학습
   - Checkpoint: `amass_teacher_checkpoint_run_name` (예: `20250410_063030_g1_deepmimic`)

2. **Student Policy (Finetuning 대상)**:
   - Teacher policy를 초기값으로 사용
   - 특정 영상의 모션과 씬에 맞게 finetuning

### 5.2 Finetuning 메커니즘

1. **데이터 혼합**:
   - AMASS 데이터 + 사용자 영상 데이터를 함께 사용
   - `use_amass = True`, `use_human_videos = True` 설정

2. **Adaptive Weighting**:
   ```python
   clip_weighting_strategy = 'success_rate_adaptive'
   ```
   - 각 클립의 성공률에 따라 가중치 조정
   - 어려운 클립에 더 많은 학습 기회 제공

3. **보상을 통한 학습**:
   - 영상의 모션을 target으로 설정
   - Policy가 target과 일치하도록 보상 최대화
   - 특히 `joint_pos_tracking`과 `link_pos_tracking`이 핵심

### 5.3 학습 과정 요약

1. **초기화**: Pretrained policy 로드
2. **에피소드 시작**: 
   - 영상의 첫 프레임 위치에 로봇 생성
   - 해당 영상의 씬(terrain) 로드
3. **스텝마다**:
   - 현재 프레임의 target state 가져오기
   - Policy가 액션 생성
   - 보상 계산 (target과의 차이 기반)
   - Policy 업데이트 (PPO 등)
4. **에피소드 종료**: 다음 클립으로 리셋

---

## 6. 핵심 포인트 정리

### 6.1 로봇 생성 위치

**질문**: "영상의 첫 프레임 위치에 정확히 생성되어서 그 모션을 따라하게 만드나?"

**답변**: **네, 정확히 그렇습니다.**

1. `reset_idx()`에서 `self.reset_start_state.root_pos`를 사용
2. 이 값은 영상의 첫 프레임에서 추출한 로봇 위치
3. `env_frame_to_world_frame()`로 변환하여 시뮬레이션에 배치
4. Terrain offset을 고려하여 씬의 첫 프레임 위치에 정확히 생성됨

### 6.2 Scene과 Motion의 통합

1. **Scene (Terrain)**:
   - `background_mesh.obj` 파일로 로드
   - 각 환경에 배치되어 물리 시뮬레이션에 사용
   - 로봇이 실제로 걸어다니는 지형

2. **Motion (Replay Data)**:
   - `retarget_poses_g1.h5` 파일로 로드
   - 각 타임스텝의 target 자세 제공
   - 보상 계산의 기준이 됨

3. **통합**:
   - 로봇이 씬 위에서 영상의 모션을 따라하도록 학습
   - 씬의 지형 특성(계단, 경사 등)을 고려하여 모션 재현

### 6.3 보상의 역할

보상은 **영상에서 추출한 모션 정보를 학습 신호로 변환**합니다:

- **Positive Rewards**: Target과 일치할수록 높음
  - 관절 각도, 링크 위치, 접촉 상태 등
- **Negative Rewards**: 물리적으로 불가능하거나 위험한 동작 페널티
  - 충돌, 관절 한계 초과, 종료 등

이를 통해 policy는 영상의 모션을 자연스럽고 물리적으로 타당하게 재현하도록 학습됩니다.

---

## 7. 코드 참조 위치

- **환경 초기화**: `simulation/videomimic_gym/legged_gym/envs/g1/g1_deepmimic.py`
- **보상 계산**: `simulation/videomimic_gym/legged_gym/envs/base/robot_deepmimic.py:433`
- **데이터 로더**: `simulation/videomimic_gym/legged_gym/tensor_utils/replay_data.py`
- **Terrain 로딩**: `simulation/videomimic_gym/legged_gym/utils/deepmimic_terrain.py`
- **설정 파일**: `simulation/videomimic_gym/legged_gym/envs/g1/g1_deepmimic_config.py`

---

## 결론

VideoMimic의 시뮬레이션은 다음과 같이 동작합니다:

1. **영상에서 추출한 scene과 motion 데이터를 로드**
2. **로봇을 영상의 첫 프레임 위치에 정확히 생성**
3. **각 타임스텝마다 영상의 모션을 target으로 설정**
4. **보상을 통해 policy가 target을 따라하도록 학습**
5. **Pretrained policy를 초기값으로 사용하여 finetuning**

이 과정을 통해 로봇은 영상의 모션을 자연스럽게 재현하면서도 물리적으로 타당한 동작을 학습합니다.

