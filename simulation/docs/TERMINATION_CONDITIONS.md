# VideoMimic Simulation Termination Conditions

이 문서는 VideoMimic 프로젝트에서 학습 및 play 시 episode가 종료(terminate)되는 모든 조건들을 정리합니다.

## 목차
1. [Base LeggedRobot Termination Conditions](#base-leggedrobot-termination-conditions)
2. [RobotDeepMimic Termination Conditions](#robotdeepmimic-termination-conditions)
3. [G1DeepMimic Termination Conditions](#g1deepmimic-termination-conditions)
4. [설정값 및 Threshold](#설정값-및-threshold)
5. [Episode Reset 동작](#episode-reset-동작)

---

## Base LeggedRobot Termination Conditions

모든 로봇 환경의 기본 termination 조건들입니다. (`legged_robot.py`의 `check_termination()` 메서드)

### 1. Termination Contact 충돌
- **조건**: 특정 body part가 terrain이나 다른 객체와 충돌할 때
- **코드**: `torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)`
- **설명**:
  - `termination_contact_indices`에 지정된 body part들에서 접촉력(contact force)의 크기가 1.0 N을 초과하면 terminate
  - 기본적으로 `terminate_after_contacts_on` 설정이 비어있으면 이 조건은 작동하지 않음
  - 주로 머리, 팔꿈치 등이 바닥에 닿았을 때를 감지

### 2. 로봇 자세 각도 초과 (Roll/Pitch)
- **조건**: 로봇이 너무 많이 기울어졌을 때
- **코드**: `torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)`
- **설명**:
  - **Roll (rpy[:,0])**: 로봇이 좌우로 기울어진 각도가 **0.8 rad (약 45.8도)** 초과 시 terminate
  - **Pitch (rpy[:,1])**: 로봇이 앞뒤로 기울어진 각도가 **1.0 rad (약 57.3도)** 초과 시 terminate
  - 로봇이 쓰러졌거나 균형을 잃었을 때를 감지

### 3. Timeout (최대 Episode Length 도달)
- **조건**: Episode가 최대 길이에 도달했을 때
- **코드**: `self.episode_length_buf > self.max_episode_length`
- **설명**:
  - `max_episode_length`는 `episode_length_s / dt`로 계산됨
  - 기본 `episode_length_s = 20.0` 초
  - Timeout은 **성공(success)**으로 간주됨 (다른 termination은 실패)

---

## RobotDeepMimic Termination Conditions

DeepMimic 환경에서 추가되는 termination 조건들입니다. (`robot_deepmimic.py`의 `check_termination()` 메서드)

### 1. Base 조건 상속
- Base `LeggedRobot`의 모든 termination 조건을 상속받습니다.

### 2. Link Position Error 초과 ⚠️ **가장 중요**
- **조건**: 로봇의 link 위치가 target 위치에서 너무 멀어졌을 때
- **코드**: `torch.any(link_pos_error > link_pos_error_threshold, dim=1) & (self.episode_length_buf >= 2)`
- **설명**:
  - `link_pos_error`: 로봇의 실제 link 위치와 target link 위치 간의 거리
  - `link_pos_error_threshold`: 허용 가능한 최대 오차 (기본값: **0.3 m**)
  - `episode_length_buf >= 2`: 최소 2 스텝 이후에만 적용 (초기 불안정성 방지)
  - **이 조건이 가장 흔한 조기 종료 원인입니다**
  - 로봇이 target motion을 따라가지 못할 때 terminate

### 3. 큰 발 접촉력 (옵션)
- **조건**: 발에 과도한 접촉력이 가해졌을 때 (옵션)
- **코드**: `if self.cfg.asset.terminate_after_large_feet_contact_forces: ...`
- **설명**:
  - 기본적으로 **비활성화**되어 있음 (`terminate_after_large_feet_contact_forces = False`)
  - 활성화 시, 발 접촉력이 `large_feet_contact_force_threshold` (기본값: 1000.0 N) 초과 시 terminate
  - 로봇이 너무 강하게 착지하거나 충돌했을 때를 감지

---

## G1DeepMimic Termination Conditions

G1 로봇용 DeepMimic 환경입니다. (`g1_deepmimic.py`의 `check_termination()` 메서드)

### 1. 부모 클래스 조건 상속
- `RobotDeepMimic`의 모든 termination 조건을 상속받습니다.

### 2. Success Rate Tracking
- **참고**: 이것은 termination 조건이 아니라, episode 종료 시 성공/실패를 기록하는 기능입니다.
- **성공(success)**: Timeout으로 종료된 경우 (`time_out_buf == True`)
- **실패(failure)**: 다른 termination 조건으로 종료된 경우

---

## 설정값 및 Threshold

### Base LeggedRobot 설정
- **`episode_length_s`**: 기본 20.0 초
- **`terminate_after_contacts_on`**: 기본 `[]` (비어있음, termination contact 비활성화)
- **Roll threshold**: 0.8 rad (약 45.8도) - 하드코딩
- **Pitch threshold**: 1.0 rad (약 57.3도) - 하드코딩

### RobotDeepMimic 설정 (`g1_deepmimic_config.py`)
- **`link_pos_error_threshold`**: 기본 **0.3 m**
  - Play 모드에서는 종종 더 큰 값 사용 (예: `10.0`)
  - 학습 모드에서는 작은 값 사용 (예: `0.3`, `0.5`)
- **`truncate_rollout_length`**: 기본 `-1` (비활성화)
  - 양수로 설정 시, replay data 길이와 이 값 중 작은 값으로 episode 길이 제한
- **`terminate_after_large_feet_contact_forces`**: 기본 `False`
- **`large_feet_contact_force_threshold`**: 기본 `1000.0 N` (위 옵션이 활성화된 경우에만 사용)

### Play 모드에서의 일반적인 설정
```python
--env.deepmimic.link_pos_error_threshold=10.0  # 더 관대한 threshold
```

---

## Episode Reset 동작

### Termination 발생 시
1. `check_termination()`이 호출되어 `reset_buf`가 설정됨
2. `post_physics_step()`에서 `reset_buf.nonzero()`로 terminate된 환경들을 찾음
3. `reset_idx(env_ids)`가 호출되어:
   - 환경을 초기 상태로 리셋
   - Replay data의 시작 프레임(또는 `clip_start_offset`으로 지정된 프레임)부터 다시 시작
   - Episode 카운터 초기화

### Timeout vs Failure
- **Timeout (성공)**: `time_out_buf == True`
  - Episode가 정상적으로 완료됨
  - Success rate tracking에서 성공으로 기록
- **Failure (실패)**: 다른 termination 조건
  - 조기 종료 (early termination)
  - Success rate tracking에서 실패로 기록

---

## 일반적인 Termination 시나리오

### 1. 조기 종료 (가장 흔함)
- **원인**: `link_pos_error_threshold` 초과
- **증상**: 로봇이 target motion을 따라가지 못함
- **해결**:
  - `link_pos_error_threshold` 값을 증가시킴 (play 모드)
  - 정책을 더 학습시킴 (학습 모드)

### 2. 로봇 쓰러짐
- **원인**: Roll/Pitch 각도 초과
- **증상**: 로봇이 균형을 잃고 넘어짐
- **해결**: 정책 학습 개선 필요

### 3. 충돌
- **원인**: Termination contact 충돌 (설정된 경우)
- **증상**: 특정 body part가 terrain에 닿음
- **해결**: 정책 학습 개선 또는 threshold 조정

### 4. 정상 완료
- **원인**: Timeout (max episode length 도달)
- **증상**: Episode가 정상적으로 완료됨
- **의미**: 성공으로 간주됨

---

## 코드 위치 참조

- **Base termination**: `simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py:237-243`
- **DeepMimic termination**: `simulation/videomimic_gym/legged_gym/envs/base/robot_deepmimic.py:412-425`
- **G1DeepMimic termination**: `simulation/videomimic_gym/legged_gym/envs/g1/g1_deepmimic.py:322-362`
- **설정 파일**: `simulation/videomimic_gym/legged_gym/envs/g1/g1_deepmimic_config.py`

---

## 주의사항

1. **`link_pos_error_threshold`는 가장 중요한 설정값입니다**
   - 너무 작으면 조기 종료가 빈번하게 발생
   - 너무 크면 로봇이 target을 완전히 벗어나도 계속 진행

2. **Play 모드에서는 더 관대한 threshold 사용 권장**
   - 학습된 정책이 완벽하지 않을 수 있음
   - `link_pos_error_threshold=10.0` 정도로 설정하면 더 긴 재생 가능

3. **빈 공간에서 시작하면 즉시 terminate될 수 있음**
   - `background_mesh.obj`의 범위 밖에서 시작하면 로봇이 떨어짐
   - Roll/Pitch 각도 초과로 즉시 terminate 가능

4. **Termination 조건은 OR 연산으로 결합됨**
   - 하나의 조건만 만족해도 terminate
   - 가장 먼저 만족하는 조건이 적용됨
