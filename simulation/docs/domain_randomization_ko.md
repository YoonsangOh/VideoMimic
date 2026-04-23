# VideoMimic Domain Randomization 정리

이 문서는 VideoMimic simulation 코드에서 `domain randomization`이 무엇을 대상으로, 어떤 방식으로, 언제 적용되는지 정리한다.
범위는 `simulation/videomimic_gym` 기준이며, 특히 우리가 자주 쓰는 Stage 1 MCPT와 Stage 2 scene-aware tracking을 중심으로 설명한다.

## 1. 한 줄 요약

- Domain randomization은 **real deploy 코드에만 있는 기능이 아니다**.
- VideoMimic에서는 이미 **simulation 학습 중**에 여러 randomization이 들어간다.
- 다만 Stage 1과 Stage 2의 강도와 종류는 다르다.
- 또한 코드상 `domain_rand.*`와 `noise.*`는 별개다.
  - `domain_rand.*`: 물리/제어/지연/마찰 같은 **환경/제어기 랜덤화**
  - `noise.*`: observation, reset state, replay cue 등에 넣는 **노이즈**

실제로 sim2real에서 중요한 robustness는 이 둘이 함께 만들어낸다.

## 2. 결론부터: Stage 1, 2, 3, 4에서 무엇이 켜지나

### Stage 1: MCPT pretraining

공식 스크립트 [train_stage_1_mcpt.sh](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/scripts/train_stage_1_mcpt.sh) 는 domain randomization을 **상당히 적극적으로 켠다**.

명시적으로 켜는 항목:

- `--env.domain_rand.p_gain_rand=True`
- `--env.domain_rand.d_gain_rand=True`
- `--env.domain_rand.push_robots=True`
- `--env.domain_rand.control_delays=True`
- `--env.domain_rand.control_delay_min=0`
- `--env.domain_rand.control_delay_max=5`
- `--env.domain_rand.randomize_base_mass=True`

같이 켜는 noise / robustness 관련 항목:

- `--env.noise.add_noise=True`
- `--env.noise.offset_scales.gravity=0.02`
- `--env.noise.offset_scales.dof_pos=0.005`
- `--env.noise.init_noise_scales.root_xy=0.1`
- `--env.noise.init_noise_scales.root_z=0.02`
- `--env.noise.init_noise_scales.dof_pos=0.01`
- `--env.deepmimic.randomize_terrain_offset=True`
- `--env.asset.use_alt_files=True`

즉 Stage 1은 단순 imitation pretraining이라기보다, **꽤 강한 robustness augmentation이 들어간 motion prior 학습**에 가깝다.

### Stage 2: scene-aware tracking

공식 스크립트 [train_stage_2_terrain_rl.sh](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/scripts/train_stage_2_terrain_rl.sh) 는 Stage 1보다 훨씬 보수적이다.

스크립트에서 따로 켜는 `domain_rand.*`는 거의 없고, 오히려:

- `--env.deepmimic.randomize_terrain_offset=False`

만 명시적으로 보인다.

하지만 Stage 2에 domain randomization이 **전혀 없는 것은 아니다**.
이유는 Stage 2가 쓰는 `G1DeepMimicCfg` 기본 config 안에 아래 값이 이미 들어 있기 때문이다.

[g1_deepmimic_config.py:635](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/g1/g1_deepmimic_config.py#L635)

- `randomize_friction = True`
- `friction_range = [0.1, 1.25]`
- `push_robots = False`
- `randomize_base_mass = False`
- `torque_rfi_rand = False`
- `p_gain_rand = False`
- `d_gain_rand = False`
- `randomize_dof_friction = False`

즉 공식 Stage 2는 기본적으로:

- **마찰 랜덤화는 켜져 있음**
- **push / gain rand / delay / base mass rand는 꺼져 있음**

그리고 noise 쪽은 [g1_deepmimic_config.py:276](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/g1/g1_deepmimic_config.py#L276) 에서 `add_noise = True`가 기본이라, Stage 2도 별도 override가 없으면 observation noise는 유지된다.

### Stage 3: distillation

공식 스크립트 [train_stage_3_distillation.sh](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/scripts/train_stage_3_distillation.sh) 에는 domain randomization을 강하게 켜는 override가 없다.

따라서 Stage 3는:

- task/config가 갖는 기본 마찰 랜덤화 여부에 따르고
- Stage 1처럼 강한 gain/delay/push 랜덤화는 스크립트 차원에서 추가하지 않는다.

### Stage 4: RL finetuning

공식 스크립트 [train_stage_4_rl_finetune.sh](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/scripts/train_stage_4_rl_finetune.sh) 에서는:

- `--env.noise.add_noise=True`

가 명시적으로 들어간다.
하지만 Stage 1처럼 강한 `domain_rand.*` override는 보이지 않는다.

즉 Stage 4도 기본 config 수준의 randomization은 유지하되, Stage 1만큼 공격적이지는 않다.

## 3. Domain randomization은 어디에서 실제로 적용되나

적용 시점은 크게 세 가지다.

1. **환경 생성 시 1회**
2. **episode reset 시 재샘플**
3. **매 step 적용**

이 구분이 중요하다. 같은 randomization이라도 어떤 것은 env가 처음 만들어질 때만 정해지고, 어떤 것은 매 episode마다 다시 뽑히며, 어떤 것은 매 step에서 바로 policy input이나 torque에 영향을 준다.

## 4. 항목별 정리

### 4.1 마찰 계수 randomization

설정:

- `cfg.domain_rand.randomize_friction`
- `cfg.domain_rand.friction_range`

기본 정의는 [legged_robot_config.py:188](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot_config.py#L188),
G1 DeepMimic override는 [g1_deepmimic_config.py:635](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/g1/g1_deepmimic_config.py#L635) 에 있다.

실제 적용은 [legged_robot.py:401](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L401) 의 `_process_rigid_shape_props()` 에서 이루어진다.

동작:

- env 생성 시 64개의 friction bucket을 샘플링
- 각 env는 bucket 하나를 배정받음
- 해당 env의 모든 rigid shape에 동일 friction 적용

즉 이 항목은 **env 생성 시 1회 randomize**된다.

### 4.2 DOF friction randomization

설정:

- `cfg.domain_rand.randomize_dof_friction`
- `cfg.domain_rand.max_dof_friction`
- `cfg.domain_rand.dof_friction_buckets`

적용 위치는 [legged_robot.py:440](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L440) 의 `_process_dof_props()` 이다.

동작:

- env 생성 시 DOF friction 값을 하나 샘플링
- bucketed value로 반올림
- 모든 joint friction에 같은 값 적용

이것도 **env 생성 시 1회 적용**이다.
현재 G1 DeepMimic 기본 config에서는 꺼져 있다.

### 4.3 Base mass randomization

설정:

- `cfg.domain_rand.randomize_base_mass`
- `cfg.domain_rand.added_mass_range`

적용 위치는 [legged_robot.py:476](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L476) 의 `_process_rigid_body_props()` 이다.

동작:

- `torso_link` body를 찾음
- 지정한 범위에서 추가 질량을 샘플링
- torso mass에 더함

즉 **env 생성 시 1회 적용**된다.

Stage 1에서는 스크립트에서 `True`로 켜고, Stage 2 기본 config에서는 `False`다.

### 4.4 Base COM randomization

설정:

- `cfg.domain_rand.randomize_base_com`
- `cfg.domain_rand.added_com_range`

적용 위치는 동일하게 [legged_robot.py:499](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L499).

동작:

- torso COM에 xyz offset을 더함

이것도 **env 생성 시 1회 적용**이다.
현재 기본 G1 DeepMimic config에서는 꺼져 있다.

### 4.5 Random pushes

설정:

- `cfg.domain_rand.push_robots`
- `cfg.domain_rand.push_interval_s`
- `cfg.domain_rand.max_push_vel_xy`

`push_interval_s`는 [legged_robot.py:1106](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L1106) 의 `_parse_cfg()` 에서 policy-step 기준 `push_interval`로 변환된다.

실제 push는:

- [legged_robot.py:213](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L213)
- [legged_robot.py:710](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L710)

에서 이루어진다.

동작:

- 각 env의 episode step을 보고 주기적으로 push 대상 env 선택
- root linear velocity에 랜덤 impulse를 직접 주입
- XY뿐 아니라 소량의 Z push도 들어간다

즉 이것은 **학습 중 매 step 검사, 특정 주기마다 적용**되는 항목이다.

Stage 1에서는 켜져 있고, Stage 2 기본 config에서는 꺼져 있다.

### 4.6 Torque RFI randomization

설정:

- `cfg.domain_rand.torque_rfi_rand`
- `cfg.domain_rand.torque_rfi_rand_scale`

초기화는 [legged_robot.py:557](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L557),
재샘플은 [legged_robot.py:588](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L588),
실제 토크에 더하는 것은 [legged_robot.py:647](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L647) 이다.

동작:

- episode마다 `torque_rfi_seed`를 다시 샘플링
- 각 step의 토크 계산에서 `seed * torque_limit * scale`을 더함

즉 **episode reset 시 재샘플 + 매 step 적용**이다.

현재 G1 DeepMimic 기본 config에서는 꺼져 있다.

### 4.7 P gain / D gain randomization

설정:

- `cfg.domain_rand.p_gain_rand`
- `cfg.domain_rand.p_gain_rand_scale`
- `cfg.domain_rand.d_gain_rand`
- `cfg.domain_rand.d_gain_rand_scale`

초기화/재샘플:

- [legged_robot.py:563](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L563)
- [legged_robot.py:592](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L592)

실제 적용:

- [legged_robot.py:626](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L626)

동작:

- episode마다 gain seed를 다시 뽑고
- torque 계산 시 `p_gain * (1 + seed * scale)`, `d_gain * (1 + seed * scale)` 형태로 사용

즉 **episode reset 시 재샘플 + 매 step 적용**이다.

Stage 1에서는 켜져 있고, Stage 2 기본 config에서는 꺼져 있다.

### 4.8 Control delay

설정:

- `cfg.domain_rand.control_delays`
- `cfg.domain_rand.control_delay_min`
- `cfg.domain_rand.control_delay_max`

초기화는 [legged_robot.py:571](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L571),
재샘플은 [legged_robot.py:596](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L596),
실제 사용은 [legged_robot.py:618](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L618) 이다.

동작:

- action history queue를 만든다
- env마다 delay index를 랜덤 샘플
- 현재 action 대신 과거 시점의 control을 꺼내 PD/torque 계산에 사용

즉 **episode reset 시 delay 길이 재샘플 + 매 step 적용**이다.

Stage 1에서는 켜져 있고, Stage 2 기본 config에서는 꺼져 있다.

### 4.9 Action delay

코드에는 action delay도 구현되어 있다.

- queue 초기화: [legged_robot.py:580](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L580)
- step 적용: [legged_robot.py:133](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py#L133)

하지만 base config [legged_robot_config.py:216](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/legged_robot_config.py#L216) 를 보면 `action_delays` 아래의 `action_delay_min/max` 정의가 현재 빠져 있고, `control_delay_min/max`가 중복되어 있다.
즉 구현 경로는 있지만, **base config 정의는 현재 불완전**하다. 실제 공식 Stage 1/2 스크립트도 이 경로를 사용하지 않는다.

### 4.10 Odom update frequency randomization

설정:

- `cfg.domain_rand.randomize_odom_update_frequency`
- `cfg.domain_rand.odom_update_steps_min`
- `cfg.domain_rand.odom_update_steps_max`

이 항목은 `robot_deepmimic.py` 안에서 target-relative cue가 매 step 즉시 갱신되지 않도록 만든다.

초기화/재샘플:

- [robot_deepmimic.py:605](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/robot_deepmimic.py#L605)
- [robot_deepmimic.py:615](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/robot_deepmimic.py#L615)

실제 적용:

- `torso_xy_rel`: [robot_deepmimic.py:675](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/robot_deepmimic.py#L675)
- `torso_yaw_rel`: [robot_deepmimic.py:736](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/robot_deepmimic.py#L736)

동작:

- env마다 update 주기를 랜덤 샘플
- 갱신 주기가 아닌 step에서는 이전 relative target cue를 그대로 유지

즉 **episode reset 시 재샘플 + 매 step에서 hold/update 적용**이다.

현재 G1 DeepMimic 기본 config에서는 꺼져 있다.

### 4.11 Terrain offset randomization

이 항목은 `domain_rand.*` 네임스페이스에는 없지만, Stage 1/2 robustness를 이해할 때 자주 같이 봐야 한다.

설정:

- `cfg.deepmimic.randomize_terrain_offset`
- `cfg.deepmimic.randomize_terrain_offset_range`

정의는 [g1_deepmimic_config.py:73](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/g1/g1_deepmimic_config.py#L73),
적용은 [robot_deepmimic.py:245](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/robot_deepmimic.py#L245) 에서 된다.

동작:

- clip index로부터 terrain grid offset을 계산하고
- reset 시 XY 방향으로 Gaussian perturbation을 더한다

즉 **episode reset 시 재샘플**되는 terrain placement noise다.

Stage 1에서는 켜고, 공식 Stage 2는 스크립트에서 `False`로 명시한다.

### 4.12 Alternate robot asset file

이것도 엄밀한 `domain_rand.*`는 아니지만, Stage 1에서 함께 쓰는 morphology-side robustness 요소다.

설정:

- `cfg.asset.use_alt_files`
- `cfg.asset.alt_files`

예시는 [g1_deepmimic_config.py:344](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/g1/g1_deepmimic_config.py#L344) 에 있다.

실제 적용은 multi-GPU helper [helpers.py:247](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/utils/helpers.py#L247) 에서 한다.

동작:

- multi-GPU 실행 시 rank마다 `file`과 `alt_files` 중 하나를 선택
- 서로 다른 collision geometry / URDF 변형을 rank별로 사용

즉 **프로세스 시작 시 1회 선택**되는 asset-level randomization이다.

## 5. Noise는 무엇이 다른가

VideoMimic에서는 domain randomization과 별도로 `noise.*`가 중요하다.

대표 항목:

- observation white noise
- fixed offset noise
- reset 시 root/joint perturbation
- replay cue 관련 relative observation noise

주요 정의는 [g1_deepmimic_config.py:276](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/g1/g1_deepmimic_config.py#L276) 이다.

예를 들어:

- reset 시 joint/root perturbation: [robot_deepmimic.py:282](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/robot_deepmimic.py#L282)
- torso proprio observation noise: [robot_deepmimic.py:625](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/robot_deepmimic.py#L625)
- `torso_xy_rel`, `torso_yaw_rel` 노이즈: [robot_deepmimic.py:674](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/robot_deepmimic.py#L674), [robot_deepmimic.py:734](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/base/robot_deepmimic.py#L734)

즉 VideoMimic의 robustness는 보통 다음 두 층으로 생긴다.

1. `domain_rand.*`로 physics/control 자체를 흔든다
2. `noise.*`로 observation과 reset state를 흔든다

## 6. Stage별 실전 해석

### Stage 1

Stage 1은 motion prior를 강하게 robust하게 만들려는 의도가 보인다.

- push on
- gain rand on
- control delay on
- base mass rand on
- observation noise on
- reset perturbation on
- terrain offset randomization on
- alt robot file on

즉 “flat-ish mocap tracking”이어도 sim2real에 가까운 robustness 학습을 이미 일부 시작한다.

### Stage 2

Stage 2는 scene-aware tracking 정확도가 중요하기 때문에, Stage 1보다 강한 randomization을 많이 뺀다.

실질적으로는:

- friction randomization은 유지
- push/gain/delay/base mass rand는 대부분 끔
- observation noise는 유지
- terrain offset randomization은 명시적으로 끔

즉 Stage 2는 **scene/motion 정합성을 크게 해치지 않는 수준의 robustness만 남기는 설계**로 볼 수 있다.

### Stage 3/4

Stage 3는 teacher imitation이 핵심이라 domain randomization override가 적고,
Stage 4는 RL finetune에서 noise를 유지하면서 더 정책을 안정화하는 형태다.

## 7. Evaluation / Play에서는 어떻게 되나

eval/play 계열 스크립트는 domain randomization과 noise를 대부분 꺼서 deterministic하게 본다.

예:

- [play.py:74](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/scripts/play.py#L74)
- [eval_single_clip_checkpoint_to_mp4.py:77](/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/eval_single_clip_checkpoint_to_mp4.py#L77)

보통 다음을 끈다.

- `env_cfg.noise.add_noise = False`
- `env_cfg.domain_rand.randomize_friction = False`
- `env_cfg.domain_rand.push_robots = False`

따라서 우리가 보는 eval mp4는 “훈련 때의 randomized world”가 아니라, **randomization을 걷어낸 정적 평가 world**에 가깝다.

## 8. 이것이 real deploy 코드에만 있나?

아니다. 오히려 반대다.

- domain randomization은 **simulation 학습 중** robustness를 주기 위해 존재한다.
- real deploy 코드는 이미 학습된 robust policy를 실제 로봇에 올리는 단계다.

repo 설명상 sim2real/physical deployment 코드는 일부만 공개되어 있고, unreleased 부분도 있다.
하지만 domain randomization 자체는 training-time 기법이며, real deploy 코드에서 처음 등장하는 기능이 아니다.

## 9. 최종 정리

- Stage 1에는 domain randomization이 분명히 들어간다. 꽤 강하다.
- Stage 2에도 domain randomization이 완전히 없는 것은 아니다.
  - 최소한 friction randomization은 기본 config상 켜져 있다.
  - 다만 Stage 1처럼 push / gain rand / control delay / mass rand를 적극적으로 쓰지는 않는다.
- VideoMimic에서는 `domain_rand.*`와 `noise.*`를 함께 봐야 robustness 설계를 제대로 이해할 수 있다.
- eval / render / play에서는 이들 대부분을 끈다.

따라서 “domain randomization은 real deploy 코드에만 있다”는 이해는 틀리고,
정확한 설명은 “VideoMimic은 simulation 학습 단계에서 이미 다양한 randomization과 noise를 사용하며, Stage 1이 가장 공격적이고 Stage 2는 훨씬 절제되어 있다”이다.
