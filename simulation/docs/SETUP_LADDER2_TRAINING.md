# Ladder2 강화학습 설정 가이드

이 가이드는 ladder2.mp4의 real2sim 결과를 사용하여 sim에서 강화학습을 진행하는 방법을 설명합니다.

## 1. 가상환경 설정

### 1.1 Conda 환경 생성 및 활성화

```bash
cd /home/nas5/kyungminlee/VideoMimic/simulation

# Conda 환경 생성 (이미 있다면 스킵)
conda create -n videomimic python=3.8 -y
conda activate videomimic
```

### 1.2 필수 패키지 설치

```bash
# PyTorch 설치
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Isaac Gym 설치 (이미 다운로드했다면)
# cd /path/to/isaacgym/python
# pip install -e .

# VideoMimic 패키지 설치
cd /home/nas5/kyungminlee/VideoMimic/simulation/videomimic_rl
pip install -e .
cd ../videomimic_gym
pip install -e .
cd ..
```

## 2. 데이터 준비

### 2.1 데이터 디렉토리 생성

```bash
mkdir -p /home/nas5/kyungminlee/VideoMimic/simulation/data/videomimic_captures
```

### 2.2 Real2Sim 결과 복사

ladder2 데이터를 simulation이 찾을 수 있는 위치로 복사합니다:

```bash
# 방법 1: 심볼릭 링크 생성 (권장, 디스크 공간 절약)
ln -s /home/nas5/kyungminlee/VideoMimic/real2sim/demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_ladder2_cam01_frame_100_700_subsample_2 \
      /home/nas5/kyungminlee/VideoMimic/simulation/data/videomimic_captures/megahunter_megasam_reconstruction_results_ladder2_cam01_frame_100_700_subsample_2

# 방법 2: 실제 복사 (디스크 공간이 충분하다면)
# cp -r /home/nas5/kyungminlee/VideoMimic/real2sim/demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_ladder2_cam01_frame_100_700_subsample_2 \
#       /home/nas5/kyungminlee/VideoMimic/simulation/data/videomimic_captures/
```

### 2.3 필요한 파일 확인

다음 파일들이 존재하는지 확인하세요:
- `retarget_poses_g1.h5` ✓ (확인됨)
- `background_mesh.obj` ✓ (확인됨)

## 3. YAML 설정 파일 생성

`simulation/videomimic_gym/resources/data_config/ladder2_motion.yaml` 파일이 생성되어 있습니다.

## 4. Teacher 체크포인트 준비

Stage 2 학습을 위해서는 Stage 1 (MoCap Pre-training)의 체크포인트가 필요합니다.

### 4.1 체크포인트 다운로드 (선택사항)

```bash
cd /home/nas5/kyungminlee/VideoMimic/simulation/data
bash download_videomimic_data.sh
```

이 스크립트는 체크포인트를 `videomimic_gym/logs/g1_deepmimic/`로 복사합니다.

### 4.2 기본 Teacher 체크포인트

기본적으로 `20250410_063030_g1_deepmimic` 체크포인트를 사용합니다.
이 체크포인트가 없다면 Stage 1을 먼저 실행해야 합니다.

## 5. 학습 실행

### 5.1 학습 스크립트 실행

```bash
cd /home/nas5/kyungminlee/VideoMimic/simulation
conda activate videomimic

# GPU가 1개인 경우
python videomimic_gym/legged_gym/scripts/train.py \
  --task=g1_deepmimic_proj_heightfield \
  --headless \
  --env.terrain.n_rows=1 \
  --num_envs=2048 \
  --wandb_note "ladder2_training" \
  --env.deepmimic.human_motion_source=resources/data_config/ladder2_motion.yaml \
  --env.deepmimic.use_human_videos=True \
  --env.deepmimic.use_amass=False \
  --load_run=20250410_063030_g1_deepmimic \
  --resume \
  --train.policy.re_init_std=True \
  --train.policy.init_noise_std=0.5 \
  --train.algorithm.learning_rate=2e-5 \
  --train.algorithm.schedule=fixed \
  --env.deepmimic.upsample_data=True \
  --env.deepmimic.link_pos_error_threshold=0.5 \
  --train.runner.save_interval=500 \
  --env.deepmimic.respawn_z_offset=0.1 \
  --env.deepmimic.randomize_terrain_offset=False \
  --env.terrain.cast_mesh_to_heightfield=False \
  --env.deepmimic.truncate_rollout_length=500 \
  --train.runner.load_model_strict=False \
  --env.rewards.scales.termination=-2000 \
  --env.rewards.scales.alive=200.0 \
  --env.rewards.scales.ankle_action=-3.0 \
  --env.rewards.scales.action_rate=-3.0

# GPU가 2개 이상인 경우
torchrun --nproc-per-node 2 videomimic_gym/legged_gym/scripts/train.py \
  --task=g1_deepmimic_proj_heightfield \
  --multi_gpu \
  --headless \
  --env.terrain.n_rows=1 \
  --num_envs=4096 \
  --wandb_note "ladder2_training" \
  --env.deepmimic.human_motion_source=resources/data_config/ladder2_motion.yaml \
  --env.deepmimic.use_human_videos=True \
  --env.deepmimic.use_amass=False \
  --load_run=20250410_063030_g1_deepmimic \
  --resume \
  --train.policy.re_init_std=True \
  --train.policy.init_noise_std=0.5 \
  --train.algorithm.learning_rate=2e-5 \
  --train.algorithm.schedule=fixed \
  --env.deepmimic.upsample_data=True \
  --env.deepmimic.link_pos_error_threshold=0.5 \
  --train.runner.save_interval=500 \
  --env.deepmimic.respawn_z_offset=0.1 \
  --env.deepmimic.randomize_terrain_offset=False \
  --env.terrain.cast_mesh_to_heightfield=False \
  --env.deepmimic.truncate_rollout_length=500 \
  --train.runner.load_model_strict=False \
  --env.rewards.scales.termination=-2000 \
  --env.rewards.scales.alive=200.0 \
  --env.rewards.scales.ankle_action=-3.0 \
  --env.rewards.scales.action_rate=-3.0
```

### 5.2 학습 중 체크포인트 확인

학습 중 생성되는 체크포인트는 다음 위치에 저장됩니다:
```
simulation/videomimic_gym/logs/g1_deepmimic/YYYYMMDD_HHMMSS_ladder2_training/
```

## 6. Viser로 학습 과정 시각화

학습 중에는 `--headless` 모드로 실행되므로 시각화가 없습니다.
학습된 모델을 시각화하려면 별도로 play 스크립트를 실행해야 합니다.

### 6.1 학습된 모델 재생

```bash
cd /home/nas5/kyungminlee/VideoMimic/simulation
conda activate videomimic

python videomimic_gym/legged_gym/scripts/play.py \
  --task=g1_deepmimic_proj_heightfield \
  --num_envs=1 \
  --env.viser.enabled=True \
  --load_run=YYYYMMDD_HHMMSS_ladder2_training \
  --env.deepmimic.human_motion_source=resources/data_config/ladder2_motion.yaml \
  --env.deepmimic.use_human_videos=True \
  --env.deepmimic.use_amass=False
```

### 6.2 Viser 접속

브라우저에서 `http://localhost:8080`에 접속하면 시각화를 볼 수 있습니다.

원격 서버에서 실행하는 경우:
- SSH 포트 포워딩: `ssh -L 8080:localhost:8080 user@server`
- VS Code 포트 포워딩 사용
- Cloudflare 터널 사용

## 7. 문제 해결

### 7.1 데이터를 찾을 수 없는 경우

`data_root` 경로를 확인하세요:
```bash
# 기본 경로: simulation/data/videomimic_captures
# 또는 alt_data_root 사용 가능
```

### 7.2 Teacher 체크포인트를 찾을 수 없는 경우

Stage 1을 먼저 실행하거나, 체크포인트를 다운로드하세요:
```bash
cd simulation/data
bash download_videomimic_data.sh
```

### 7.3 Isaac Gym 오류

Isaac Gym이 제대로 설치되었는지 확인:
```bash
cd /path/to/isaacgym/python/examples
python 1080_balls_of_solitude.py
```

## 8. 다음 단계

학습이 완료되면:
1. 학습된 체크포인트 확인
2. Viser로 재생하여 결과 확인
3. 필요시 Stage 3 (Distillation) 또는 Stage 4 (RL Finetuning) 진행
