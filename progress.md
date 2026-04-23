# VideoMimic Progress Log

## 2026-04-01

### Single-clip 0331 run eval 및 future reference overlay mp4 생성
- `20260331_040113_0331_ft_5568`
- `20260331_040113_0331_ft_5585`
- `20260331_040157_0331_ft_7276seg2`
  의 `model_330000.pt`를 각각 eval하고, policy rollout 위에 reference future motion을 반투명 ghost처럼 겹쳐 보이는 mp4를 생성했다.

### 코드 수정
- `simulation/eval_single_clip_checkpoint_to_mp4.py`
  - 추가 이유: single-clip finetune run을 일반화해서 eval하고, rollout별 mp4에 "미래 reference motion"을 겹쳐 렌더하기 위해.
  - 구현 방식:
    - `eval` 모드: checkpoint를 불러와 5개의 start offset에서 rollout을 생성하고 `policy_rollout.pkl`, `metrics.json`, `summary_eval.json` 저장
    - `render` 모드: 저장된 rollout 하나를 다시 읽어 follow/fixed camera mp4 생성
    - policy robot은 cyan, 미래 reference는 주황 계열 반투명 overlay로 합성
    - future offsets 기본값은 `10,20,30` step
  - 추가 수정:
    - 처음에는 eval env와 render env를 한 프로세스 안에서 같이 만들었는데 Isaac Gym이 foundation object 중복 생성으로 실패했다.
    - 이를 해결하기 위해 `eval`과 `render`를 분리하고, render 시에는 별도 sim을 하나만 사용하도록 구조를 변경했다.
    - reference 쪽은 두 번째 sim/env를 만들지 않고 `ReplayDataLoader`만 따로 읽어 같은 env에 state를 적용해 ghost overlay를 만들도록 바꿨다.
- `simulation/run_single_clip_eval_render_future_ref.sh`
  - 추가 이유: rollout eval을 먼저 끝낸 뒤, rollout별 render를 별도 프로세스로 순차 수행하도록 자동화하기 위해.
  - 수정 이유:
    - Isaac Gym은 render 종료 시점에 종종 segfault를 내지만 mp4 파일은 정상 저장되는 경우가 많다.
    - 따라서 `render-one-rollout -> mp4 validation -> 다음 rollout` 방식이 전체 배치를 가장 안정적으로 끝까지 밀 수 있었다.

### 생성 결과
- 출력 루트:
  - `simulation/data/single_clip_policy_eval/20260331_040113_0331_ft_5568_ckpt330000`
  - `simulation/data/single_clip_policy_eval/20260331_040113_0331_ft_5585_ckpt330000`
  - `simulation/data/single_clip_policy_eval/20260331_040157_0331_ft_7276seg2_ckpt330000`
- 각 run마다:
  - `rollouts/eval_01` ~ `eval_05`
  - `mp4_follow_future_ref/eval_01.mp4` ~ `eval_05.mp4`
  - `summary_eval.json`
- 검증:
  - 세 run 모두 `5개 mp4`가 생성되었고 OpenCV로 열림을 확인했다.
  - 해상도는 `1280x720`, fps는 `50`.
  - render 프로세스는 일부 rollout에서 종료 시점 segfault를 냈지만, wrapper가 각 mp4를 즉시 검증해 유효한 산출물만 통과시켰다.

## 2026-03-17

### Holosoma stairs policy eval 재정리 및 고정 시점 mp4 생성
- `holosoma_stairs_heightfield_single_gpu_30k_20260316` run은 VideoMimic simulation 코드에서 돌아가는 `g1_deepmimic_proj_heightfield` 학습이다.
- 이 run은 `20250410_063030_g1_deepmimic` 체크포인트를 `--resume`로 이어받아 학습을 계속하는 구조이며, source motion은 이름과 달리 "사람 원본"이 아니라 G1 기준으로 retarget된 stairs motion/scene 데이터다.
- 근거:
  - `simulation/videomimic_gym/resources/data_config/holosoma_stairs_motion.yaml`
  - `simulation/data/videomimic_captures/holosoma_stairs_cam01_frame_0_140_subsample_1/retarget_poses_g1.h5`
  - 위 `h5`에는 `root_pos`, `root_quat`, `joints(23 dof)`, `link_pos`, `link_quat`가 저장되어 있다.

### 코드 수정
- `simulation/videomimic_gym/legged_gym/utils/__init__.py`
  - 수정 이유: `place_trace.py`가 `export_policy_as_jit` import 에러로 즉시 종료되던 문제를 해소하기 위해.
  - 수정 방식: `from rsl_rl.utils.jit import export_policy_as_jit`를 다시 export 하도록 추가.
- `simulation/videomimic_gym/legged_gym/envs/base/legged_robot.py`
  - 수정 이유: rollout export가 `replay_data_loader.pkl_paths`를 직접 참조해 `h5` 기반 stairs 데이터에서 깨지던 문제를 해소하기 위해.
  - 수정 방식: 내부 속성 직접 참조를 `replay_data_loader.get_pkl_paths()` 호출로 교체해 loader 구현 차이를 흡수.
- `simulation/eval_holosoma_stairs_checkpoint_to_mp4.py`
  - 추가 이유: 최신 checkpoint를 5회 eval 하고 rollout/pkl/mp4를 한 경로 아래에 남기기 위해.
  - 구현 방식:
    - eval 모드에서 특정 checkpoint를 5개의 start offset으로 rollout.
    - 각 rollout을 `policy_rollout.pkl`과 `metrics.json`으로 저장.
    - render 모드에서 rollout 디렉토리를 다시 읽어 offscreen mp4 생성.
  - 2026-03-17 추가 수정:
    - 기존 렌더가 root-relative follow camera라 ego-view처럼 보이던 문제를 확인.
    - `background_mesh.obj`와 rollout root trajectory bounds를 읽어 scene 전체를 볼 수 있는 world-fixed camera를 계산하도록 변경.
    - `--camera-mode {fixed,follow}` 인자를 추가했고 기본값은 `fixed`.
  - 2026-03-17 추가 수정 2:
    - 저장된 rollout은 정상인데 mp4가 정지 화면처럼 보이던 문제를 확인.
    - 원인: Isaac에서 root/dof state tensor만 갱신하고 실제 articulation pose를 렌더 직전 rigid-body pose로 flush하지 않아, 거의 동일한 프레임이 반복 저장되고 있었다.
    - 수정 방식: replay state를 적용한 뒤 `simulate -> fetch_results -> refresh_*_state_tensor`를 한 번 강제로 수행하는 `flush_pose_to_renderer()`를 추가.
- `simulation/render_videomimic_captures_mp4.py`
  - 수정 이유: 위와 동일한 offscreen articulation flush 문제가 이 스크립트에도 동일하게 존재했기 때문.
  - 수정 방식: eval 렌더 스크립트와 같은 `flush_pose_to_renderer()` 경로를 추가.

### 생성 결과
- 최신 checkpoint 기준 eval/mp4 산출물은 `simulation/data/holosoma_stairs_policy_eval/` 아래에 저장.
- 고정 시점 mp4는 각 checkpoint 결과 폴더의 `mp4_fixed_camera/` 아래에 저장.
- 2026-03-17 실제 생성 결과:
  - checkpoint: `model_330000.pt`
  - output root: `simulation/data/holosoma_stairs_policy_eval/20260316_034146_holosoma_stairs_heightfield_single_gpu_30k_20260316_ckpt330000`
  - mp4 dir: `simulation/data/holosoma_stairs_policy_eval/20260316_034146_holosoma_stairs_heightfield_single_gpu_30k_20260316_ckpt330000/mp4_fixed_camera`
  - 검증:
    - 5개 mp4 모두 `1280x720`, `50 fps`로 열림
    - 수정 전 `eval_02`의 mean frame diff는 `0.00022` 수준이었고, 수정 후 재렌더에서는 약 `0.0151`로 증가해 프레임별 pose 변화가 실제로 반영됨을 확인

## 2026-03-14

### Holosoma stairs -> VideoMimic geometry-aware tracking 연결 및 단일 GPU 학습 검증
- `g1_deepmimic` pretraining checkpoint를 `g1_deepmimic_proj_heightfield` task로 이어받아 stage-2 terrain RL을 수행하는 경로를 확인했다.
- 새 terrain head가 추가되는 구조이므로 `train.runner.load_model_strict=False`로 공통 가중치만 로드하고, 새 head는 초기화한다.
- 추가 자산:
  - `simulation/videomimic_gym/resources/data_config/holosoma_stairs_motion.yaml`
  - `simulation/videomimic_gym/legged_gym/scripts/train_holosoma_stairs_heightfield_single_gpu.sh`
  - `simulation/data/videomimic_captures/holosoma_stairs_cam01_frame_0_140_subsample_1/`
- 단일 GPU 학습 검증을 수행했고, `20260314_155558_holosoma_stairs_heightfield_single_gpu_20260314` run에서 checkpoint 생성과 PPO iteration 진행을 확인했다.

## 2026-03-05

### vm1recon 설치 이슈 정리 문서화
- `vm1recon` 재설치 과정에서 발생한 컴파일러/CUDA/xformers/nksr/chumpy-smplx/python-pycg[full] 관련 문제를 해결했다.
- 해결 절차와 재현 방법을 `real2sim/docs/vm1recon_setup_troubleshooting_2026-03-05.md`에 정리했다.
- Stage 1/3 진입 검증과 핵심 모듈 import 확인까지 마쳤다.

## 2026-03-04

### Real2Sim 파이프라인 정리 및 Stage1 depth backend 전환
- `real2sim`의 Stage 0~4 입출력과 환경 제약을 정리했다.
- 서버 간 전송 제약을 반영해 `rtx3090 -> h200` 중심의 단계별 운영 방식을 정리했다.
- Stage1 depth prior를 기본 `moge`로 전환하고, 기존 `depth_anything` 경로는 옵션으로 유지했다.
- 관련 변경:
  - `real2sim/stage1_reconstruction/megasam_reconstruction.py`
  - `real2sim/sequential_processing/stage1_sequential_megasam_reconstruction.py`
  - `real2sim/process_video.sh`
  - `real2sim/stage1_reconstruction/moge_depth_utils.py`
  - `real2sim/moge`
- `python -m py_compile`과 `bash -n`으로 비GPU 문법 검증을 마쳤다.
- 당시 로컬 커밋:
  - `4a6db9f Add MoGe-based Stage1 depth backend with fallback and output tagging`
  - `3d64d54 Update progress log with MoGe integration and push attempt status`
