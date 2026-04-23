# Pretraining Ablation 보고서 (2026-04-06)

## 범위

- 동일한 stage-2 scene-aware tracking 설정에서, 네 개의 clip에 대해 pretrained 초기화와 scratch 초기화를 짝지어 비교했다.
- 세 개의 VideoMimic clip은 이 repo 내부의 raw mp4와, 처리된 `retarget_poses_g1.h5` 및 `background_mesh.obj`를 사용한다.
- `holosoma_stairs`는 처리된 asset이 `videomimic_captures` 안에 있지만, 실제 `h5/obj`는 별도의 `holosoma` repo를 가리키는 symlink이며, `simulation/data/videomimic_raw_videos_mp4` 아래에는 holosoma raw mp4가 없다.
- 아래 metric은 success rate만 보지 않는다. 저장된 rollout pickle을 50 Hz 기준 reference motion과 정렬하여 joint RMSE, link position error, root error, contact accuracy, completion ratio를 계산했다.

## 핵심 결론

1. `5568`, `5585`에서는 pretraining이 completion과 matched-horizon tracking error 양쪽에서 모두 강하게 도움이 된다. `7276seg2`에서는 pretraining도 도움이 되지만, scratch도 이미 clip을 잘 학습하기 때문에 격차가 작다.
2. `holosoma_stairs`는 동일한 학습 budget 아래에서 가장 어려운 clip이다. pretraining은 completion을 분명히 개선하지만, matched-horizon tracking error는 크게 개선되지 않는다. 즉 주된 이득은 초반 몇 step의 tracking을 더 깨끗하게 만드는 것보다, 더 오래 살아남게 만드는 데 있다.
3. `holosoma_stairs`는 clip 길이가 가장 길고, uphill displacement가 가장 크며, motion z-range와 scene z-range도 가장 크다. 이런 특성은 동일한 학습 budget에서 scratch 성능이 가장 약하다는 결과와 일치한다.
4. success rate만으로는 충분하지 않다. 특히 `holosoma_stairs`의 pretrained policy는 많은 offset에서 살아남을 수는 있지만, 쉬운 clip들에 비해 tracking error는 여전히 크게 남는다.

## Clip Asset 정리

| Clip | repo 내부 raw video | ghost/reference replay | 처리된 motion | 처리된 scene |
| --- | --- | --- | --- | --- |
| 5568 | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_raw_videos_mp4/videomimic_release_video_data__anthony_apr21_IMG_5568__cam01.mp4` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures_mp4/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5568_cam01_frame_0_350_subsample_1.mp4` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5568_cam01_frame_0_350_subsample_1/retarget_poses_g1.h5` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5568_cam01_frame_0_350_subsample_1/background_mesh.obj` |
| 5585 | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_raw_videos_mp4/videomimic_release_video_data__anthony_apr21_IMG_5585__cam01.mp4` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures_mp4/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5585_cam01_frame_0_350_subsample_1.mp4` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5585_cam01_frame_0_350_subsample_1/retarget_poses_g1.h5` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5585_cam01_frame_0_350_subsample_1/background_mesh.obj` |
| 7276seg2 | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_raw_videos_mp4/videomimic_release_video_data__apr21_IMG_7276-00.02.11.110-00.02.16.628-seg2__cam01.mp4` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures_mp4/megahunter_align3r_reconstruction_results_apr21_IMG_7276-00.02.11.110-00.02.16.628-seg2_cam01_frame_0_300_subsample_2.pkl.mp4` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures/megahunter_align3r_reconstruction_results_apr21_IMG_7276-00.02.11.110-00.02.16.628-seg2_cam01_frame_0_300_subsample_2.pkl/retarget_poses_g1.h5` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures/megahunter_align3r_reconstruction_results_apr21_IMG_7276-00.02.11.110-00.02.16.628-seg2_cam01_frame_0_300_subsample_2.pkl/background_mesh.obj` |
| holosoma_stairs | `videomimic raw mp4 폴더 아래에는 없음` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/holosoma_stairs_reference_motion/holosoma_stairs_cam01_frame_0_140_subsample_1_reference_follow.mp4` | `/home/nas4_user/kyungminlee/work/yoonsangoh/holosoma/data/stairs/retarget_poses_g1.h5` | `/home/nas4_user/kyungminlee/work/yoonsangoh/holosoma/data/stairs/scene_mesh_gravity_aligned.obj` |

## 정량적 Tracking 비교

`success_rate_5`는 기존 mp4 렌더에 사용했던 5개의 canonical eval offset에 대한 결과다. `completion`은 종료되기 전까지 남아 있던 clip 구간 중 얼마나 완료했는지의 비율이다. `first50` metric은 매우 이른 실패 때문에 초반 tracking 품질이 가려지지 않도록, 초반 short-horizon tracking 품질만 따로 본 것이다.

| Clip | 초기화 | success_rate_5 | completion | first50 link err (cm) | first50 joint RMSE (deg) | full link err (cm) | full joint RMSE (deg) | root pos err (cm) | contact acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5568 | pretrained | 1.00 | 1.00 | 13.69 | 13.69 | 11.02 | 14.91 | 9.27 | 0.59 |
| 5568 | scratch | 0.20 | 0.34 | 19.07 | 28.61 | 19.29 | 29.54 | 18.60 | 0.56 |
| 5585 | pretrained | 0.80 | 1.00 | 12.58 | 8.53 | 13.39 | 5.54 | 12.53 | 0.70 |
| 5585 | scratch | 0.00 | 0.60 | 14.89 | 10.78 | 18.73 | 11.77 | 17.47 | 0.60 |
| 7276seg2 | pretrained | 1.00 | 1.00 | 9.59 | 4.65 | 10.33 | 3.82 | 9.79 | 0.48 |
| 7276seg2 | scratch | 1.00 | 1.00 | 10.33 | 9.81 | 12.67 | 7.90 | 12.35 | 0.57 |
| holosoma_stairs | pretrained | 0.80 | 0.81 | 40.55 | 30.69 | 44.59 | 37.17 | 17.94 | 0.74 |
| holosoma_stairs | scratch | 0.00 | 0.25 | 39.63 | 27.64 | 36.03 | 27.68 | 15.60 | 0.49 |

## Pretraining이 실제로 바꾼 것

| Clip | success delta | completion delta | matched-horizon link err delta (cm) | matched-horizon joint RMSE delta (deg) | full link err delta (cm) | cosine to MCPT: pretrained | cosine to MCPT: scratch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5568 | 0.80 | 0.66 | -5.82 | -15.10 | -8.28 | 0.9934 | 0.0063 |
| 5585 | 0.80 | 0.40 | -4.99 | -3.13 | -5.33 | 0.9922 | 0.0029 |
| 7276seg2 | 0.00 | 0.00 | -1.85 | -3.61 | -2.34 | 0.9964 | 0.0068 |
| holosoma_stairs | 0.80 | 0.57 | 0.15 | -1.70 | 8.56 | 0.9951 | 0.0010 |

해석:

- 위 표에서 error delta가 음수이면 pretrained 모델이 scratch보다 더 잘 추적했다는 뜻이다.
- `matched-horizon`은 각 eval offset마다 두 모델을 동일한 길이만큼만 비교한다. 그래서 아주 짧게 실패하는 scratch rollout이 겉보기에만 깨끗해 보이는 survivorship bias를 줄여준다.
- MCPT cosine similarity는 네트워크 수준의 점검이다. pretrained stage-2 policy는 stage-1 motion prior에 훨씬 더 가깝게 남아 있고, scratch policy는 random initialization에서 workable motion prior를 새로 찾아가야 한다.

## 동일 학습 budget 기준 Difficulty Ranking

여기서 difficulty는 동일한 30k stage-2 iteration을 썼을 때, final train success가 낮고, final train episode length가 짧고, 5-offset eval completion이 낮고, tracking error가 큰 clip일수록 더 어렵다고 정의했다.

| Clip | source dur (s) | 50 Hz frames | path_xy (m) | net_z (m) | motion z-range (m) | scene z-range (m) | switches/s | train success scratch | train ep len scratch | collection time scratch (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5568 | 8.13 | 407 | 2.37 | -1.11 | 1.11 | 1.44 | 5.78 | 0.80 | 165.53 | 2.46 |
| 5585 | 5.80 | 290 | 2.22 | 0.26 | 0.55 | 0.71 | 3.97 | 0.37 | 81.52 | 2.12 |
| 7276seg2 | 6.07 | 304 | 1.38 | -0.38 | 0.39 | 3.18 | 2.31 | 0.97 | 143.24 | 2.48 |
| holosoma_stairs | 9.33 | 467 | 6.29 | 2.37 | 2.49 | 5.18 | 0.64 | 0.11 | 117.56 | 3.20 |

30k stage-2 step 기준 추천 difficulty ranking, 쉬운 것부터 어려운 것 순:

1. `7276seg2`: scratch만으로도 canonical 5-offset eval에서 5/5 success에 도달한다.
2. `5568`: pretraining 효과는 크지만, scratch도 어느 정도는 학습 가능한 tracker를 만든다.
3. `5585`: clip 길이는 짧지만, scratch는 불안정하고 canonical 5 eval에서 한 번도 성공하지 못한다.
4. `holosoma_stairs`: 가장 길고, climb가 가장 크고, terrain relief도 가장 크며, 동일 budget에서 scratch 결과가 가장 약하다.

## 왜 `holosoma_stairs`가 더 어려운가

실제 수치로 뒷받침되는 이유는 다음과 같다.

- 전체 시퀀스가 가장 길다. 50 Hz 기준 467 policy step이며, 다른 세 clip은 407 / 290 / 304 step이다.
- 공간 이동량이 압도적으로 크다. xy path length가 6.29 m이고, net vertical gain도 +2.37 m이다.
- scene relief도 가장 크다. mesh z-range가 5.18 m이며, 다른 세 clip은 1.44 / 0.71 / 3.18 m 수준이다.
- 네트워크 구조와 하이퍼파라미터가 유사함에도 scratch policy는 30k step 이후에도 약한 상태에 머문다.
- pretrained 모델은 survival은 크게 개선하지만, tracking error는 쉬운 `7276seg2`에 비해 여전히 크다. 즉 0.5 m termination 기준에서 성공했다고 해서 high-fidelity tracking이 보장되지는 않는다.
- matched horizon 기준으로 보면, holosoma는 `5568`이나 `5585`보다 pretraining에 따른 error 개선 폭이 훨씬 작다. 이는 pretraining의 주효과가 즉각적인 low-level correction보다는, 장기 안정성을 위한 motion prior 제공에 더 가깝다는 뜻이다.

가능한 메커니즘은 이렇다. stairs는 긴 uphill trajectory 위에서 foothold placement를 지속적으로 맞춰야 한다. 정책은 whole-body motion prior를 유지하면서 동시에 terrain-conditioned correction도 배워야 한다. 반면 더 쉬운 clip들은 flat-ground imitation에 가까운 동작으로도 상당 부분 해결되므로, scratch 최적화가 덜 불안정하다.

## 네트워크 수준 관찰

- stage-1 MCPT checkpoint에는 terrain 전용 parameter key가 없다. stage-2 scene-aware policy에서는 공유 actor/critic trunk 위에 `actor_input_net.extra_proj_heads.terrain_height.*`, `critic_input_net.extra_proj_heads.terrain_height.*`가 추가된다.
- 즉 scene awareness는 완전히 다른 policy architecture로 바뀌는 것이 아니라, stage-2 finetuning 과정에서 terrain projection head가 붙는 방식이다.
- final terrain-attention magnitude는 clip마다 다르다. 가장 쉬운 `7276seg2`가 terrain-attention signal도 가장 강하다. 반면 `holosoma_stairs`는 이 값이 상대적으로 낮게 남아 있는데, 이는 단순히 mesh가 커서가 아니라 어려운 uphill clip에서 optimizer가 terrain observation을 충분히 활용하지 못했다는 해석과 더 잘 맞는다.

### actor / critic 구조를 코드 기준으로 풀어보면

이번 실험에 사용한 task는 `g1_deepmimic_proj_heightfield`이고, policy class는 `ActorCritic`이다. 관련 구현은 아래 경로에 있다.

- actor/critic 본체: `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_rl/rsl_rl/modules/actor_critic.py`
- task별 observation 구성: `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/envs/g1/g1_deepmimic_config.py`

이번 설정에서 네트워크는 개념적으로 아래처럼 볼 수 있다.

```text
Actor input
  = history_torso_real
  + history_torso_xy_rel
  + history_torso_yaw_rel
  + target_joints
  + target_root_roll
  + target_root_pitch
  + terrain_height
        └─ flatten
        └─ linear projection
        └─ channel-wise attention
        └─ first hidden layer에 add

Actor MLP
  [512] -> ELU -> [256] -> ELU -> [128] -> ELU -> [23 actions]

Critic input
  = torso
  + deepmimic
  + history_torso_real
  + history_torso_xy_rel
  + history_torso_yaw_rel
  + target_joints
  + target_root_roll
  + target_root_pitch
  + terrain_height
        └─ same style terrain projection

Critic MLP
  [512] -> ELU -> [256] -> ELU -> [128] -> ELU -> [1 value]
```

핵심은 actor와 critic이 완전히 같은 입력을 보지 않는다는 점이다.

- actor는 비교적 압축된 tracking cue와 terrain cue를 본다.
- critic은 여기에 `torso`, `deepmimic`을 더 보므로, 현재 상태와 tracking 오차를 더 풍부하게 평가한다.

즉 actor는 “무엇을 할지”를 결정하는 쪽이고, critic은 “현재 상태가 얼마나 좋은지”를 평가하는 쪽이라서, critic 쪽 입력이 더 풍부하다.

### attention 값은 정확히 무엇을 의미하나

이 부분은 이름 때문에 오해하기 쉽다. 여기의 `attention`은 Transformer식 spatial attention이나 softmax attention map이 아니다.

코드상 `terrain_height`는 다음 순서로 처리된다.

1. heightfield observation을 flatten한다.
2. 이를 `Linear(input_size -> hidden_dim)`으로 임베딩한다.
3. 그 결과에 `attention`이라는 learnable vector를 원소별 곱으로 적용한다.
4. 이렇게 만든 terrain embedding을 actor/critic의 첫 hidden layer 출력에 더한다.

즉 수식 형태로 보면 대략:

```text
terrain_proj = Linear(flatten(terrain_height))
terrain_gate = terrain_proj * attention
hidden_0 = Linear(other_obs) + terrain_gate
```

그래서 W&B에 기록한 값들의 뜻은 다음과 같다.

- `Network/attention_terrain_height_actor`
  - actor의 `attention` 벡터에 대해 `abs().mean()`을 기록한 값
  - terrain 임베딩이 첫 hidden layer에 얼마나 강하게 주입되는지의 대략적인 크기
- `Network/max_attention_terrain_height_actor`
  - actor attention 벡터의 최대값
  - 일부 hidden channel이 terrain 신호를 아주 강하게 쓰는지 보는 보조 지표
- critic 쪽도 동일

중요한 해석 포인트:

- 이 값은 `확률`이 아니다.
- 값들의 합이 1이 되는 softmax map도 아니다.
- 단순히 “terrain projection을 hidden 채널별로 얼마나 증폭/감쇠시키는가”를 나타내는 learned gate 크기다.
- 따라서 `0.08`이 `0.02`보다 크다고 해서 “4배 더 잘 본다”는 뜻은 아니고, terrain 신호가 네트워크 내부 pre-activation에 더 강하게 주입된다고 해석하는 것이 맞다.

### 이 표를 어떻게 읽으면 되나

- pretrained와 scratch의 `actor terrain attn`이 비슷한데 성능이 다를 수 있다.
  - 이는 terrain 신호의 크기 자체보다, stage-1에서 이미 확보한 motion prior가 있는지가 더 중요할 수 있음을 뜻한다.
- `holosoma_stairs`의 경우 pretrained도 scratch도 attention 수치 자체는 아주 크지 않다.
  - 즉 stairs 성능 한계는 “terrain head가 아예 없는 문제”라기보다, 긴 uphill clip에서 motion prior와 terrain correction을 동시에 최적화하기 어렵다는 쪽에 가깝다.
- `sofa` clip은 scratch에서도 attention이 크게 올라간다.
  - 이는 이 clip이 terrain/scene 정보를 비교적 쉽게 활용할 수 있는 구조였음을 시사한다.

| Clip | 초기화 | train success | train ep len | actor terrain attn | critic terrain attn | actor max terrain attn | critic max terrain attn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stair down | pretrained | 0.85 | 161.66 | 0.02 | 0.04 | 0.09 | 0.48 |
| stair down | scratch | 0.80 | 165.53 | 0.02 | 0.04 | 0.11 | 0.17 |
| stair up | pretrained | 0.97 | 153.97 | 0.03 | 0.08 | 0.14 | 1.30 |
| stair up | scratch | 0.37 | 81.52 | 0.03 | 0.03 | 0.16 | 0.27 |
| sofa | pretrained | 1.00 | 159.70 | 0.04 | 0.06 | 0.24 | 0.27 |
| sofa | scratch | 0.97 | 143.24 | 0.08 | 0.12 | 0.58 | 0.58 |
| holosoma_stairs | pretrained | 0.89 | 206.38 | 0.02 | 0.05 | 0.11 | 0.57 |
| holosoma_stairs | scratch | 0.11 | 117.56 | 0.02 | 0.06 | 0.07 | 0.48 |

| Clip | train success | train ep len | actor terrain attn | critic terrain attn | actor max terrain attn | critic max terrain attn |
| --- | --- | --- | --- | --- | --- | --- |
| stair down | 0.85 | 161.66 | 0.02 | 0.04 | 0.09 | 0.48 |
| stair up | 0.97 | 153.97 | 0.03 | 0.08 | 0.14 | 1.30 |
| sofa | 1.00 | 159.70 | 0.04 | 0.06 | 0.24 | 0.27 |
| holosoma_stairs | 0.89 | 206.38 | 0.02 | 0.05 | 0.11 | 0.57 |

## `holosoma_stairs` success rate를 해석할 때의 주의점

- 기존 100-episode eval만 봐도 success는 protocol에 매우 민감하다. `random-start 100eps`에서는 success가 0.98이지만, `first-frame 100eps`에서는 0.00이다.
- `0..50` offset 구간에서, train-time threshold인 0.5 m를 쓰면 pretrained는 0.95까지 올라간다. 하지만 stricter한 0.3 m eval threshold를 쓰면 0.00으로 떨어진다.
- 그래서 이 보고서는 success rate 하나가 아니라 tracking error와 completion ratio를 함께 강조한다.

## 공식 VideoMimic 파이프라인과의 연결

- root repo는 simulation pipeline이 mocap pretraining, scene-conditioned tracking, distillation, RL finetuning의 네 단계라고 설명한다.
- 관련 경로는 `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/README.md`, `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/docs/README.md`이다.
- 공식 stage-2 스크립트는 단일 run에서 `human_motion_list_123_motions.yaml`을 사용한다. 이는 stage-2가 123개의 independent clip policy를 각각 학습하는 것이 아니라, 하나의 multi-clip teacher를 학습한다는 직접적인 근거다.
- 관련 스크립트는 `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/scripts/train_stage_2_terrain_rl.sh`, `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/scripts/train_stage_3_distillation.sh`이다.
- stage-3는 다시 단일 `LOAD_RUN=stage_2_run_name`만 받기 때문에, 결국 distillation 대상도 하나의 teacher policy이지 123개의 별도 teacher가 아니다.

## 산출물 파일

- Eval metrics CSV: `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/docs/reports/pretraining_ablation_20260406_eval_metrics.csv`
- Training summary CSV: `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/docs/reports/pretraining_ablation_20260406_training_summary.csv`
- Clip stats CSV: `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/docs/reports/pretraining_ablation_20260406_clip_stats.csv`
- Pairwise delta CSV: `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/docs/reports/pretraining_ablation_20260406_pairwise_delta.csv`
