# Pretraining Ablation Report (2026-04-06)

## Scope

- Four clips were compared as pretrained-vs-scratch pairs using the same stage-2 scene-aware tracking setup.
- The three VideoMimic clips use raw mp4 + processed `retarget_poses_g1.h5` + `background_mesh.obj` inside this repo.
- `holosoma_stairs` has processed assets inside `videomimic_captures`, but its underlying `h5/obj` are symlinks to the separate `holosoma` repo and there is no raw holosoma mp4 under `simulation/data/videomimic_raw_videos_mp4`.
- Metrics below go beyond success rate: joint RMSE, link-position error, root error, contact accuracy, and completion ratio were computed from saved rollout pickles aligned to the source reference motion at 50 Hz.

## Key Conclusions

1. Pretraining helps strongly on `5568` and `5585` in both completion and matched-horizon tracking error. On `7276seg2`, pretraining still helps but the gap is small because scratch already learns the clip well.
2. `holosoma_stairs` is the hardest clip under equal compute. Pretraining clearly improves completion there, but matched-horizon tracking error does not improve much. The main gain is staying alive longer, not making the very early steps cleaner.
3. `holosoma_stairs` is the longest clip, has the largest uphill displacement, the largest motion z-range, and the largest scene z-range. These factors align with its weaker equal-budget scratch result.
4. Success rate alone is misleading. In particular, `holosoma_stairs` pretrained can survive many offsets while still showing materially larger tracking error than the easy clips.

## Clip Assets

| Clip | Raw video in repo | Ghost/reference replay | Processed motion | Processed scene |
| --- | --- | --- | --- | --- |
| 5568 | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_raw_videos_mp4/videomimic_release_video_data__anthony_apr21_IMG_5568__cam01.mp4` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures_mp4/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5568_cam01_frame_0_350_subsample_1.mp4` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5568_cam01_frame_0_350_subsample_1/retarget_poses_g1.h5` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5568_cam01_frame_0_350_subsample_1/background_mesh.obj` |
| 5585 | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_raw_videos_mp4/videomimic_release_video_data__anthony_apr21_IMG_5585__cam01.mp4` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures_mp4/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5585_cam01_frame_0_350_subsample_1.mp4` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5585_cam01_frame_0_350_subsample_1/retarget_poses_g1.h5` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures/megahunter_megasam_reconstruction_results_anthony_apr21_IMG_5585_cam01_frame_0_350_subsample_1/background_mesh.obj` |
| 7276seg2 | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_raw_videos_mp4/videomimic_release_video_data__apr21_IMG_7276-00.02.11.110-00.02.16.628-seg2__cam01.mp4` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures_mp4/megahunter_align3r_reconstruction_results_apr21_IMG_7276-00.02.11.110-00.02.16.628-seg2_cam01_frame_0_300_subsample_2.pkl.mp4` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures/megahunter_align3r_reconstruction_results_apr21_IMG_7276-00.02.11.110-00.02.16.628-seg2_cam01_frame_0_300_subsample_2.pkl/retarget_poses_g1.h5` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/videomimic_captures/megahunter_align3r_reconstruction_results_apr21_IMG_7276-00.02.11.110-00.02.16.628-seg2_cam01_frame_0_300_subsample_2.pkl/background_mesh.obj` |
| holosoma_stairs | `not present under videomimic raw mp4 folder` | `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/data/holosoma_stairs_reference_motion/holosoma_stairs_cam01_frame_0_140_subsample_1_reference_follow.mp4` | `/home/nas4_user/kyungminlee/work/yoonsangoh/holosoma/data/stairs/retarget_poses_g1.h5` | `/home/nas4_user/kyungminlee/work/yoonsangoh/holosoma/data/stairs/scene_mesh_gravity_aligned.obj` |

## Quantitative Tracking Comparison

`success_rate_5` is over the 5 canonical eval offsets already used for the mp4 renders. `completion` is the fraction of the remaining clip completed before termination. `first50` metrics isolate short-horizon tracking quality so that very early failures do not hide initial tracking behavior.

| Clip | Init | success_rate_5 | completion | first50 link err (cm) | first50 joint RMSE (deg) | full link err (cm) | full joint RMSE (deg) | root pos err (cm) | contact acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5568 | pretrained | 1.00 | 1.00 | 13.69 | 13.69 | 11.02 | 14.91 | 9.27 | 0.59 |
| 5568 | scratch | 0.20 | 0.34 | 19.07 | 28.61 | 19.29 | 29.54 | 18.60 | 0.56 |
| 5585 | pretrained | 0.80 | 1.00 | 12.58 | 8.53 | 13.39 | 5.54 | 12.53 | 0.70 |
| 5585 | scratch | 0.00 | 0.60 | 14.89 | 10.78 | 18.73 | 11.77 | 17.47 | 0.60 |
| 7276seg2 | pretrained | 1.00 | 1.00 | 9.59 | 4.65 | 10.33 | 3.82 | 9.79 | 0.48 |
| 7276seg2 | scratch | 1.00 | 1.00 | 10.33 | 9.81 | 12.67 | 7.90 | 12.35 | 0.57 |
| holosoma_stairs | pretrained | 0.80 | 0.81 | 40.55 | 30.69 | 44.59 | 37.17 | 17.94 | 0.74 |
| holosoma_stairs | scratch | 0.00 | 0.25 | 39.63 | 27.64 | 36.03 | 27.68 | 15.60 | 0.49 |

## What Pretraining Changed

| Clip | success delta | completion delta | matched-horizon link err delta (cm) | matched-horizon joint RMSE delta (deg) | full link err delta (cm) | cosine to MCPT: pretrained | cosine to MCPT: scratch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5568 | 0.80 | 0.66 | -5.82 | -15.10 | -8.28 | 0.9934 | 0.0063 |
| 5585 | 0.80 | 0.40 | -4.99 | -3.13 | -5.33 | 0.9922 | 0.0029 |
| 7276seg2 | 0.00 | 0.00 | -1.85 | -3.61 | -2.34 | 0.9964 | 0.0068 |
| holosoma_stairs | 0.80 | 0.57 | 0.15 | -1.70 | 8.56 | 0.9951 | 0.0010 |

Interpretation:

- Negative error deltas above mean the pretrained model tracks better than scratch.
- `matched-horizon` compares both models over the same number of steps for each eval offset, which removes the survivorship bias that otherwise makes very short scratch rollouts look deceptively clean.
- The MCPT cosine similarity is a network-level check: pretrained stage-2 policies stay much closer to the stage-1 motion prior, while scratch models must discover a workable motion prior from random initialization.

## Difficulty Ranking Under Equal Compute

Difficulty is defined here by equal-budget evidence: lower final train success, lower final train episode length, lower 5-offset eval completion, and larger tracking error after the same 30k stage-2 iterations.

| Clip | source dur (s) | 50 Hz frames | path_xy (m) | net_z (m) | motion z-range (m) | scene z-range (m) | switches/s | train success scratch | train ep len scratch | collection time scratch (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5568 | 8.13 | 407 | 2.37 | -1.11 | 1.11 | 1.44 | 5.78 | 0.80 | 165.53 | 2.46 |
| 5585 | 5.80 | 290 | 2.22 | 0.26 | 0.55 | 0.71 | 3.97 | 0.37 | 81.52 | 2.12 |
| 7276seg2 | 6.07 | 304 | 1.38 | -0.38 | 0.39 | 3.18 | 2.31 | 0.97 | 143.24 | 2.48 |
| holosoma_stairs | 9.33 | 467 | 6.29 | 2.37 | 2.49 | 5.18 | 0.64 | 0.11 | 117.56 | 3.20 |

Suggested ranking from easiest to hardest at 30k stage-2 steps:

1. `7276seg2`: scratch already reaches 5/5 success on the canonical 5-offset evals.
2. `5568`: pretraining helps a lot, but scratch still learns a partially workable tracker.
3. `5585`: short clip, but scratch remains unstable and never succeeds on the 5 canonical evals.
4. `holosoma_stairs`: longest clip, largest climb, largest terrain relief, and the weakest equal-budget scratch outcome.

## Why `holosoma_stairs` Is Harder

Evidence-backed reasons:

- It is the longest sequence: 467 policy steps at 50 Hz, versus 407 / 290 / 304 for the other three clips.
- It covers by far the largest spatial excursion: 6.29 m of xy path length and +2.37 m net vertical gain.
- Its scene relief is also the largest: mesh z-range 5.18 m, versus 1.44 / 0.71 / 3.18 m for the other clips.
- The scratch policy remains weak after 30k steps despite similar network structure and hyperparameters.
- The pretrained model improves survival a lot, but its tracking error remains materially above the easy `7276seg2` case, which shows that success under a 0.5 m termination rule does not imply high-fidelity tracking.
- On a matched horizon, holosoma shows much smaller error gains from pretraining than `5568` or `5585`, which suggests the main benefit is a better motion prior for long-horizon stability rather than an immediate low-level correction.

A likely mechanism is that stairs require coordinated foothold placement over a long uphill trajectory. The policy has to preserve the whole-body motion prior and learn terrain-conditioned corrections at the same time. In the easier clips, motion can remain closer to flat-ground whole-body imitation, so scratch optimization is less brittle.

## Network-Level Observations

- The stage-1 MCPT checkpoint has no terrain-specific parameter keys. The stage-2 scene-aware policies add `actor_input_net.extra_proj_heads.terrain_height.*` and `critic_input_net.extra_proj_heads.terrain_height.*` on top of the shared actor/critic trunk.
- This means scene awareness is added as a terrain projection head during stage-2 finetuning, not as a completely different policy architecture.
- Final terrain-attention magnitudes are clip-dependent. The easiest clip, `7276seg2`, ends with the strongest terrain-attention signal. `holosoma_stairs` remains comparatively weak here, which is consistent with the optimizer struggling to exploit terrain observations effectively on the hardest uphill clip.

| Clip | Init | train success | train ep len | actor terrain attn | critic terrain attn | actor max terrain attn | critic max terrain attn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5568 | pretrained | 0.85 | 161.66 | 0.02 | 0.04 | 0.09 | 0.48 |
| 5568 | scratch | 0.80 | 165.53 | 0.02 | 0.04 | 0.11 | 0.17 |
| 5585 | pretrained | 0.97 | 153.97 | 0.03 | 0.08 | 0.14 | 1.30 |
| 5585 | scratch | 0.37 | 81.52 | 0.03 | 0.03 | 0.16 | 0.27 |
| 7276seg2 | pretrained | 1.00 | 159.70 | 0.04 | 0.06 | 0.24 | 0.27 |
| 7276seg2 | scratch | 0.97 | 143.24 | 0.08 | 0.12 | 0.58 | 0.58 |
| holosoma_stairs | pretrained | 0.89 | 206.38 | 0.02 | 0.05 | 0.11 | 0.57 |
| holosoma_stairs | scratch | 0.11 | 117.56 | 0.02 | 0.06 | 0.07 | 0.48 |

## Success-Rate Caveat for `holosoma_stairs`

- Existing 100-episode evals already show that success is protocol-sensitive: `random-start 100eps` success is 0.98, but `first-frame 100eps` success is 0.00.
- On the `0..50`-offset window with the train-time threshold (0.5 m), pretrained reaches 0.95; with a stricter 0.3 m eval threshold, it drops to 0.00.
- This is why the report above emphasizes tracking error and completion ratio, not success rate alone.

## Grounding in the Official VideoMimic Pipeline

- The root repo states that the sim pipeline has four stages: motion-capture pretraining, scene-conditioned tracking, distillation, and RL finetuning.
- See `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/README.md` and `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/docs/README.md`.
- The official stage-2 script uses `human_motion_list_123_motions.yaml` in a single run, which is direct evidence that stage-2 trains one multi-clip teacher rather than 123 independent clip policies.
- See `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/scripts/train_stage_2_terrain_rl.sh` and `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/videomimic_gym/legged_gym/scripts/train_stage_3_distillation.sh`.
- Stage-3 then takes a single `LOAD_RUN=stage_2_run_name`, again indicating one teacher policy is distilled, not 123 separate teachers.

## Output Files

- Eval metrics CSV: `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/docs/reports/pretraining_ablation_20260406_eval_metrics.csv`
- Training summary CSV: `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/docs/reports/pretraining_ablation_20260406_training_summary.csv`
- Clip stats CSV: `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/docs/reports/pretraining_ablation_20260406_clip_stats.csv`
- Pairwise delta CSV: `/home/nas4_user/kyungminlee/work/yoonsangoh/videomimic/simulation/docs/reports/pretraining_ablation_20260406_pairwise_delta.csv`
