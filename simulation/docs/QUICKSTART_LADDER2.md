# Ladder2 강화학습 빠른 시작 가이드

## ✅ 준비 완료된 항목

1. ✅ Real2Sim 결과물 생성 완료
   - `retarget_poses_g1.h5` ✓
   - `background_mesh.obj` ✓

2. ✅ 데이터 심볼릭 링크 생성 완료
   - 위치: `simulation/data/videomimic_captures/megahunter_megasam_reconstruction_results_ladder2_cam01_frame_100_700_subsample_2`

3. ✅ YAML 설정 파일 생성 완료
   - 위치: `simulation/videomimic_gym/resources/data_config/ladder2_motion.yaml`

4. ✅ 학습 스크립트 생성 완료
   - `train_ladder2.sh`: 학습 실행 스크립트
   - `play_ladder2.sh`: 결과 시각화 스크립트

5. ✅ **패키지 설치 완료**
   - Isaac Gym 설치 완료
   - videomimic_rl 설치 완료
   - videomimic_gym 설치 완료

## 🚀 다음 단계

### Step 1: 설치 확인 (선택사항)

```bash
cd /home/nas5/kyungminlee/VideoMimic/simulation
bash verify_installation.sh
```

### Step 2: Teacher 체크포인트 준비

Stage 2 학습을 위해서는 Stage 1의 체크포인트가 필요합니다.

**옵션 A: 기존 체크포인트 다운로드 (권장)**
```bash
cd /home/nas5/kyungminlee/VideoMimic/simulation/data
bash download_videomimic_data.sh
```

**옵션 B: Stage 1 먼저 학습**
```bash
bash videomimic_gym/legged_gym/scripts/train_stage_1_mcpt.sh
```

### Step 3: 학습 실행

```bash
cd /home/nas5/kyungminlee/VideoMimic/simulation
conda activate videomimic

# GPU 1개인 경우
bash train_ladder2.sh 1

# GPU 2개인 경우
bash train_ladder2.sh 2
```

학습이 시작되면:
- 체크포인트는 `videomimic_gym/logs/g1_deepmimic/YYYYMMDD_HHMMSS_ladder2_training/`에 저장됩니다
- `--headless` 모드이므로 학습 중에는 시각화가 없습니다
- WandB를 사용하면 온라인으로 학습 진행 상황을 볼 수 있습니다

### Step 4: 학습 결과 시각화 (Viser)

학습이 완료되거나 중간 체크포인트를 확인하려면:

```bash
cd /home/nas5/kyungminlee/VideoMimic/simulation
conda activate videomimic

# 최신 체크포인트로 재생
bash play_ladder2.sh YYYYMMDD_HHMMSS_ladder2_training

# 특정 체크포인트로 재생
bash play_ladder2.sh YYYYMMDD_HHMMSS_ladder2_training 1000
```

브라우저에서 `http://localhost:8080` 접속하면 Viser UI가 열립니다.

**원격 서버인 경우:**
```bash
# SSH 포트 포워딩
ssh -L 8080:localhost:8080 user@server

# 또는 VS Code의 포트 포워딩 기능 사용
```

## 📋 체크리스트

학습 전 확인사항:

- [x] Conda 환경 `videomimic` 생성 및 활성화
- [x] PyTorch 설치 완료
- [x] Isaac Gym 설치 완료
- [x] VideoMimic 패키지 설치 완료 (`videomimic_rl`, `videomimic_gym`)
- [x] 데이터 심볼릭 링크 확인 (`simulation/data/videomimic_captures/...`)
- [x] `retarget_poses_g1.h5` 파일 존재 확인
- [x] `background_mesh.obj` 파일 존재 확인
- [ ] Teacher 체크포인트 준비 (`20250410_063030_g1_deepmimic` 또는 Stage 1 완료)

## 🔍 문제 해결

### Isaac Gym import 오류

Isaac Gym이 설치되었지만 import 시 오류가 발생할 수 있습니다. 이는 일반적으로 런타임 라이브러리 경로 문제입니다. 실제 학습 시에는 문제가 없을 수 있습니다.

만약 학습 중 오류가 발생하면:
```bash
# Isaac Gym 재설치
bash install_isaacgym.sh
```

### "Data directory not found" 오류
```bash
# 심볼릭 링크 다시 생성
cd /home/nas5/kyungminlee/VideoMimic/simulation
rm -rf data/videomimic_captures/megahunter_megasam_reconstruction_results_ladder2_cam01_frame_100_700_subsample_2
ln -s /home/nas5/kyungminlee/VideoMimic/real2sim/demo_data/output_calib_mesh/megahunter_megasam_reconstruction_results_ladder2_cam01_frame_100_700_subsample_2 \
      data/videomimic_captures/megahunter_megasam_reconstruction_results_ladder2_cam01_frame_100_700_subsample_2
```

### "Teacher checkpoint not found" 오류
```bash
# 체크포인트 다운로드
cd simulation/data
bash download_videomimic_data.sh
```

## 📚 추가 정보

자세한 내용은 `SETUP_LADDER2_TRAINING.md`를 참고하세요.
