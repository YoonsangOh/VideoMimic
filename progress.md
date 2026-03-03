# VideoMimic Progress Log

## 2026-03-04

### Real2Sim Understanding (Current)
- `real2sim` 전체 파이프라인(0~4단계)과 각 단계의 입출력/의존 관계를 문서 및 스크립트 기준으로 확인했다.
- 기준 흐름:
  - Stage 0: 전처리 (프레임/마스크/2D pose/3D mesh/contact)
  - Stage 1: 장면 재구성 (MegaSam)
  - Stage 2: 인간-장면 정렬 최적화 (MegaHunter)
  - Stage 3: 중력 보정 + 메시화 (GeoCalib/NKSR)
  - Stage 4: 로봇 모션 리타겟팅

### Environment/CUDA Constraints
- 파이프라인 실행에는 두 가상환경이 모두 필요함:
  - `vm1rs` (Python 3.12, CUDA 12.4+): Stage 0/2/4 중심
  - `vm1recon` (Python 3.10, CUDA 11.8): Stage 1/3 중심
- CUDA 버전 요구사항이 달라 단일 로컬 GPU에서 full pipeline 실행이 불가할 수 있음을 확인.

### Multi-Server Operation Note
- 서버 접근 제약을 반영한 운영 방식 확정:
  - `h200 -> rtx3090` 직접 접속/전송은 구조적으로 불가
  - `rtx3090 -> h200` 접속은 가능
- 따라서 `rtx3090`에서 `h200` 중간 산출물을 다운로드하고, 필요한 위치로 재전송/배치하는 수동 방식으로 단계별 진행.

### Branch Status Snapshot
- 로컬 브랜치: `main`, `moge`
- 원격 브랜치: `origin/main`
- 스냅샷 시점 기준 `main`/`moge`/`origin/main`은 동일 HEAD를 가리킴.

### Real2Sim Stage1 Depth Backend Migration (MoGe) - 상세 작업 기록
- 목표:
  - Stage1 depth prior를 기본 `moge`로 전환
  - 기존 VideoMimic 방식(`depth_anything`)은 옵션으로 유지
  - 결과 비교를 위해 `output_tag` 기반 분리 실행 가능하게 구성

- 코드 변경:
  - `real2sim/stage1_reconstruction/megasam_reconstruction.py`
    - 신규 인자 추가:
      - `--depth-model {moge,depth_anything}` (기본: `moge`)
      - `--moge-pretrained` (기본: `Ruicheng/moge-vitl`)
      - `--output-tag`
    - `moge` 선택 시 MoGe inverse depth(disparity-like prior) 사용
    - `depth_anything` 선택 시 기존 경로 유지
    - `cvd_optimize`의 `freeze_shift` 인자 지원 여부를 런타임 시그니처 검사 후 조건부 전달(환경 호환성 확보)
  - `real2sim/sequential_processing/stage1_sequential_megasam_reconstruction.py`
    - 위와 동일한 인자/분기 로직 반영
    - sequential 모드에서 MoGe 모델 1회 로드 후 재사용
  - `real2sim/process_video.sh`
    - 인자 확장:
      - `[depth_model]` (기본: `moge`)
      - `[output_tag]` (선택)
    - Stage1 호출 시 `--depth-model`, `--output-tag` 전달
    - Stage2/3/4 입력 경로를 실제 Stage1/2 출력 basename 기반으로 계산하도록 개선 (tag 사용 시에도 경로 일치)
  - 신규 파일:
    - `real2sim/stage1_reconstruction/moge_depth_utils.py`
      - `load_moge_model()`
      - `infer_moge_inverse_depths()`

- 모델 코드 이식:
  - CRISP 경로의 MoGe 코드를 VideoMimic에 복사:
    - source: `/home/kyungminlee/CRISP-Real2Sim/prep/MogeSAM/third_party/megasam/MoGe`
    - target: `real2sim/moge`

- 검증(비GPU):
  - `python -m py_compile`로 변경된 Python 파일 문법 확인 완료
  - `bash -n real2sim/process_video.sh` 문법 확인 완료
  - VRAM 점유가 필요한 Stage 실행 테스트는 의도적으로 수행하지 않음

### MoGe Weights 상태
- `CRISP-Real2Sim/prep/MogeSAM` 폴더 내부에는 MoGe 전용 `model.pt`가 없고, `tapip3d_final.pth`는 TAPIP3D 체크포인트임을 재확인했다.
- CRISP의 MoGe 코드는 `MoGeModel.from_pretrained('Ruicheng/moge-vitl')`를 직접 호출하므로, 기본 동작은 HF에서 `model.pt`를 가져오거나(또는 기존 캐시를 재사용)하는 방식이다.
- 동일 모델 ID 확인:
  - 코드 근거: `prep/MogeSAM/third_party/megasam/MoGe/run_videos.py` 등에서 `Ruicheng/moge-vitl` 고정 사용
- 실제 캐시 확인:
  - `/home/nas_main/.cache/huggingface/hub/models--Ruicheng--moge-vitl/.../model.pt` 존재
  - blob 크기 약 `1.2G`, sha256: `da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f`
- 결론:
  - "CRISP에서 이미 돌아갔다"는 사실은 `model.pt`가 프로젝트 폴더가 아닌 HF 캐시에 저장되어 있었기 때문으로 해석된다.
  - VideoMimic 실행 시에도 `--moge-pretrained`에 로컬 `model.pt` 경로를 주면 HF 재다운로드 없이 동일 모델 사용 가능.

### Git 반영 상태
- 로컬 커밋 생성 완료:
  - `4a6db9f Add MoGe-based Stage1 depth backend with fallback and output tagging`
  - `3d64d54 Update progress log with MoGe integration and push attempt status`
- 원격 반영:
  - 사용자 수동 작업으로 `moge` 브랜치를 GitHub 원격에 푸시 완료됨.
- 참고:
  - 본 실행 환경에서의 직접 푸시는 DNS 제한으로 실패했으나, 최종 원격 반영은 사용자 수동 푸시로 해결됨.

### 정리 작업
- 중복 복사본 삭제 완료:
  - 삭제: `real2sim/third_party/megasam-package/MoGe`
  - 유지: `real2sim/moge` (VideoMimic Stage1 통합 경로)
