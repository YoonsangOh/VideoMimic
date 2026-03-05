# vm1recon Setup Troubleshooting (2026-03-05)

## 목적
- `real2sim`의 Stage 1, Stage 3 실행을 위한 `vm1recon` (Python 3.10 + CUDA 11.8) 환경 재구축 과정에서 발생한 문제와 해결 과정을 기록한다.
- 동일 서버/유사 환경에서 재설치 시 반복 시행착오를 줄이기 위한 운영 문서다.

## 최종 상태
- `vm1recon`에서 다음이 모두 동작 확인됨:
  - `droid_backends`, `lietorch_backends` import 가능
  - `nksr` import 가능 (`Reconstructor` 존재)
  - `stage1_reconstruction/megasam_reconstruction.py --help` 정상
  - `stage3_postprocessing/postprocessing_pipeline.py --help` 정상
- 파이프라인 본 실행은 수행하지 않았고, 설치/임포트/진입 검증만 수행함.

## 주요 문제와 해결

### 1) `droid_backends` 빌드 실패 (gcc 13 + CUDA 11.8 조합)
- 증상:
  - `__builtin_dynamic_object_size` undefined
  - `_Float32` 관련 다수 오류
  - CUDA extension 컴파일 실패
- 원인:
  - Ubuntu 24.04의 기본 `gcc/g++ 13`과 CUDA 11.8의 조합 호환성 이슈
- 해결:
  - `conda-forge` 기반 `gcc/g++ 11.4` 툴체인 설치
  - 빌드 시 아래 환경변수 고정:
    - `CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc`
    - `CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++`
    - `CUDAHOSTCXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++`
  - `setup.py` 빌드 옵션의 `-allow-unsupported-compiler` 유지

### 2) conda 툴체인 강제 설치 후 헤더 누락
- 증상:
  - `cassert: No such file or directory`
  - `linux/limits.h: No such file or directory`
- 원인:
  - solver 우회를 위해 `--no-deps` 설치한 컴파일러 패키지의 의존 헤더 패키지 누락
- 해결:
  - 누락 패키지 추가 설치:
    - `binutils_linux-64`
    - `kernel-headers_linux-64`
    - `libgcc-devel_linux-64`
    - `libstdcxx-devel_linux-64`
    - `sysroot_linux-64`

### 3) 설치 도중 CUDA 13 계열 패키지 혼입
- 증상:
  - 환경에 CUDA 11.8/13.x 혼재
  - 컴파일 시 `CCCL` 관련 버전 충돌성 오류
- 원인:
  - `conda remove/install` 과정에서 solver가 `pkgs/main` 최신 CUDA 계열을 부분 반영
- 해결:
  - `vm1recon` 환경 삭제 후 재생성
  - CUDA dev 패키지는 `nvidia/label/cuda-11.8.0`에서 재설치

### 4) `xformers` 패키지 다운로드 중단/손상
- 증상:
  - 다운로드 파일 손상 (`bzip2: file ends unexpectedly`)
- 원인:
  - 네트워크 중단으로 tarball 불완전 수신
- 해결:
  - 손상 파일 삭제 후 재다운로드 (`curl -L --retry ...`)
  - 다운로드 완료 후 `conda install /tmp/xformers-...tar.bz2`

### 5) `nksr`가 placeholder 패키지로 설치되는 문제
- 증상:
  - `nksr 0.0.0`, `Reconstructor` 없음
- 원인:
  - 잘못된/접근 불가 인덱스로 인해 PyPI placeholder 수신
- 해결:
  - 공식 wheel 링크로 재설치:
    - `pip install nksr -f https://nksr.s3.ap-northeast-1.amazonaws.com/whl/torch-2.0.0%2Bcu118.html`
  - 결과: `nksr 1.0.3+pt20cu118`, `Reconstructor` 확인

### 6) `nksr` import 시 `libcusparse.so.11` 로딩 실패
- 증상:
  - `ImportError: libcusparse.so.11: cannot open shared object file`
- 원인:
  - `LD_LIBRARY_PATH`에 conda env의 `lib` 경로가 자동 반영되지 않음
- 해결:
  - `vm1recon` activate/deactivate 스크립트 추가:
    - `.../envs/vm1recon/etc/conda/activate.d/real2sim_vm1recon.sh`
    - `.../envs/vm1recon/etc/conda/deactivate.d/real2sim_vm1recon.sh`
  - 활성화 시 `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` 자동 주입

### 7) `chumpy`/`smplx` 설치 실패 (build isolation)
- 증상:
  - `ModuleNotFoundError: No module named 'pip'` (build env 내부)
- 원인:
  - build isolation과 패키지 setup 동작 충돌
- 해결:
  - `--no-build-isolation`로 설치:
    - `pip install --no-build-isolation git+https://github.com/hongsukchoi/chumpy`
    - `pip install --no-build-isolation git+https://github.com/hongsukchoi/smplx`

### 8) `python-pycg[full]` 미설치로 `open3d` 누락
- 증상:
  - `ModuleNotFoundError: No module named 'open3d'`
  - `nksr` import 체인 실패
- 원인:
  - 초기에는 `python-pycg` 기본 설치만 적용
- 해결:
  - 제공 인덱스로 `python-pycg[full]==0.5.2` 재설치
  - custom `open3d 0.16.1+c65c7ef` wheel 포함 설치 완료

## 참고: 최종 검증 포인트
- `python -c "import droid_backends, lietorch_backends"`
- `python -c "import nksr; print(hasattr(nksr, 'Reconstructor'))"`
- `python stage1_reconstruction/megasam_reconstruction.py --help`
- `python stage3_postprocessing/postprocessing_pipeline.py --help`

