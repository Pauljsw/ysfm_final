# 환경 설정 가이드

처음 시작하는 사람을 위한 전체 환경 설정 가이드입니다.

---

## 📋 목차

1. [시스템 요구사항](#시스템-요구사항)
2. [Python 환경 설정](#python-환경-설정)
3. [필수 패키지 설치](#필수-패키지-설치)
4. [외부 도구 설치](#외부-도구-설치)
5. [프로젝트 설정](#프로젝트-설정)
6. [설치 검증](#설치-검증)
7. [트러블슈팅](#트러블슈팅)

---

## 🖥️ 시스템 요구사항

### 최소 사양
- **OS**: Ubuntu 18.04+ / Windows 10+ / macOS 10.15+
- **CPU**: 4 cores 이상
- **RAM**: 16GB 이상 (32GB 권장)
- **GPU**: NVIDIA GPU (CUDA 지원) - YOLO inference 가속용
- **저장공간**: 10GB 이상 (데이터 + 모델)

### 권장 사양
- **GPU**: NVIDIA RTX 3060 이상 (8GB+ VRAM)
- **RAM**: 32GB 이상
- **저장공간**: 50GB 이상 (SSD 권장)

---

## 🐍 Python 환경 설정

### 1. Python 설치

**Python 3.8 - 3.11** 버전 필요 (3.9 권장)

#### Ubuntu/Linux
```bash
# Python 3.9 설치
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-dev

# 확인
python3.9 --version
```

#### Windows
[Python 공식 사이트](https://www.python.org/downloads/)에서 3.9 설치

#### macOS
```bash
# Homebrew로 설치
brew install python@3.9
```

### 2. 가상환경 생성 (권장)

```bash
# 가상환경 생성
python3.9 -m venv yolov11

# 가상환경 활성화
# Linux/Mac:
source yolov11/bin/activate

# Windows:
yolov11\Scripts\activate

# 확인 (yolov11) 프롬프트가 보이면 성공
```

---

## 📦 필수 패키지 설치

### 1. CUDA 설치 (GPU 사용 시)

#### Ubuntu
```bash
# CUDA Toolkit 11.8 설치 (PyTorch 2.0+ 호환)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# 환경변수 설정
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 확인
nvcc --version
```

#### Windows
[NVIDIA CUDA 다운로드](https://developer.nvidia.com/cuda-downloads)에서 설치

### 2. PyTorch 설치

**CUDA 있는 경우:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CPU만 사용:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**설치 확인:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. 프로젝트 의존성 설치

```bash
cd ysfm_final
pip install -r requirements.txt
```

**requirements.txt 내용:**
```txt
# Core dependencies
numpy>=1.21.0
scipy>=1.7.0
opencv-python>=4.5.0
scikit-learn>=0.24.0
scikit-image>=0.18.0
PyYAML>=5.4.0
torch>=2.0.0
ultralytics>=8.0.0

# For 3D reconstruction
open3d>=0.16.0

# For geometry operations
shapely>=2.0.0

# For progress bars
tqdm>=4.60.0

# Visualization
matplotlib>=3.3.0

# Testing
pytest>=6.2.0
pytest-cov>=2.12.0
```

### 4. 개별 패키지 설치 (문제 발생 시)

```bash
# 핵심 패키지
pip install numpy scipy opencv-python scikit-learn scikit-image PyYAML

# YOLO
pip install ultralytics

# 3D 처리
pip install open3d

# 기타
pip install shapely tqdm matplotlib
```

---

## 🔧 외부 도구 설치

### COLMAP (Phase 0 - SFM용)

COLMAP은 Structure from Motion (카메라 위치 계산)에 필요합니다.

#### Ubuntu
```bash
# 의존성 설치
sudo apt install \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev

# COLMAP 설치
sudo apt install colmap

# 또는 소스에서 빌드:
git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build
cd build
cmake ..
make -j
sudo make install
```

#### Windows
[COLMAP Releases](https://github.com/colmap/colmap/releases)에서 설치 프로그램 다운로드

#### macOS
```bash
brew install colmap
```

**설치 확인:**
```bash
colmap -h
```

---

## ⚙️ 프로젝트 설정

### 1. 프로젝트 클론

```bash
git clone <repository-url>
cd ysfm_final
```

### 2. 디렉토리 구조 생성

```bash
# 필수 디렉토리 생성
mkdir -p data/rgb data/depth data/yolo_masks
mkdir -p calib
mkdir -p models
mkdir -p outputs
mkdir -p logs
```

### 3. 카메라 캘리브레이션 파일 준비

#### RGB 카메라 (`calib/rgb_camera_info.json`)
```json
{
  "width": 3840,
  "height": 2160,
  "K": [
    [2800.0, 0.0, 1920.0],
    [0.0, 2800.0, 1080.0],
    [0.0, 0.0, 1.0]
  ],
  "D": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "distortion_model": "rational_polynomial"
}
```

#### Depth 카메라 (`calib/depth_camera_info.json`)
```json
{
  "width": 512,
  "height": 512,
  "K": [
    [450.0, 0.0, 256.0],
    [0.0, 450.0, 256.0],
    [0.0, 0.0, 1.0]
  ],
  "D": [0.0, 0.0, 0.0, 0.0, 0.0],
  "distortion_model": "radial_tangential"
}
```

#### Extrinsic (`calib/extrinsic_depth_to_color.json`)
```json
{
  "R": [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
  ],
  "t": [0.0, 0.0, 0.0]
}
```

### 4. YOLO 모델 준비

학습된 YOLO 모델을 `models/` 폴더에 배치:
```bash
models/best.pt
```

### 5. 데이터 준비

RGB 및 Depth 이미지를 준비:
```
data/
├── rgb/
│   ├── camera_RGB_0_0.png
│   ├── camera_RGB_0_1.png
│   └── ...
└── depth/
    ├── camera_DPT_0_0.png
    ├── camera_DPT_0_1.png
    └── ...
```

**중요:** RGB와 Depth 파일명의 타임스탬프 부분이 일치해야 합니다.

### 6. Config 파일 설정

`configs/simple.yaml`:
```yaml
# SFM Configuration
sfm:
  camera_model: 'OPENCV'
  quality: 'high'
  dense: true

# YOLO Configuration
yolo:
  model_path: 'models/best.pt'
  conf_threshold: 0.25
  iou_threshold: 0.45
  img_size: 1024

# Paths
paths:
  rgb_dir: 'data/rgb'
  depth_dir: 'data/depth'
  masks_dir: 'data/yolo_masks'
  output_dir: 'outputs'
```

### 7. 실행 권한 부여

```bash
chmod +x run.sh
```

---

## ✅ 설치 검증

### 1. Python 패키지 확인

```bash
python -c "
import numpy
import cv2
import torch
import ultralytics
import open3d
import sklearn
import skimage
import scipy
print('✅ All packages imported successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Ultralytics: {ultralytics.__version__}')
"
```

### 2. COLMAP 확인

```bash
colmap -h
```

### 3. GPU 확인 (선택)

```bash
nvidia-smi
```

### 4. 테스트 실행

```bash
# 도움말 확인
./run.sh

# Phase 0만 테스트 (데이터 필요)
# ./run.sh 1
```

---

## 🐛 트러블슈팅

### 1. ImportError: No module named 'cv2'

**원인:** OpenCV 미설치

**해결:**
```bash
pip install opencv-python
```

### 2. CUDA out of memory

**원인:** GPU 메모리 부족

**해결:**
```bash
# YOLO 이미지 크기 줄이기
# configs/simple.yaml에서:
yolo:
  img_size: 640  # 1024 → 640
```

또는 CPU 모드로 실행:
```bash
# run.sh에서 YOLO 부분 수정
python -m src.pipeline infer --config configs/simple.yaml --device cpu
```

### 3. COLMAP not found

**원인:** COLMAP 미설치 또는 PATH 설정 누락

**해결:**
```bash
# Ubuntu
sudo apt install colmap

# PATH 확인
which colmap

# PATH 추가 필요 시
echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 4. Permission denied: ./run.sh

**원인:** 실행 권한 없음

**해결:**
```bash
chmod +x run.sh
```

또는 소유권 변경:
```bash
sudo chown $USER:$USER run.sh
chmod +x run.sh
```

### 5. RuntimeError: torch not compiled with CUDA

**원인:** CPU 버전 PyTorch 설치됨

**해결:**
```bash
# 재설치
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 6. No module named 'ultralytics'

**원인:** Ultralytics 미설치

**해결:**
```bash
pip install ultralytics
```

### 7. open3d import error

**원인:** Open3D 버전 호환 문제

**해결:**
```bash
pip uninstall open3d
pip install open3d==0.16.0
```

### 8. 메모리 부족 (OOM)

**원인:** 대용량 포인트 클라우드

**해결:**
- `run.sh`에서 파라미터 조정:
  ```bash
  # Phase 3에서 --max-points 추가
  python -m src.point_cloud_overlay \
    ... \
    --max-points 1000000  # 100만 포인트로 제한
  ```

### 9. configs/simple.yaml not found

**원인:** Config 파일 없음

**해결:**
```bash
cp configs/default.yaml configs/simple.yaml
# 또는 새로 생성
```

---

## 📚 추가 리소스

- **COLMAP 문서**: https://colmap.github.io/
- **Ultralytics 문서**: https://docs.ultralytics.com/
- **PyTorch 설치 가이드**: https://pytorch.org/get-started/locally/
- **Open3D 문서**: http://www.open3d.org/docs/

---

## 🚀 다음 단계

설치가 완료되었다면:

1. **데이터 준비** - RGB/Depth 이미지 수집
2. **캘리브레이션** - 카메라 파라미터 설정
3. **YOLO 모델** - 학습된 모델 준비
4. **파이프라인 실행** - `./run.sh all`

자세한 실행 방법은 [README.md](README.md)를 참고하세요.

---

## ❓ 도움이 필요한가요?

문제가 해결되지 않으면:
1. 로그 확인: `logs/` 디렉토리
2. Issue 등록: GitHub Issues
3. 문의: [이메일 주소]

---

**버전 정보:**
- Python: 3.8 - 3.11
- PyTorch: 2.0+
- CUDA: 11.8+
- COLMAP: 3.8+
- Ultralytics: 8.0+
