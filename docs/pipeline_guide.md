# 3D Crack Detection and Measurement Pipeline

균열 탐지 및 3D 측정을 위한 전체 파이프라인 가이드입니다.

---

## Pipeline Overview

```
Phase 0: SfM → Phase 1: YOLO → Phase 2: D2C Scale Maps
                    ↓                      ↓
            Phase 3: Point Cloud Overlay ←─┘
                    ↓
            Phase 4: DBSCAN Clustering
                    ↓
            Phase 5: Measurement
```

---

## Phase 0: Structure from Motion (SfM)

### 목적
RGB 이미지들로부터 3D 장면 복원 및 카메라 포즈 추정

### Input
- `data/rgb/*.png` - RGB 이미지들

### Output
- `data/sfm/sparse/0/cameras.bin` - 카메라 내부 파라미터
- `data/sfm/sparse/0/images.bin` - 카메라 포즈 및 2D-3D 대응점
- `data/sfm/sparse/0/points3D.bin` - 3D 포인트 클라우드

### 실행 명령
```bash
python -m src.pipeline sfm --config configs/simple.yaml
```

### 왜 필요한가?
- 각 이미지의 카메라 위치/방향 정보 제공
- 2D 픽셀과 3D 좌표의 대응 관계 (track) 제공
- Multi-view voting을 위한 기반 데이터

---

## Phase 1: YOLO Inference

### 목적
RGB 이미지에서 균열 영역을 세그멘테이션

### Input
- `data/rgb/*.png` - RGB 이미지들
- YOLO 모델 가중치

### Output
- `data/yolo_masks/*.json` - 이미지별 마스크 정보
  ```json
  {
    "image_id": "camera_RGB_...",
    "masks": [
      {"class": "crack", "score": 0.95, "polygon": [[x,y], ...]}
    ]
  }
  ```

### 실행 명령
```bash
python -m src.pipeline infer --config configs/simple.yaml
```

### 왜 필요한가?
- 각 이미지에서 균열 영역 식별
- 3D 포인트가 균열인지 판단하는 기준 제공
- 2D 측정의 ROI 정의

---

## Phase 2: D2C Alignment + Per-pixel Scale Maps

### 목적
Depth 이미지를 RGB 좌표계로 정렬하고 픽셀별 mm/px 스케일 맵 생성

### Input
- `data/depth/*.png` - Depth 이미지들 (512x512, mm 단위)
- `calib/rgb_camera_info.json` - RGB 카메라 내부 파라미터
- `calib/depth_camera_info.json` - Depth 카메라 내부 파라미터
- `calib/extrinsic_depth_to_color.json` - Depth→RGB 외부 파라미터

### Output
- `outputs/d2c_pixel_scale/aligned_depth_*.npy` - 정렬된 depth (meters)
- `outputs/d2c_pixel_scale/aligned_depth_*.png` - 정렬된 depth (mm, 시각화용)
- `outputs/d2c_pixel_scale/scale_map_iso_*.npy` - **픽셀별 mm/px 스케일 맵**

### 실행 명령
```bash
python -m src.d2c_and_pixel_scale \
    --depth-dir data/depth \
    --rgb-calib calib/rgb_camera_info.json \
    --depth-calib calib/depth_camera_info.json \
    --extrinsic calib/extrinsic_depth_to_color.json \
    --output-dir outputs/d2c_pixel_scale
```

### 왜 필요한가?
- Depth와 RGB의 좌표계가 다름 → 정렬 필요
- 균열 측정 시 픽셀 거리를 실제 mm로 변환하는 기준
- 픽셀별로 다른 depth에 따른 정확한 스케일 적용

### 파일명 매칭
```
Depth: camera_DPT_1761702052_213355008.png
Scale: scale_map_iso_camera_DPT_1761702052_213355008.npy
RGB:   camera_RGB_1761702052_213355008.png
```
→ 동일 timestamp로 RGB와 Depth를 매칭

---

## Phase 3: Point Cloud Overlay

### 목적
SfM 포인트 클라우드에 YOLO 마스크를 오버레이하여 균열 3D 포인트 식별

### Input
- `data/sfm/sparse/0/` - SfM 결과 (Phase 0)
- `data/yolo_masks/*.json` - YOLO 마스크 (Phase 1)

### Output
- `outputs/sfm_masked_cloud.ply` - 균열 포인트가 표시된 PLY
- `outputs/crack_points.json` - 균열 포인트 정보:
  ```json
  {
    "points": [
      {
        "point_id": 12345,
        "xyz": [x, y, z],
        "source_masks": [
          {"image_id": "...", "mask_id": 0, "confidence": 0.95, "uv": [u, v]}
        ]
      }
    ]
  }
  ```

### 실행 명령
```bash
python -m src.point_cloud_overlay \
    --sparse-dir data/sfm/sparse/0 \
    --masks-dir data/yolo_masks \
    --output outputs/sfm_masked_cloud.ply \
    --output-json outputs/crack_points.json \
    --vote-threshold 0.5 \
    --min-confidence 0.3
```

### 왜 필요한가?
- Multi-view voting으로 노이즈 필터링 (50% 이상의 뷰에서 균열로 판단)
- 3D 좌표와 2D 픽셀 좌표의 매핑 정보 저장
- source_masks: 어떤 이미지/마스크에서 왔는지 추적 → Phase 5에서 사용

### 주요 파라미터
- `--vote-threshold`: 균열로 판정하기 위한 최소 투표 비율
- `--min-confidence`: YOLO confidence 임계값
- `--min-track-length`: 최소 관측 뷰 수

---

## Phase 4: DBSCAN Clustering

### 목적
균열 3D 포인트들을 개별 균열로 클러스터링

### Input
- `outputs/crack_points.json` (Phase 3)

### Output
- `outputs/crack_clusters.json` - 클러스터 정보:
  ```json
  {
    "clusters": [
      {
        "cluster_id": 0,
        "n_points": 456,
        "point_ids": [12345, 12346, ...],
        "centroid_3d": [x, y, z],
        "source_masks": [...]
      }
    ]
  }
  ```
- `outputs/clustered_cracks.ply` - 클러스터별 색상 시각화 (선택)

### 실행 명령
```bash
python -m src.cluster_crack_points_dbscan \
    --input outputs/crack_points.json \
    --output outputs/crack_clusters.json \
    --output-ply outputs/clustered_cracks.ply \
    --eps 0.05 \
    --min-samples 10 \
    --merge-distance 0.1 \
    --merge-angle 30
```

### 왜 필요한가?
- 여러 균열이 섞여 있는 포인트들을 개별 균열로 분리
- 각 균열별로 측정 수행 가능
- 공간적으로 가까운 포인트들을 같은 균열로 그룹화
- 방향이 유사한 클러스터는 병합하여 sparse로 끊긴 균열 복원

### 주요 파라미터

**Stage 1: DBSCAN**
- `--eps`: 같은 클러스터로 볼 최대 거리 (meters, 기본 0.05 = 5cm)
- `--min-samples`: 클러스터로 인정할 최소 포인트 수 (기본 10)

**Stage 2: 방향 인식 병합**
- `--merge-distance`: 병합 고려할 최대 중심점 거리 (meters, 기본 0.1 = 10cm)
- `--merge-angle`: 병합할 최대 주축 각도 차이 (degrees, 기본 30°)
- `--no-merge`: Stage 2 비활성화 (DBSCAN만 수행)

### 파라미터 조절 가이드

| 상황 | 권장 설정 |
|------|-----------|
| 균열이 너무 많이 분리됨 | `--eps` 증가, `--merge-distance` 증가 |
| 서로 다른 균열이 합쳐짐 | `--eps` 감소, `--merge-angle` 감소 |
| Sparse로 끊긴 균열 병합 | `--merge-distance` 증가, `--merge-angle` 유지 |
| 보수적 클러스터링 | `--no-merge` 사용 |

---

## Phase 5: Measurement

### 목적
각 클러스터(균열)의 길이와 폭을 측정

### Input
- `outputs/crack_clusters.json` (Phase 4)
- `outputs/crack_points.json` (Phase 3)
- `data/yolo_masks/*.json` (Phase 1)
- `outputs/d2c_pixel_scale/scale_map_iso_*.npy` (Phase 2)
- `data/rgb/*.png` (Phase 0) - edge 기반 폭 측정용

### Output
- `outputs/cluster_measurements.json`:
  ```json
  {
    "measurements": [
      {
        "cluster_id": 0,
        "total_length_mm": 125.3,
        "avg_width_mm": 2.15,
        "max_width_mm": 3.82,
        "n_segments": 4,
        "segments": [
          {
            "segment_id": 0,
            "image_id": "camera_RGB_...",
            "length_mm": 35.2,
            "avg_width_mm": 2.1,
            "width_method": "edge"
          }
        ]
      }
    ]
  }
  ```

### 실행 명령
```bash
python -m src.measure_clusters \
    --clusters outputs/crack_clusters.json \
    --crack-points outputs/crack_points.json \
    --masks-dir data/yolo_masks \
    --scale-maps-dir outputs/d2c_pixel_scale \
    --rgb-dir data/rgb \
    --output outputs/cluster_measurements.json \
    --image-width 3840 \
    --image-height 2160 \
    --n-segments 10 \
    --detection-method gradient \
    --gradient-percentile 85 \
    --min-component-ratio 0.3 \
    --max-width-filter 1.0 \
    --sample-interval 10 \
    --log-level INFO \
    --viz-dir outputs/visualizations_measurements
```

### 왜 필요한가?
- 균열의 실제 길이와 폭을 mm 단위로 측정
- 3D 세그먼트 분할 → 큰 균열도 정확히 측정
- Per-pixel scale map으로 각 픽셀의 정확한 스케일 적용

### 측정 방법
1. **3D 세그먼트 분할**: 클러스터를 principal axis 따라 N개 구간으로 분할
2. **세그먼트별 최적 마스크 선택**: 각 세그먼트를 가장 잘 커버하는 마스크 선택
3. **UV 기반 마스크 crop**: 세그먼트의 2D 영역만 측정
4. **Skeleton 기반 길이**: 스켈레톤화 → 방향별 픽셀 연결 × scale
5. **Gradient 기반 폭**: Sobel gradient로 균열 edge 검출 → 수직 방향 거리 측정

### 주요 파라미터
- `--n-segments`: 클러스터당 세그먼트 수 (기본 5)
- `--detection-method`: 균열 검출 방식 (gradient/percentile/adaptive/otsu)
- `--gradient-percentile`: gradient 상위 N% 사용 (기본 70, 높을수록 엄격)
- `--min-component-ratio`: 최소 영역 비율 (기본 0.1, 높을수록 엄격)
- `--max-width-filter`: 이 값(mm) 초과 폭 샘플 제외 (기본 None)
- `--sample-interval`: N 픽셀마다 폭 샘플링 (기본 5)
- `--viz-dir`: 측정 시각화 출력 경로
- `--no-edge-width`: gradient 측정 비활성화 (mask 기반 사용)

---

## Input/Output Dependencies

| Phase | Depends On | Produces |
|-------|------------|----------|
| 0. SfM | RGB images | sparse/0/ |
| 1. YOLO | RGB images | yolo_masks/ |
| 2. D2C Scale | Depth, Calibrations | scale_map_iso_*.npy |
| 3. Overlay | Phase 0, 1 | crack_points.json |
| 4. Clustering | Phase 3 | crack_clusters.json |
| 5. Measurement | Phase 1, 2, 3, 4 | cluster_measurements.json |

---

## Quick Start

```bash
# Phase 0: SfM
python -m src.pipeline sfm --config configs/simple.yaml

# Phase 1: YOLO inference
python -m src.pipeline infer --config configs/simple.yaml

# Phase 2: D2C + Scale Maps
python -m src.d2c_and_pixel_scale \
    --depth-dir data/depth \
    --rgb-calib calib/rgb_camera_info.json \
    --depth-calib calib/depth_camera_info.json \
    --extrinsic calib/extrinsic_depth_to_color.json \
    --output-dir outputs/d2c_pixel_scale

# Phase 3: Point Cloud Overlay
python -m src.point_cloud_overlay \
    --sparse-dir data/sfm/sparse/0 \
    --masks-dir data/yolo_masks \
    --output outputs/sfm_masked_cloud.ply \
    --output-json outputs/crack_points.json

# Phase 4: DBSCAN Clustering (with direction-aware merging)
python -m src.cluster_crack_points_dbscan \
    --input outputs/crack_points.json \
    --output outputs/crack_clusters.json \
    --output-ply outputs/clustered_cracks.ply \
    --eps 0.05 --min-samples 10 \
    --merge-distance 0.1 --merge-angle 30

# Phase 5: Measurement
python -m src.measure_clusters \
    --clusters outputs/crack_clusters.json \
    --crack-points outputs/crack_points.json \
    --masks-dir data/yolo_masks \
    --scale-maps-dir outputs/d2c_pixel_scale \
    --rgb-dir data/rgb \
    --output outputs/cluster_measurements.json \
    --image-width 3840 --image-height 2160 \
    --n-segments 10 --detection-method gradient \
    --gradient-percentile 85 --min-component-ratio 0.3 \
    --max-width-filter 1.0 --sample-interval 10 \
    --viz-dir outputs/visualizations_measurements
```

---

## Directory Structure

```
project/
├── data/
│   ├── rgb/                    # RGB 이미지 (Phase 0, 1 input)
│   ├── depth/                  # Depth 이미지 (Phase 2 input)
│   ├── sfm/sparse/0/           # SfM 결과 (Phase 0 output)
│   └── yolo_masks/             # YOLO 마스크 (Phase 1 output)
├── calib/
│   ├── rgb_camera_info.json
│   ├── depth_camera_info.json
│   └── extrinsic_depth_to_color.json
├── outputs/
│   ├── d2c_pixel_scale/        # Phase 2 output
│   │   └── scale_map_iso_*.npy
│   ├── crack_points.json       # Phase 3 output
│   ├── sfm_masked_cloud.ply    # Phase 3 output
│   ├── crack_clusters.json     # Phase 4 output
│   ├── clustered_cracks.ply    # Phase 4 output
│   └── cluster_measurements.json # Phase 5 output
└── src/
    ├── d2c_and_pixel_scale.py
    ├── point_cloud_overlay.py
    ├── cluster_crack_points_dbscan.py
    └── measure_clusters.py
```
