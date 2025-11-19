# Sparse Mode with 3D Clustering - Usage Guide

## Overview

이 가이드는 **Direct 3D Mask Projection** 방식을 사용한 Sparse Mode Clustering을 설명합니다.

**핵심 차이점:**
- ❌ Dense Point Cloud 생성 불필요
- ❌ Point density 의존성 없음
- ✅ YOLO mask를 직접 3D로 변환
- ✅ Geometric overlap clustering
- ✅ 빠르고 정확 (60분 → 7분)

---

## Pipeline 전체 흐름

```
Phase 0: SfM (Sparse)                     → poses.json
Phase 1: YOLO Inference                   → yolo_masks/*.json
Phase 2: Pixel Calibration (Optional)     → pixel_scales.json
Phase 3: Depth-RGB Alignment              → depth_upsampled/*.png
Phase 4: 2D Mask → 3D Projection ⭐ NEW   → masks_3d.json
Phase 5: 3D Geometric Clustering ⭐ NEW   → mask_clusters.json
Phase 6: Visualization ⭐ NEW              → clusters_3d.ply
Phase 7: Measurement (TBD)                → cluster_measurements.csv
```

---

## 전제조건

### 완료되어야 할 Phase들

✅ **Phase 0: SfM** - Sparse reconstruction
```bash
python -m src.pipeline sfm --config configs/sparse_mode.yaml
```

출력:
- `data/sfm/sparse/0/cameras.bin`
- `data/sfm/sparse/0/images.bin`
- `data/sfm/sparse/0/points3D.bin`
- `data/sfm/poses.json`

---

✅ **Phase 1: YOLO Inference**
```bash
python -m src.pipeline infer --config configs/sparse_mode.yaml
```

출력:
- `data/yolo_masks/*.json` - YOLO mask polygons

---

✅ **Phase 2: Pixel Calibration** (Optional, for measurement)
```bash
python -m src.pixel_calibration \
  --rgb-dir data/rgb \
  --depth-dir data/depth \
  --calib calib/rgb_camera_info.json \
  --output calibration/pixel_scales.json
```

출력:
- `calibration/pixel_scales.json`

---

✅ **Phase 3: Depth-RGB Alignment** ⚠️ **중요!**

**사용자 환경:** 이미 upsampled depth가 `data/depth_upsampled/`에 있음

만약 없다면:
```bash
python -m src.pipeline align --config configs/sparse_mode.yaml
```

**config 설정 (Sparse mode):**
```yaml
align:
  hole_fill: false      # Sparse aligned depth
  do_dense: false       # No JBU completion
  # → 30-50% coverage, 빠르고 정확
```

출력:
- `data/depth_upsampled/*.png` (RGB 해상도 3840×2160)

---

## 새로운 Phase 실행

### Phase 4: 2D Mask → 3D Projection ⭐

**목적:** YOLO mask polygon을 3D global 좌표로 변환

```bash
python -m src.project_masks_to_3d \
  --masks-dir data/yolo_masks \
  --poses-json data/sfm/poses.json \
  --depth-dir data/depth_upsampled \
  --calib-rgb calib/rgb_camera_info.json \
  --output outputs/masks_3d.json \
  --polygon-spacing 5 \
  --depth-search-radius 10 \
  --min-valid-points 5
```

**파라미터:**
- `--polygon-spacing`: Polygon boundary sampling 간격 (pixels)
  - 작을수록: 더 dense (정확, 느림)
  - 클수록: 더 sparse (빠름)
  - **권장: 5**

- `--depth-search-radius`: Nearest depth search 반경 (pixels)
  - Depth 없는 pixel의 주변 검색 범위
  - **권장: 10**

- `--min-valid-points`: Mask를 유효로 판정하는 최소 3D points
  - 이보다 적으면 mask skip
  - **권장: 5**

**출력:**
```json
{
  "metadata": {
    "total_input_masks": 200,
    "valid_3d_masks": 185,
    "skipped_masks": 15,
    "mean_coverage": 0.423
  },
  "masks": [
    {
      "image_id": "camera_RGB_0_0",
      "mask_id": 0,
      "class": "crack",
      "confidence": 0.87,
      "points_3d": [[x1, y1, z1], [x2, y2, z2], ...],
      "centroid_3d": [cx, cy, cz],
      "bbox_3d": {"min": [...], "max": [...]},
      "n_sampled": 40,
      "n_valid": 18,
      "n_fallback": 5,
      "coverage": 0.45,
      "fallback_ratio": 0.27
    }
  ]
}
```

**통계 확인:**
```
Projection Statistics
  Total masks: 200
  Valid 3D masks: 185
  Skipped: 15 (7.5%)
  Mean coverage: 42.3%
  Median coverage: 45.1%
  Mean fallback ratio: 18.2%
```

**Coverage 해석:**
- Mean coverage > 30%: ✅ Good
- Mean coverage < 20%: ⚠️ Consider dense alignment
- Fallback ratio > 50%: ⚠️ Check depth quality

---

### Phase 5: 3D Geometric Clustering ⭐

**목적:** 3D mask들을 geometric overlap으로 clustering (중복/분할 균열 합치기)

```bash
python -m src.cluster_masks_3d_geometric \
  --masks-3d outputs/masks_3d.json \
  --output outputs/mask_clusters.json \
  --max-centroid-distance 0.5 \
  --proximity-threshold 0.05 \
  --overlap-threshold 0.3
```

**파라미터 (매우 중요!):**

#### `--max-centroid-distance` (meters)
- **의미:** 중심점이 이 거리보다 멀면 무조건 다른 균열
- **조정:**
  - 작을수록: 보수적 clustering (cluster 많음)
  - 클수록: 적극적 merging (cluster 적음)
- **권장:**
  - 일반 균열: `0.5` (50cm)
  - 작은 균열: `0.3` (30cm)
  - 큰 균열/구조물: `1.0` (1m)

#### `--proximity-threshold` (meters)
- **의미:** 점들끼리 이 거리 이내면 "가까움"
- **조정:**
  - 균열 폭에 비례
  - 작을수록: 더 strict (cluster 많음)
  - 클수록: 더 loose (cluster 적음)
- **권장:**
  - 얇은 균열 (< 5mm): `0.03` (3cm)
  - 일반 균열: `0.05` (5cm)
  - 넓은 손상: `0.1` (10cm)

#### `--overlap-threshold` (0~1)
- **의미:** Final overlap score가 이 값보다 높으면 같은 균열
- **조정:**
  - 높을수록: 보수적 (중복 많이 남음)
  - 낮을수록: 적극적 (과병합 위험)
- **권장:**
  - 중복 최소화: `0.25`
  - 균형: `0.3`
  - 보수적: `0.4`

**파라미터 조합 예시:**

```bash
# Case 1: 중복 최소화 (적극적 병합)
--max-centroid-distance 0.5 \
--proximity-threshold 0.08 \
--overlap-threshold 0.25

# Case 2: 균형 (권장)
--max-centroid-distance 0.5 \
--proximity-threshold 0.05 \
--overlap-threshold 0.3

# Case 3: 보수적 (분리 우선)
--max-centroid-distance 0.3 \
--proximity-threshold 0.03 \
--overlap-threshold 0.4
```

**출력:**
```json
{
  "metadata": {
    "input_masks": 185,
    "output_clusters": 47,
    "reduction_percent": 74.6,
    "single_view_clusters": 8
  },
  "clusters": [
    {
      "cluster_id": 0,
      "n_masks": 5,
      "n_points": 625,
      "masks": [
        {"image_id": "camera_RGB_0_0", "mask_id": 0, "confidence": 0.87},
        {"image_id": "camera_RGB_0_1", "mask_id": 2, "confidence": 0.82},
        ...
      ],
      "centroid_3d": [x, y, z],
      "bbox_3d": {...},
      "mean_confidence": 0.85
    }
  ]
}
```

**통계 확인:**
```
Clustering Statistics
  Input masks: 185
  Output clusters: 47
  Reduction: 74.6%
  Mean masks per cluster: 3.9
  Median masks per cluster: 4
  Max masks per cluster: 12
  Single-view clusters: 8 (17.0%)
```

**해석:**
- Reduction > 60%: ✅ 효과적인 중복 제거
- Reduction < 30%: ⚠️ 너무 보수적, `overlap_threshold` 낮추기
- Single-view > 50%: ⚠️ Under-clustering, 파라미터 조정 필요

---

### Phase 6: Visualization ⭐

**목적:** Clustering 결과를 global 좌표계 PLY로 시각화

#### Option A: Cluster만 (빠름)

```bash
python -m src.visualize_clusters_3d \
  --clusters outputs/mask_clusters.json \
  --masks-3d outputs/masks_3d.json \
  --output-clusters-ply outputs/clusters_3d.ply
```

**출력:**
- `outputs/clusters_3d.ply` - Cluster points만, 각 cluster 다른 색상

**CloudCompare에서 보기:**
```bash
cloudcompare outputs/clusters_3d.ply
```

각 cluster가 다른 색으로 표시됨:
- Cluster 0: Red
- Cluster 1: Green
- Cluster 2: Blue
- ...

---

#### Option B: SFM + Cluster (combined)

```bash
python -m src.visualize_clusters_3d \
  --clusters outputs/mask_clusters.json \
  --masks-3d outputs/masks_3d.json \
  --output-clusters-ply outputs/clusters_3d.ply \
  --sfm-sparse-dir data/sfm/sparse/0 \
  --output-combined-ply outputs/sfm_with_clusters.ply
```

**출력:**
- `outputs/clusters_3d.ply` - Cluster points만
- `outputs/sfm_with_clusters.ply` - SFM (gray) + Cluster (colored)

**CloudCompare에서 보기:**
```bash
cloudcompare outputs/sfm_with_clusters.ply
```

- SFM points: Gray (배경)
- Cluster points: Colored (균열)

**Global 좌표계 확인:**
- SFM point cloud와 cluster가 정확히 같은 좌표계
- Cluster가 구조물 표면에 정확히 위치함

---

## 파라미터 튜닝 가이드

### 문제: "Too many clusters (중복 많음)"

**증상:**
```
Output clusters: 150
Reduction: 18.9%
Single-view clusters: 120 (80%)
```

**해결:**
```bash
# overlap_threshold 낮추기
--overlap-threshold 0.25  # 0.3 → 0.25

# proximity_threshold 높이기
--proximity-threshold 0.08  # 0.05 → 0.08

# max_centroid_distance 높이기
--max-centroid-distance 0.7  # 0.5 → 0.7
```

---

### 문제: "Too few clusters (과병합)"

**증상:**
```
Output clusters: 10
Reduction: 94.6%
Max masks per cluster: 45
```

**해결:**
```bash
# overlap_threshold 높이기
--overlap-threshold 0.4  # 0.3 → 0.4

# proximity_threshold 낮추기
--proximity-threshold 0.03  # 0.05 → 0.03

# max_centroid_distance 낮추기
--max-centroid-distance 0.3  # 0.5 → 0.3
```

---

### 문제: "Low coverage (depth 부족)"

**증상:**
```
Mean coverage: 15.2%
Skipped masks: 78 (39%)
```

**해결:**

1. **Depth search radius 증가:**
```bash
--depth-search-radius 20  # 10 → 20
```

2. **Min valid points 감소:**
```bash
--min-valid-points 3  # 5 → 3
```

3. **Dense alignment 고려:**
```yaml
# configs/sparse_mode.yaml
align:
  hole_fill: true
  do_dense: true  # JBU completion
```

---

## Troubleshooting

### Q1: "FileNotFoundError: depth_upsampled/..."

**원인:** Depth upsampled 없음

**해결:**
```bash
# Depth alignment 실행
python -m src.pipeline align --config configs/sparse_mode.yaml
```

---

### Q2: "No pose for camera_RGB_X_Y"

**원인:** SfM에서 해당 이미지 reconstruction 실패

**해결:**
1. SfM log 확인
2. 해당 이미지 품질 확인 (blur, low feature)
3. YOLO mask JSON 삭제 또는 skip

---

### Q3: "Skipped: 150/200 (75%)"

**원인:** Depth coverage 매우 낮음

**해결:**
1. Depth 품질 확인 (glass, reflective surface?)
2. `--depth-search-radius` 증가
3. `--min-valid-points` 감소
4. Dense alignment 사용

---

### Q4: "Clustering result looks wrong"

**해결 순서:**

1. **Visualization 확인:**
```bash
cloudcompare outputs/clusters_3d.ply
```
- Cluster가 실제로 공간적으로 가까운가?
- 색상 분리가 명확한가?

2. **Statistics 확인:**
```
Mean masks per cluster: 3.9  ← 적당함
Single-view clusters: 17%     ← 괜찮음
```

3. **파라미터 조정:** (위 튜닝 가이드 참고)

---

## 전체 Pipeline 예시

```bash
# === 전제조건 (완료되어야 함) ===
# Phase 0: SfM
python -m src.pipeline sfm --config configs/sparse_mode.yaml

# Phase 1: YOLO
python -m src.pipeline infer --config configs/sparse_mode.yaml

# Phase 2: Pixel calibration (optional)
python -m src.pixel_calibration \
  --rgb-dir data/rgb \
  --depth-dir data/depth \
  --calib calib/rgb_camera_info.json \
  --output calibration/pixel_scales.json

# Phase 3: Depth alignment (if not already done)
# 사용자의 경우: data/depth_upsampled/ 이미 있음, skip!

# === 새로운 Pipeline ===
# Phase 4: 3D Projection
python -m src.project_masks_to_3d \
  --masks-dir data/yolo_masks \
  --poses-json data/sfm/poses.json \
  --depth-dir data/depth_upsampled \
  --calib-rgb calib/rgb_camera_info.json \
  --output outputs/masks_3d.json \
  --polygon-spacing 5 \
  --depth-search-radius 10

# Phase 5: Clustering
python -m src.cluster_masks_3d_geometric \
  --masks-3d outputs/masks_3d.json \
  --output outputs/mask_clusters.json \
  --max-centroid-distance 0.5 \
  --proximity-threshold 0.05 \
  --overlap-threshold 0.3

# Phase 6: Visualization
python -m src.visualize_clusters_3d \
  --clusters outputs/mask_clusters.json \
  --masks-3d outputs/masks_3d.json \
  --output-clusters-ply outputs/clusters_3d.ply \
  --sfm-sparse-dir data/sfm/sparse/0 \
  --output-combined-ply outputs/sfm_with_clusters.ply

# === 결과 확인 ===
echo "✅ Pipeline complete!"
echo "Clusters: $(jq '.metadata.output_clusters' outputs/mask_clusters.json)"
echo "Reduction: $(jq '.metadata.reduction_percent' outputs/mask_clusters.json)%"

# CloudCompare로 시각화
cloudcompare outputs/sfm_with_clusters.ply
```

**예상 소요 시간:**
- Phase 4 (Projection): ~1분
- Phase 5 (Clustering): ~30초
- Phase 6 (Visualization): ~10초
- **Total: ~2분** (기존 Dense 방식 60분 대비 **30배 빠름**)

---

## Output Files

```
outputs/
├── masks_3d.json              # 3D transformed masks
├── mask_clusters.json         # Clustered masks
├── clusters_3d.ply            # Cluster visualization (colored)
└── sfm_with_clusters.ply      # Combined: SFM (gray) + Clusters (colored)

calibration/
└── pixel_scales.json          # Pixel-to-mm scales (for measurement)
```

---

## Next Steps

Phase 7: **Measurement** (TBD)

Cluster별 균열 측정:
- 길이 (mm)
- 폭 (mm)
- 면적 (mm²)
- 방향 (degrees)

구현 예정!

---

## 비교: 3가지 방식

| Aspect | Dense Point Cloud | Simple (No Clustering) | **Sparse Clustering** ⭐ |
|--------|-------------------|------------------------|------------------------|
| **SfM** | Dense (30분) | Sparse (5분) | Sparse (5분) |
| **Point 수** | 10M+ | 100K | 20K-100K |
| **Clustering** | DBSCAN (point density) | ❌ 없음 | Geometric overlap |
| **중복 제거** | ✅ | ❌ | ✅ |
| **속도** | 60분 | 10분 | **7분** |
| **정확도** | Point density 의존 | N/A | **Geometric, robust** |

**Sparse Clustering이 최선입니다!** 🎯

---

## 참고 자료

- [SPARSE_MODE_UNDERSTANDING.md](SPARSE_MODE_UNDERSTANDING.md) - 설계 문서
- [DENSE_PIPELINE.md](DENSE_PIPELINE.md) - Dense 방식 (비교용)
- [SIMPLE_PIPELINE.md](SIMPLE_PIPELINE.md) - Simple 방식 (비교용)

---

**문제 발생 시:**
- GitHub Issues
- Log 파일 확인 (`--log-level DEBUG`)
- CloudCompare로 중간 결과 시각화
