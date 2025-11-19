# Sparse Mode Understanding - Direct 3D Mask Projection

## 문제: 기존 Point Cloud 기반 방식의 한계

### Dense Point Cloud 방식 (DENSE_PIPELINE.md)
```
RGB images → Dense reconstruction (30분)
                ↓
        Dense Point Cloud (10M+ points)
                ↓
        YOLO mask overlay on points (느림)
                ↓
        DBSCAN clustering (point density 의존)
```

**문제점:**
- Dense reconstruction 너무 느림
- Point density가 불균일 (texture 없는 균열 → points 적음)
- DBSCAN이 point density에 의존적 → 불안정

### Sparse Point Cloud 방식 (SIMPLE_PIPELINE.md)
```
RGB images → Sparse SfM (빠름)
                ↓
        Sparse Point Cloud (feature points만)
                ↓
        YOLO mask overlay on sparse points (점 적음!)
                ↓
        ❌ Clustering 없음 → 중복/분할 균열 처리 안 됨
```

**문제점:**
- Feature points가 sparse → mask coverage 낮음
- Clustering 단계 자체가 없음
- 중복 균열, 분할 균열 처리 불가

---

## 해결책: Direct 3D Mask Projection (Point Cloud 버리기!)

### 핵심 아이디어

**Point Cloud에 의존하지 말고, YOLO mask polygon을 직접 3D로 변환**

```
RGB images → Sparse SfM (poses만!)
                ↓
            poses.json
                ↓
YOLO masks → Direct 3D projection (polygon만)
    (각 mask = 100-500 points)
                ↓
3D Mask Clustering (geometric overlap)
                ↓
Cluster별 measurement
```

**장점:**
- ✅ Point cloud density 무관
- ✅ 빠름 (Dense reconstruction 불필요)
- ✅ 정확함 (Direct geometric calculation)
- ✅ Robust (texture-less regions도 OK)

---

## Phase별 상세 설명

### Phase 0: Sparse SfM
```bash
python -m src.pipeline sfm --config configs/sparse_mode.yaml
```

**설정:**
```yaml
sfm:
  dense: false  # ← Sparse만!
  quality: high
```

**출력:**
- `data/sfm/sparse/0/cameras.bin` - Camera intrinsics
- `data/sfm/sparse/0/images.bin` - Camera poses
- `data/sfm/poses.json` - Parsed poses

---

### Phase 1: YOLO Inference
```bash
python -m src.pipeline infer --config configs/sparse_mode.yaml
```

**출력:**
- `data/yolo_masks/*.json` - Crack masks per image

---

### Phase 2: Pixel Calibration (Optional, for measurement)
```bash
python -m src.pixel_calibration \
  --rgb-dir data/rgb \
  --depth-dir data/depth \
  --calib calib/rgb_camera_info.json \
  --output calibration/pixel_scales.json
```

---

### Phase 3: Direct 3D Mask Projection ⭐ **NEW**

```bash
python -m src.project_masks_to_3d \
  --masks-dir data/yolo_masks \
  --poses-json data/sfm/poses.json \
  --depth-dir outputs/aligned_depth \
  --calib-rgb calib/rgb_camera_info.json \
  --output outputs/masks_3d.json
```

**알고리즘:**

```python
For each image:
    # Load camera pose
    R, t = poses[image_id]  # World to Camera
    K = camera_intrinsics

    For each YOLO mask:
        # 1. Sample polygon (sparse sampling)
        polygon_2d = mask.polygon  # [(u1,v1), (u2,v2), ...]
        sampled_pixels = sample_polygon_boundary(polygon_2d, spacing=5)
        # spacing=5 → 5px마다 1개 point
        # 예: 200px 둘레 → 40 points만

        # 2. Backproject to 3D using depth
        points_3d_world = []
        for (u, v) in sampled_pixels:
            depth = aligned_depth_map[v, u]

            if depth > 0:
                # Camera coordinates (backproject)
                X_cam = (u - cx) * depth / fx
                Y_cam = (v - cy) * depth / fy
                Z_cam = depth

                # World coordinates (transform)
                P_cam = np.array([X_cam, Y_cam, Z_cam])
                P_world = R.T @ (P_cam - t)  # Camera → World

                points_3d_world.append(P_world)

        # 3. Store as 3D mask
        mask_3d = {
            'image_id': image_id,
            'mask_id': mask_id,
            'class': mask.class_name,
            'confidence': mask.score,
            'points_3d': points_3d_world,  # List of [x, y, z]
            'centroid_3d': np.mean(points_3d_world, axis=0),
            'bbox_3d': {
                'min': np.min(points_3d_world, axis=0),
                'max': np.max(points_3d_world, axis=0)
            },
            'n_points': len(points_3d_world)
        }

        masks_3d.append(mask_3d)
```

**출력:**
- `outputs/masks_3d.json` - 3D transformed masks

**출력 포맷:**
```json
{
  "masks": [
    {
      "image_id": "camera_RGB_0_0",
      "mask_id": 0,
      "class": "crack",
      "confidence": 0.87,
      "points_3d": [[x1, y1, z1], [x2, y2, z2], ...],
      "centroid_3d": [cx, cy, cz],
      "bbox_3d": {
        "min": [xmin, ymin, zmin],
        "max": [xmax, ymax, zmax]
      },
      "n_points": 125
    }
  ]
}
```

**Point 수:**
- 각 mask당 100-500 points (polygon boundary만)
- 전체 이미지 100장, mask 200개 → **총 20,000-100,000 points**
- Dense point cloud (10M+) 대비 **100배 적음!**

---

### Phase 4: 3D Mask Clustering ⭐ **NEW**

```bash
python -m src.cluster_masks_3d \
  --masks-3d outputs/masks_3d.json \
  --output outputs/mask_clusters.json \
  --max-centroid-distance 0.5 \
  --proximity-threshold 0.05 \
  --overlap-threshold 0.3
```

**알고리즘:**

```python
def compute_3d_overlap(mask_3d, cluster, config):
    """
    Fast 3D overlap computation using geometric properties

    Returns:
        overlap_score: 0.0 ~ 1.0
    """
    # 1. Quick rejection: Centroid distance
    centroid_dist = np.linalg.norm(
        mask_3d['centroid_3d'] - cluster.centroid_3d
    )

    if centroid_dist > config.max_centroid_distance:
        return 0.0  # 너무 멀리 떨어짐 (예: 50cm 이상)

    # 2. BBox overlap (3D IoU)
    bbox_iou = compute_bbox_iou_3d(
        mask_3d['bbox_3d'],
        cluster.bbox_3d
    )

    if bbox_iou < 0.05:
        return 0.0  # BBox도 안 겹침

    # 3. Point-level verification (sampling)
    # Mask의 일부 points만 사용 (속도 향상)
    sample_points = mask_3d['points_3d'][::5]  # 5개 중 1개
    cluster_points = cluster.get_sample_points(max_points=500)

    # Pairwise distance matrix
    distances = np.linalg.norm(
        sample_points[:, None, :] - cluster_points[None, :, :],
        axis=2
    )
    min_distances = distances.min(axis=1)

    # Close ratio: 몇 %가 proximity_threshold 이내인가?
    close_ratio = np.mean(min_distances < config.proximity_threshold)

    # Final score: weighted combination
    score = 0.3 * bbox_iou + 0.7 * close_ratio

    return score


# Main clustering loop
clusters = []

for mask_3d in masks_3d:
    best_cluster = None
    best_score = 0.0

    # Find best matching cluster
    for cluster in clusters:
        score = compute_3d_overlap(mask_3d, cluster, config)

        if score > best_score:
            best_score = score
            best_cluster = cluster

    # Merge or create new
    if best_score > config.overlap_threshold:
        best_cluster.add_mask(mask_3d)
    else:
        clusters.append(Cluster([mask_3d]))
```

**파라미터:**
```yaml
clustering_3d:
  max_centroid_distance: 0.5    # 50cm 이상 떨어지면 다른 균열
  proximity_threshold: 0.05      # 5cm 이내면 "가까움"
  overlap_threshold: 0.3         # Score 0.3 이상이면 같은 균열
```

**출력:**
- `outputs/mask_clusters.json` - Clustered masks

**출력 포맷:**
```json
{
  "metadata": {
    "total_masks": 200,
    "total_clusters": 45,
    "config": {...}
  },
  "clusters": [
    {
      "cluster_id": 0,
      "n_masks": 5,
      "n_points": 625,
      "mask_ids": [
        {"image_id": "camera_RGB_0_0", "mask_id": 0},
        {"image_id": "camera_RGB_0_1", "mask_id": 2},
        ...
      ],
      "centroid_3d": [cx, cy, cz],
      "bbox_3d": {...}
    }
  ]
}
```

---

### Phase 5: Cluster Measurement

```bash
python -m src.measure_mask_clusters \
  --clusters outputs/mask_clusters.json \
  --masks-3d outputs/masks_3d.json \
  --pixel-scales calibration/pixel_scales.json \
  --output outputs/cluster_measurements.csv
```

**측정 방법:**

#### Option A: 3D Direct Measurement
```python
# 모든 mask의 3D points 병합
all_points_3d = concatenate([m.points_3d for m in cluster.masks])

# 3D skeleton
skeleton_3d = skeletonize_3d(all_points_3d)
length_3d = compute_skeleton_length_3d(skeleton_3d)

# 3D width (perpendicular distance)
width_3d = compute_perpendicular_width_3d(all_points_3d, skeleton_3d)
```

#### Option B: Best View Selection + 2D Measurement
```python
# Cluster를 가장 잘 보는 view 선택
best_view = select_best_view(cluster)

# 해당 view에서 2D measurement
length_mm = measure_2d_with_calibration(
    best_view.image,
    best_view.mask,
    pixel_scales[best_view.image_id]
)
```

#### Option C: Multi-view Aggregation (권장)
```python
measurements = []

for mask in cluster.masks:
    # 각 view에서 2D 측정
    length_mm = measure_2d(
        mask.image_id,
        mask.mask_id,
        pixel_scales
    )
    measurements.append(length_mm)

# Median aggregation (robust)
final_length_mm = np.median(measurements)
final_length_std = np.std(measurements)
```

**출력:**
- `outputs/cluster_measurements.csv`

**출력 포맷:**
```csv
cluster_id,n_masks,n_views,length_mm,width_mm,length_std,width_std,confidence_mean,...
0,5,5,1234.5,2.3,45.2,0.3,0.87,...
1,3,3,567.8,1.8,23.1,0.2,0.82,...
```

---

## 비교: 3가지 방식

| Aspect | Dense Point Cloud | Sparse Point Cloud | **Direct 3D Mask** ⭐ |
|--------|-------------------|--------------------|--------------------|
| **SfM** | Dense (30분) | Sparse (5분) | Sparse (5분) |
| **Point 수** | 10M+ | 100K | 20K-100K (mask만) |
| **Clustering** | DBSCAN on points | ❌ 없음 | Geometric overlap |
| **속도** | 매우 느림 | 빠름 | **빠름** |
| **정확도** | Point density 의존 | N/A | **Geometric, robust** |
| **중복 제거** | ✅ | ❌ | ✅ |
| **Memory** | 높음 (GB) | 낮음 | **낮음 (MB)** |

---

## 장점 요약

### 1. **속도**
- Dense reconstruction 불필요 → **30분 → 5분**
- Mask polygon만 처리 → 훨씬 적은 points
- Geometric overlap 계산 빠름

### 2. **정확도**
- Point cloud density 무관
- Direct geometric calculation
- Texture-less regions도 처리 가능

### 3. **Robustness**
- Feature point 분포에 영향 안 받음
- Depth 기반 직접 변환 → reliable
- Multi-view aggregation으로 robust measurement

### 4. **Memory 효율**
- Point cloud 저장 불필요
- Mask JSON만 처리 → MB 단위

---

## 파라미터 튜닝 가이드

### Polygon Sampling
```yaml
polygon_sampling:
  spacing: 5  # 5px마다 1 point
  # 작을수록: 더 dense → 정확하지만 느림
  # 클수록: 더 sparse → 빠르지만 detail 손실
```

**권장:** `spacing=5` (좋은 균형)

### Clustering
```yaml
clustering_3d:
  max_centroid_distance: 0.5    # 균열 크기에 따라 조정
    # 작을수록: 보수적 clustering (더 많은 cluster)
    # 클수록: 적극적 merging (적은 cluster)

  proximity_threshold: 0.05     # 5cm
    # 균열 폭에 비례
    # 일반적으로 0.03-0.1m

  overlap_threshold: 0.3
    # 높을수록: 보수적 (중복 많이 남음)
    # 낮을수록: 적극적 (과병합 위험)
```

**균열 측정용 권장:**
```yaml
clustering_3d:
  max_centroid_distance: 0.3    # 30cm (보수적)
  proximity_threshold: 0.05     # 5cm
  overlap_threshold: 0.25       # 적극적 병합
```

---

## Troubleshooting

### Q1: "Masks have no 3D points"
**원인:** Depth 값이 없음

**해결:**
1. Depth alignment 확인 (Phase 3 전에 실행)
2. Depth unit 확인 (mm vs m)
3. Polygon sampling spacing 줄이기

### Q2: "Too many clusters (중복 많음)"
**원인:** `overlap_threshold` 너무 높음

**해결:**
```yaml
clustering_3d:
  overlap_threshold: 0.2  # 0.3 → 0.2로 낮춤
  proximity_threshold: 0.08  # 0.05 → 0.08로 증가
```

### Q3: "Too few clusters (과병합)"
**원인:** `max_centroid_distance` 너무 큼

**해결:**
```yaml
clustering_3d:
  max_centroid_distance: 0.2  # 0.5 → 0.2로 감소
  overlap_threshold: 0.4       # 0.3 → 0.4로 증가
```

### Q4: "Measurements vary a lot"
**원인:** Multi-view aggregation 문제

**해결:**
1. Median 대신 trimmed mean 사용
2. Outlier 제거 (IQR method)
3. Best view만 사용

---

## 구현 우선순위

### Phase 1 (핵심):
- [ ] `src/project_masks_to_3d.py` - YOLO mask → 3D projection
- [ ] `src/cluster_masks_3d.py` - Geometric overlap clustering
- [ ] `src/measure_mask_clusters.py` - Cluster measurement

### Phase 2 (최적화):
- [ ] Parallel processing (multi-threading)
- [ ] KDTree acceleration
- [ ] Visualization tools

### Phase 3 (고급):
- [ ] Plane fitting for crack orientation
- [ ] Branch detection for crack topology
- [ ] Confidence-weighted clustering

---

## 다음 단계

1. **코드 구현** - 위 3개 파일 생성
2. **테스트** - 샘플 데이터로 검증
3. **문서화** - 사용법 가이드 작성
4. **통합** - `src/pipeline.py`에 통합

---

## 참고 자료

- 3D Transformation: Camera pose matrices (R, t)
- BBox IoU 3D: Axis-aligned bounding box intersection
- Point-to-point distance: Euclidean distance in 3D
- Polygon sampling: Bresenham's line algorithm

---

**이 방식이 최선입니다:**
- Point cloud density 문제 해결
- 빠르고 정확함
- 구현 복잡도 적당함
- Scalable (수백 장 이미지도 OK)
