# Dense Point Cloud Pipeline with 3D Clustering

Complete pipeline for 3D crack detection, deduplication, and measurement using dense COLMAP reconstruction.

## Overview

This pipeline solves two fundamental problems in multi-view crack detection:
1. **Duplication**: Same crack detected multiple times across different images
2. **Fragmentation**: Single crack split into multiple detections

**Solution**: 3D clustering for deduplication + 2D measurement with pixel calibration for accuracy

---

## Why This Approach?

### Problem with Simple 2D Measurement

```python
# Each image measured independently
Image 1: crack A → 100mm
Image 2: crack A (same!) → 95mm    # DUPLICATE
Image 3: crack B (part 1) → 50mm   # FRAGMENT
Image 4: crack B (part 2) → 45mm   # FRAGMENT
```

**Result**: 4 detections, but only 2 physical cracks!

### Our Solution

```
3D Clustering → Identify unique cracks
      ↓
2D Measurement → Accurate mm measurements (pixel-level precision)
      ↓
Multi-view Aggregation → Robust final measurements

Result: 2 cracks with accurate measurements!
```

---

## Pipeline Phases

### Phase 0: SFM (Sparse + Dense Reconstruction)

Generate camera poses and dense 3D reconstruction.

```bash
python -m src.pipeline sfm --config configs/simple.yaml
```

**Configuration** (`configs/simple.yaml`):
```yaml
sfm:
  camera_model: 'OPENCV'  # Or PINHOLE, RADIAL, etc.
  quality: 'high'         # low, medium, high, extreme
  dense: true             # ← IMPORTANT: Enable dense reconstruction
```

**Outputs:**
- `data/sfm/sparse/0/` - Initial sparse reconstruction
  - `cameras.bin, images.bin, points3D.bin`
- `data/sfm/dense/` - Dense reconstruction
  - `fused.ply` - Dense point cloud ⭐
  - `sparse/` - Undistorted sparse reconstruction
    - `cameras.bin, images.bin` ← Used for clustering & measurement
  - `images/` - Undistorted images

**Requirements:**
- RGB images in `data/rgb/`
- COLMAP installed

---

### Phase 1: YOLO Inference

Detect cracks in 2D images.

```bash
python -m src.pipeline infer --config configs/simple.yaml
```

**Outputs:**
- `data/yolo_masks/*.json` - Crack masks for each image

**Format**:
```json
{
  "image_path": "camera_RGB_0_0.png",
  "masks": [
    {
      "class": "crack",
      "score": 0.95,
      "polygon": [[x1, y1], [x2, y2], ...]
    }
  ]
}
```

---

### Phase 2: Pixel-to-MM Calibration ⭐ **REQUIRED**

Calculate pixel scale for each image using depth ground truth.

```bash
python -m src.pixel_calibration \
  --rgb-dir data/rgb \
  --depth-dir data/depth \
  --calib calib/rgb_camera_info.json \
  --output calibration/pixel_scales.json
```

**Why Required:**
- COLMAP reconstruction has **arbitrary scale** (relative units)
- Depth ground truth provides **absolute scale**
- Each image has different mm/pixel ratio (distance-dependent)

**How it works:**
```python
# For each pixel at depth d:
pixel_mm = (d / focal_length) * 1000
```

**Outputs:**
- `calibration/pixel_scales.json`

**Format**:
```json
{
  "camera_RGB_0_0": {
    "mean_scale_mm": 0.523,
    "min_scale_mm": 0.501,
    "max_scale_mm": 0.545,
    ...
  }
}
```

**This is the foundation for accurate mm measurements!**

---

### Phase 3: Dense Point Cloud Mask Overlay

Project YOLO masks onto dense point cloud in 3D.

```bash
python -m src.point_cloud_overlay_dense \
  --dense-ply data/sfm/dense/fused.ply \
  --sparse-dir data/sfm/dense/sparse \
  --masks-dir data/yolo_masks \
  --output outputs/dense_masked_cloud.ply \
  --min-votes 1
```

**Parameters:**
- `--dense-ply`: Dense point cloud (fused.ply)
- `--sparse-dir`: Camera poses (use `data/sfm/dense/sparse` or `data/sfm/sparse/0`)
- `--min-votes`: Minimum views where point must appear as crack

**Outputs:**
- `outputs/dense_masked_cloud.ply` - Point cloud with crack points colored RED

**How it works:**
1. Load dense point cloud (millions of points)
2. For each 3D point:
   - Project to all camera views
   - Check if pixel is inside crack mask
   - Count votes
3. Color points red if votes ≥ min-votes

**Camera Models Supported:**
- SIMPLE_PINHOLE, PINHOLE
- SIMPLE_RADIAL, RADIAL
- OPENCV (with radial + tangential distortion)

---

### Phase 4: 3D Crack Clustering

Cluster crack points in 3D to identify unique physical cracks.

```bash
python -m src.cluster_cracks_3d \
  --crack-cloud outputs/dense_masked_cloud.ply \
  --sparse-dir data/sfm/dense/sparse \
  --output outputs/crack_clusters.json \
  --output-clustered-ply outputs/clustered_cracks.ply \
  --eps 0.05 \
  --min-samples 10 \
  --min-cluster-size 50
```

**Parameters:**
- `--eps`: DBSCAN epsilon - maximum distance between points (relative units)
  - Larger = merge more aggressively
  - Smaller = more conservative
  - Typical: 0.02-0.1 (arbitrary scale)
- `--min-samples`: Minimum points to form dense region
- `--min-cluster-size`: Minimum points to keep cluster
- `--min-visible-points`: Minimum points visible in image to include

**Outputs:**
- `outputs/crack_clusters.json` - Cluster metadata + image mapping
- `outputs/clustered_cracks.ply` - Visualization (different color per cluster)

**Output Format**:
```json
{
  "metadata": {...},
  "clusters": [
    {
      "cluster_id": 0,
      "n_points": 1250,
      "point_indices": [1523, 1524, ...],  // Indices in original PLY
      "visible_images": {
        "camera_RGB_0_0.png": 450,  // 450 points visible in this image
        "camera_RGB_0_1.png": 380,
        "camera_RGB_0_3.png": 420
      },
      "centroid": [x, y, z],  // Relative coordinates (for reference)
      "bbox_min": [...],
      "bbox_max": [...],
      "bbox_size": [...]
    }
  ]
}
```

**Key Points:**
- **NO mm measurements here** (arbitrary scale!)
- Maps each cluster to visible images
- Stores point indices for later use

---

### Phase 5: Cluster-based 2D Measurement ⭐ **NEW**

Measure each cluster in 2D images with pixel calibration.

```bash
python -m src.measure_clusters_2d \
  --clusters outputs/crack_clusters.json \
  --crack-cloud outputs/dense_masked_cloud.ply \
  --sparse-dir data/sfm/dense/sparse \
  --pixel-scales calibration/pixel_scales.json \
  --output outputs/cluster_measurements.csv
```

**How it works:**
```python
for cluster in clusters:
    measurements = []

    for image in cluster.visible_images:
        # 1. Project 3D points to 2D
        pixels = project(cluster.points_3d, camera_pose)

        # 2. Create 2D mask from pixels
        mask_2d = convex_hull(pixels) + dilation

        # 3. Measure in 2D (pixel level)
        skeleton = skeletonize(mask_2d)
        length_px = measure_skeleton_MST(skeleton)
        width_px = measure_perpendicular_width(skeleton)

        # 4. Convert to mm using pixel calibration
        pixel_scale = pixel_calibration[image]['mean_scale_mm']
        length_mm = length_px * pixel_scale
        width_mm = width_px * pixel_scale

        measurements.append({
            'length_mm': length_mm,
            'width_mm': width_mm
        })

    # 5. Aggregate across views (robust estimation)
    final_length_mm = median([m['length_mm'] for m in measurements])
    final_width_mm = median([m['width_mm'] for m in measurements])
```

**Outputs:**
- `outputs/cluster_measurements.csv`

**Output Format**:
```csv
cluster_id,n_points,n_views,length_mm,width_mm,length_mm_std,width_mm_std,...
0,1250,3,125.3,2.1,5.2,0.3,...
1,980,4,87.4,1.8,3.1,0.2,...
```

**Benefits:**
- Pixel-level precision (2D measurement)
- Absolute scale (pixel calibration)
- Robustness (multi-view aggregation)
- Deduplication (3D clustering)

---

## Complete Workflow Example

```bash
# Phase 0: SFM (sparse + dense)
python -m src.pipeline sfm --config configs/simple.yaml

# Phase 1: YOLO crack detection
python -m src.pipeline infer --config configs/simple.yaml

# Phase 2: Pixel calibration (REQUIRED!)
python -m src.pixel_calibration \
  --rgb-dir data/rgb \
  --depth-dir data/depth \
  --calib calib/rgb_camera_info.json \
  --output calibration/pixel_scales.json

# Phase 3: Dense point cloud overlay
python -m src.point_cloud_overlay_dense \
  --dense-ply data/sfm/dense/fused.ply \
  --sparse-dir data/sfm/dense/sparse \
  --masks-dir data/yolo_masks \
  --output outputs/dense_masked_cloud.ply \
  --min-votes 1

# Phase 4: 3D clustering (deduplication)
python -m src.cluster_cracks_3d \
  --crack-cloud outputs/dense_masked_cloud.ply \
  --sparse-dir data/sfm/dense/sparse \
  --output outputs/crack_clusters.json \
  --output-clustered-ply outputs/clustered_cracks.ply \
  --eps 0.05 \
  --min-samples 10

# Phase 5: 2D measurement
python -m src.measure_clusters_2d \
  --clusters outputs/crack_clusters.json \
  --crack-cloud outputs/dense_masked_cloud.ply \
  --sparse-dir data/sfm/dense/sparse \
  --pixel-scales calibration/pixel_scales.json \
  --output outputs/cluster_measurements.csv
```

---

## Parameter Tuning Guide

### DBSCAN Parameters (Phase 4)

**`--eps` (epsilon):**
- Physical meaning: Maximum distance for two points to be neighbors
- **Units**: Relative (COLMAP arbitrary units)
- **Too small**: Cracks split into multiple clusters
- **Too large**: Multiple cracks merged into one
- Start with: 0.05
- Adjust based on visualization of `clustered_cracks.ply`

**`--min-samples`:**
- Minimum neighbors for a "core" point
- **Too small**: More noise classified as cracks
- **Too large**: Small cracks ignored
- Start with: 10

**`--min-cluster-size`:**
- Post-processing filter
- Remove tiny clusters (likely noise)
- Start with: 50

### Testing Strategy

1. **Start conservative**: `--eps 0.03 --min-samples 20`
2. **Visualize**: Open `outputs/clustered_cracks.ply` in CloudCompare
3. **Check for issues**:
   - Same crack split? → Increase `--eps`
   - Different cracks merged? → Decrease `--eps`
   - Too much noise? → Increase `--min-cluster-size`

---

## Output Files Summary

| File | Phase | Description |
|------|-------|-------------|
| `data/sfm/sparse/0/` | 0 | Initial camera poses |
| `data/sfm/dense/fused.ply` | 0 | Dense point cloud |
| `data/sfm/dense/sparse/` | 0 | Undistorted poses |
| `data/yolo_masks/*.json` | 1 | 2D crack masks |
| `calibration/pixel_scales.json` | 2 | **Pixel-to-mm scales** ⭐ |
| `outputs/dense_masked_cloud.ply` | 3 | Crack-colored point cloud |
| `outputs/crack_clusters.json` | 4 | Cluster metadata + image mapping |
| `outputs/clustered_cracks.ply` | 4 | Visualization (colored clusters) |
| `outputs/cluster_measurements.csv` | 5 | **Final measurements** ⭐ |

---

## Key Concepts

### Why 3D Clustering + 2D Measurement?

| Aspect | 3D Clustering | 2D Measurement |
|--------|---------------|----------------|
| Purpose | Deduplication | Accuracy |
| Scale | Relative (OK) | Absolute (Required) |
| Precision | Coarse | Pixel-level |
| Input | Dense point cloud | Images + Pixel calibration |

**Best of both worlds:**
- 3D: Identify unique cracks (topology)
- 2D: Measure accurately (geometry + calibration)

### Pixel Calibration vs Scale Alignment

**Pixel Calibration (This project):**
- Per-image, per-pixel calibration
- Uses depth ground truth directly
- Distance-dependent (near objects = larger pixels)
- **Required for 2D measurement**

**Scale Alignment (Alternative):**
- Global scale factor for entire reconstruction
- Aligns COLMAP scale to metric scale
- Used for 3D measurements
- Not needed here (we measure in 2D)

---

## Troubleshooting

### "No crack points found"
- Check `--crack-color` matches Phase 3 output
- Verify `dense_masked_cloud.ply` has red points (open in CloudCompare)

### "No pixel scale for image X"
- Run Phase 2 (Pixel calibration) first
- Ensure RGB-Depth pairs match
- Check image naming conventions

### "Too many/few clusters"
- Tune `--eps` in Phase 4
- Visualize `clustered_cracks.ply` to diagnose
- Start with smaller eps, increase gradually

### "Measurements seem wrong"
- Check pixel calibration (Phase 2)
- Verify depth maps are correct (mm or m units)
- Inspect `--img-shape` parameter (default: 2160×3840)

---

## Visualization

### CloudCompare (Recommended)
```bash
cloudcompare outputs/dense_masked_cloud.ply  # Phase 3 output
cloudcompare outputs/clustered_cracks.ply    # Phase 4 output
```

### Check Results
- Phase 3: Should see RED crack points
- Phase 4: Each cluster should have different color
- If all gray: Check clustering parameters

---

## Performance Notes

- **Phase 3**: Slow for large point clouds (10M+ points, ~30 min typical)
  - Use `--max-points` for testing
- **Phase 4**: Fast (clustering is O(n log n))
- **Phase 5**: Moderate (depends on number of clusters × views)

---

## References

- COLMAP: https://colmap.github.io/
- DBSCAN: https://scikit-learn.org/stable/modules/clustering.html#dbscan
- YOLOv8: https://docs.ultralytics.com/

---

## Support

For issues:
- Check logs (use `--log-level DEBUG`)
- Verify each phase output exists
- Open issue with full command and error message
