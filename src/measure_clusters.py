#!/usr/bin/env python3
"""
Measure crack clusters using 3D segmentation + 2D pixel measurement.

Pipeline:
1. 3D: Segment cluster along principal axis
2. 3D: For each segment, find best covering mask
3. 2D: Measure length/width using skeleton method
4. Aggregate measurements across segments

Inputs:
- crack_clusters.json (from DBSCAN clustering)
- crack_points.json (from point cloud overlay)
- YOLO masks directory
- pixel_mm_ratio.json (from calibration)

Output:
- cluster_measurements.json
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from skimage.morphology import skeletonize
from skimage.draw import polygon as draw_polygon

logger = logging.getLogger(__name__)


def setup_logging(level: str = 'INFO'):
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# =============================================================================
# 3D Functions: Cluster Segmentation
# =============================================================================

def compute_principal_axis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute principal axis using PCA.

    Returns:
        (centroid, principal_direction)
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    try:
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_idx = np.argmax(eigenvalues)
        principal_dir = eigenvectors[:, principal_idx]
        return centroid, principal_dir
    except:
        return centroid, np.array([1.0, 0.0, 0.0])


def segment_cluster_3d(
    cluster_points: List[Dict],
    n_segments: int = 5,
    adaptive: bool = True
) -> List[Dict]:
    """
    Segment cluster along principal axis.

    Args:
        cluster_points: List of point dicts with 'point_id', 'xyz', 'source_masks'
        n_segments: Number of segments (if not adaptive)
        adaptive: Use adaptive segmentation based on point distribution

    Returns:
        List of segment dicts with 'point_ids' and 'source_masks'
    """
    if len(cluster_points) < 2:
        # Single segment for very small clusters
        return [{
            'segment_id': 0,
            'point_ids': [p['point_id'] for p in cluster_points],
            'points': cluster_points
        }]

    # Get 3D coordinates
    xyz = np.array([p['xyz'] for p in cluster_points])

    # Compute principal axis
    centroid, principal_dir = compute_principal_axis(xyz)

    # Project points onto principal axis
    projections = np.dot(xyz - centroid, principal_dir)

    # Determine segment boundaries
    min_proj, max_proj = projections.min(), projections.max()

    if adaptive and len(cluster_points) > 20:
        # Adaptive: adjust n_segments based on crack extent
        extent = max_proj - min_proj
        # Roughly one segment per 0.05m (5cm)
        n_segments = max(2, min(10, int(extent / 0.05) + 1))

    # Create segment boundaries
    boundaries = np.linspace(min_proj, max_proj, n_segments + 1)

    # Assign points to segments
    segments = []
    for i in range(n_segments):
        low, high = boundaries[i], boundaries[i + 1]

        # Include boundary points in segment
        if i == n_segments - 1:
            mask = (projections >= low) & (projections <= high)
        else:
            mask = (projections >= low) & (projections < high)

        segment_points = [p for p, m in zip(cluster_points, mask) if m]

        if segment_points:
            segments.append({
                'segment_id': i,
                'point_ids': [p['point_id'] for p in segment_points],
                'points': segment_points
            })

    return segments


def find_best_mask_for_segment(segment: Dict) -> Optional[Tuple[str, int]]:
    """
    Find the best (image_id, mask_id) for a segment based on point contribution.

    Returns:
        (image_id, mask_id) or None if no masks found
    """
    # Count contributions from each (image_id, mask_id)
    mask_contributions = defaultdict(int)

    for point in segment['points']:
        for source in point.get('source_masks', []):
            key = (source['image_id'], source['mask_id'])
            mask_contributions[key] += 1

    if not mask_contributions:
        return None

    # Find the mask with most contributions
    best_mask = max(mask_contributions.items(), key=lambda x: x[1])
    return best_mask[0]


# =============================================================================
# 2D Functions: Skeleton-based Measurement
# =============================================================================

def load_mask_polygon(
    masks_dir: Path,
    image_id: str,
    mask_id: int
) -> Optional[List[Tuple[float, float]]]:
    """
    Load mask polygon from YOLO masks JSON.

    Returns:
        List of (x, y) polygon points or None
    """
    # Try different filename patterns
    patterns = [
        f"{image_id}.json",
        f"{image_id}.png.json",
    ]

    for pattern in patterns:
        json_path = masks_dir / pattern
        if json_path.exists():
            with open(json_path) as f:
                masks_data = json.load(f)

            # Get masks array from the JSON structure
            masks_list = None
            if isinstance(masks_data, dict) and 'masks' in masks_data:
                # Format: {"masks": [...]}
                masks_list = masks_data['masks']
            elif isinstance(masks_data, list):
                # Format: [...]
                masks_list = masks_data

            if masks_list and mask_id < len(masks_list):
                mask = masks_list[mask_id]
                return mask.get('polygon', [])

    return None


def polygon_to_binary_mask(
    polygon: List[Tuple[float, float]],
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert polygon to binary mask.

    Args:
        polygon: List of (x, y) points
        image_shape: (height, width)

    Returns:
        Binary mask array
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    if len(polygon) < 3:
        return mask

    # Convert to row, col format for skimage
    cols = np.array([p[0] for p in polygon])
    rows = np.array([p[1] for p in polygon])

    # Clip to image bounds
    cols = np.clip(cols, 0, width - 1)
    rows = np.clip(rows, 0, height - 1)

    # Draw filled polygon
    rr, cc = draw_polygon(rows, cols, shape=(height, width))
    mask[rr, cc] = 1

    return mask


def calculate_skeleton_length(
    skeleton: np.ndarray,
    D: float
) -> float:
    """
    Calculate crack length from skeleton using direction-based method.

    Args:
        skeleton: Binary skeleton image (1px thick)
        D: pixel_mm_ratio (mm per pixel)

    Returns:
        Length in mm
    """
    # Find skeleton pixels
    rows, cols = np.where(skeleton > 0)

    if len(rows) < 2:
        return 0.0

    # Create set for fast lookup
    skeleton_pixels = set(zip(rows, cols))

    # Count connections by direction
    # Directions: (dr, dc) -> type
    # Horizontal/Vertical: distance = D
    # Diagonal: distance = sqrt(2) * D

    total_length = 0.0
    visited_edges = set()

    for r, c in skeleton_pixels:
        # Check 8-connected neighbors
        neighbors = [
            (r-1, c),   # up (V)
            (r+1, c),   # down (V)
            (r, c-1),   # left (H)
            (r, c+1),   # right (H)
            (r-1, c-1), # top-left (D)
            (r-1, c+1), # top-right (D)
            (r+1, c-1), # bottom-left (D)
            (r+1, c+1), # bottom-right (D)
        ]

        for i, (nr, nc) in enumerate(neighbors):
            if (nr, nc) in skeleton_pixels:
                # Create edge key (sorted to avoid double counting)
                edge = tuple(sorted([(r, c), (nr, nc)]))

                if edge not in visited_edges:
                    visited_edges.add(edge)

                    # Determine distance based on direction
                    if i < 4:  # H or V
                        total_length += D
                    else:  # Diagonal
                        total_length += D * np.sqrt(2)

    return total_length


def calculate_skeleton_width(
    skeleton: np.ndarray,
    binary_mask: np.ndarray,
    D: float,
    sample_interval: int = 5
) -> Tuple[float, float]:
    """
    Calculate crack width by measuring perpendicular to skeleton.

    Args:
        skeleton: Binary skeleton image
        binary_mask: Original binary mask
        D: pixel_mm_ratio
        sample_interval: Sample every N skeleton pixels

    Returns:
        (average_width_mm, max_width_mm)
    """
    # Find skeleton pixels
    rows, cols = np.where(skeleton > 0)

    if len(rows) < 3:
        return 0.0, 0.0

    # Sample skeleton pixels
    n_pixels = len(rows)
    sample_indices = range(0, n_pixels, sample_interval)

    widths = []

    for idx in sample_indices:
        r, c = rows[idx], cols[idx]

        # Estimate local direction from nearby skeleton pixels
        # Use a small window
        window = 3
        nearby_rows = rows[max(0, idx-window):min(n_pixels, idx+window+1)]
        nearby_cols = cols[max(0, idx-window):min(n_pixels, idx+window+1)]

        if len(nearby_rows) < 2:
            continue

        # Fit line to get direction
        dr = nearby_rows[-1] - nearby_rows[0]
        dc = nearby_cols[-1] - nearby_cols[0]

        # Perpendicular direction
        length = np.sqrt(dr**2 + dc**2)
        if length < 1e-6:
            continue

        # Normal vector (perpendicular to skeleton direction)
        nr, nc = -dc / length, dr / length

        # Measure width along normal direction
        width_pixels = measure_width_along_normal(
            binary_mask, r, c, nr, nc
        )

        if width_pixels > 0:
            widths.append(width_pixels * D)

    if not widths:
        return 0.0, 0.0

    return np.mean(widths), np.max(widths)


def measure_width_along_normal(
    mask: np.ndarray,
    r: int, c: int,
    nr: float, nc: float,
    max_distance: int = 100
) -> int:
    """
    Measure mask width along normal direction from point (r, c).

    Returns:
        Width in pixels
    """
    height, width = mask.shape

    # Search in both directions along normal
    positive_dist = 0
    negative_dist = 0

    # Positive direction
    for d in range(1, max_distance):
        new_r = int(round(r + d * nr))
        new_c = int(round(c + d * nc))

        if not (0 <= new_r < height and 0 <= new_c < width):
            break

        if mask[new_r, new_c] == 0:
            break

        positive_dist = d

    # Negative direction
    for d in range(1, max_distance):
        new_r = int(round(r - d * nr))
        new_c = int(round(c - d * nc))

        if not (0 <= new_r < height and 0 <= new_c < width):
            break

        if mask[new_r, new_c] == 0:
            break

        negative_dist = d

    # Total width = positive + negative + 1 (center pixel)
    return positive_dist + negative_dist + 1


def measure_segment_2d(
    masks_dir: Path,
    image_id: str,
    mask_id: int,
    pixel_mm_ratio: float,
    image_shape: Tuple[int, int] = (2160, 3840)
) -> Dict:
    """
    Measure a segment using 2D skeleton method.

    Returns:
        Dict with length_mm, avg_width_mm, max_width_mm
    """
    # Load mask polygon
    polygon = load_mask_polygon(masks_dir, image_id, mask_id)

    if not polygon:
        logger.warning(f"Could not load mask: {image_id}, mask {mask_id}")
        return {'length_mm': 0, 'avg_width_mm': 0, 'max_width_mm': 0}

    # Convert to binary mask
    binary_mask = polygon_to_binary_mask(polygon, image_shape)

    if binary_mask.sum() == 0:
        return {'length_mm': 0, 'avg_width_mm': 0, 'max_width_mm': 0}

    # Skeletonize
    skeleton = skeletonize(binary_mask > 0)

    # Calculate length
    D = pixel_mm_ratio
    length_mm = calculate_skeleton_length(skeleton, D)

    # Calculate width
    avg_width_mm, max_width_mm = calculate_skeleton_width(
        skeleton, binary_mask, D
    )

    return {
        'length_mm': round(length_mm, 2),
        'avg_width_mm': round(avg_width_mm, 2),
        'max_width_mm': round(max_width_mm, 2)
    }


# =============================================================================
# Main Measurement Function
# =============================================================================

def measure_cluster(
    cluster: Dict,
    crack_points_lookup: Dict[int, Dict],
    masks_dir: Path,
    pixel_mm_ratios: Dict[str, float],
    image_shape: Tuple[int, int],
    n_segments: int = 5
) -> Dict:
    """
    Measure a single cluster.

    Args:
        cluster: Cluster dict from crack_clusters.json
        crack_points_lookup: point_id -> point dict
        masks_dir: Path to YOLO masks
        pixel_mm_ratios: image_id -> pixel_mm_ratio
        image_shape: (height, width)
        n_segments: Number of segments

    Returns:
        Measurement dict
    """
    cluster_id = cluster['cluster_id']
    point_ids = cluster['point_ids']

    # Get full point data
    cluster_points = []
    for pid in point_ids:
        if pid in crack_points_lookup:
            cluster_points.append(crack_points_lookup[pid])

    if not cluster_points:
        logger.warning(f"Cluster {cluster_id}: No points found")
        return {
            'cluster_id': cluster_id,
            'total_length_mm': 0,
            'avg_width_mm': 0,
            'max_width_mm': 0,
            'n_segments': 0,
            'segments': []
        }

    # Segment cluster in 3D
    segments = segment_cluster_3d(cluster_points, n_segments)

    # Measure each segment in 2D
    segment_measurements = []

    for segment in segments:
        # Find best mask for this segment
        best_mask = find_best_mask_for_segment(segment)

        if not best_mask:
            logger.debug(f"Segment {segment['segment_id']}: No mask found")
            continue

        image_id, mask_id = best_mask

        # Get pixel_mm_ratio for this image
        # Try different key formats (handle camera_RGB_ prefix)
        D = None
        # Extract timestamp part from image_id (e.g., camera_RGB_1761702052_213355008 -> 1761702052_213355008)
        timestamp_key = image_id
        for prefix in ['camera_RGB_', 'camera_DPT_']:
            if image_id.startswith(prefix):
                timestamp_key = image_id[len(prefix):]
                break

        for key in [image_id, timestamp_key, f"{image_id}.png", image_id.replace('.png', '')]:
            if key in pixel_mm_ratios:
                D = pixel_mm_ratios[key]
                break

        if D is None:
            logger.warning(f"No pixel_mm_ratio for {image_id}")
            continue

        # Measure in 2D
        measurement = measure_segment_2d(
            masks_dir, image_id, mask_id, D, image_shape
        )

        measurement['segment_id'] = segment['segment_id']
        measurement['image_id'] = image_id
        measurement['mask_id'] = mask_id
        measurement['n_points'] = len(segment['point_ids'])

        segment_measurements.append(measurement)

    # Aggregate measurements
    if segment_measurements:
        total_length = sum(m['length_mm'] for m in segment_measurements)
        avg_widths = [m['avg_width_mm'] for m in segment_measurements if m['avg_width_mm'] > 0]
        max_widths = [m['max_width_mm'] for m in segment_measurements if m['max_width_mm'] > 0]

        avg_width = np.mean(avg_widths) if avg_widths else 0
        max_width = max(max_widths) if max_widths else 0
    else:
        total_length = 0
        avg_width = 0
        max_width = 0

    return {
        'cluster_id': cluster_id,
        'total_length_mm': round(total_length, 2),
        'avg_width_mm': round(avg_width, 2),
        'max_width_mm': round(max_width, 2),
        'n_segments': len(segment_measurements),
        'n_points': len(cluster_points),
        'segments': segment_measurements
    }


def run_measurement(
    clusters_json: str,
    crack_points_json: str,
    masks_dir: str,
    pixel_mm_json: str,
    output_json: str,
    image_width: int = 3840,
    image_height: int = 2160,
    n_segments: int = 5
):
    """
    Run measurement on all clusters.
    """
    logger.info("=" * 80)
    logger.info("Cluster Measurement")
    logger.info("=" * 80)

    # Load inputs
    logger.info(f"Loading clusters: {clusters_json}")
    with open(clusters_json) as f:
        clusters_data = json.load(f)

    logger.info(f"Loading crack points: {crack_points_json}")
    with open(crack_points_json) as f:
        crack_points_data = json.load(f)

    logger.info(f"Loading pixel_mm_ratio: {pixel_mm_json}")
    with open(pixel_mm_json) as f:
        pixel_mm_data = json.load(f)

    # Build lookup tables
    crack_points_lookup = {
        p['point_id']: p for p in crack_points_data['points']
    }

    # pixel_mm_ratios: handle different formats from pixel_calibration.py
    if 'images' in pixel_mm_data:
        # Format: {"images": [{"image_id": ..., "pixel_mm_ratio": ...}]}
        pixel_mm_ratios = {
            img['image_id'].replace('.png', ''): img['pixel_mm_ratio']
            for img in pixel_mm_data['images']
        }
    else:
        # Format from pixel_calibration.py: {"image_id": {"mean_scale_mm": ...}}
        pixel_mm_ratios = {}
        for key, value in pixel_mm_data.items():
            if isinstance(value, dict) and 'mean_scale_mm' in value:
                # pixel_calibration.py format
                clean_key = key.replace('.png', '')
                pixel_mm_ratios[clean_key] = value['mean_scale_mm']
            elif isinstance(value, (int, float)):
                # Simple format: {"image_id": ratio}
                clean_key = key.replace('.png', '')
                pixel_mm_ratios[clean_key] = value

    masks_path = Path(masks_dir)
    image_shape = (image_height, image_width)

    clusters = clusters_data.get('clusters', [])
    logger.info(f"Measuring {len(clusters)} clusters...")

    # Measure each cluster
    measurements = []

    for cluster in clusters:
        measurement = measure_cluster(
            cluster,
            crack_points_lookup,
            masks_path,
            pixel_mm_ratios,
            image_shape,
            n_segments
        )
        measurements.append(measurement)

        logger.info(
            f"Cluster {measurement['cluster_id']}: "
            f"L={measurement['total_length_mm']:.1f}mm, "
            f"W_avg={measurement['avg_width_mm']:.2f}mm, "
            f"W_max={measurement['max_width_mm']:.2f}mm"
        )

    # Sort by total length (largest first)
    measurements.sort(key=lambda x: x['total_length_mm'], reverse=True)

    # Output
    output_data = {
        'metadata': {
            'n_clusters': len(measurements),
            'n_segments_per_cluster': n_segments,
            'image_shape': [image_height, image_width]
        },
        'measurements': measurements
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nSaved measurements: {output_json}")
    logger.info("=" * 80)

    return measurements


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Measure crack clusters using 3D segmentation + 2D pixel measurement'
    )

    parser.add_argument('--clusters', required=True,
                       help='Input crack_clusters.json')
    parser.add_argument('--crack-points', required=True,
                       help='Input crack_points.json')
    parser.add_argument('--masks-dir', required=True,
                       help='YOLO masks directory')
    parser.add_argument('--pixel-mm', required=True,
                       help='pixel_mm_ratio.json')
    parser.add_argument('--output', required=True,
                       help='Output cluster_measurements.json')
    parser.add_argument('--image-width', type=int, default=3840)
    parser.add_argument('--image-height', type=int, default=2160)
    parser.add_argument('--n-segments', type=int, default=5,
                       help='Number of segments per cluster (default: 5)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        measurements = run_measurement(
            args.clusters,
            args.crack_points,
            args.masks_dir,
            args.pixel_mm,
            args.output,
            args.image_width,
            args.image_height,
            args.n_segments
        )

        if measurements:
            print(f"\n✅ Measurement complete!")
            print(f"   Total clusters: {len(measurements)}")
            if measurements:
                largest = measurements[0]
                print(f"   Largest crack: {largest['total_length_mm']:.1f}mm length")
            print(f"   Output: {args.output}")
        else:
            print(f"\n⚠️ No measurements generated")

    except Exception as e:
        logger.error(f"Measurement failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
