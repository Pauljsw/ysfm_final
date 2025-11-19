"""
3D Mask Clustering using Geometric Overlap

Clusters 3D masks based on geometric properties:
- Centroid distance (fast rejection)
- BBox overlap (medium rejection)
- Point-to-point distance (final verification)

Usage:
    python -m src.cluster_masks_3d_geometric \
        --masks-3d outputs/masks_3d.json \
        --output outputs/mask_clusters.json \
        --max-centroid-distance 0.5 \
        --proximity-threshold 0.05 \
        --overlap-threshold 0.3
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Cluster:
    """3D Mask Cluster"""

    def __init__(self, initial_mask: Dict):
        """Initialize cluster with first mask"""
        self.masks = [initial_mask]
        self._update_metadata()

    def add_mask(self, mask: Dict):
        """Add mask to cluster"""
        self.masks.append(mask)
        self._update_metadata()

    def _update_metadata(self):
        """Update cluster metadata (centroid, bbox, points)"""
        # Collect all points
        all_points = []
        for mask in self.masks:
            points = np.array(mask['points_3d'])
            all_points.append(points)

        self.all_points = np.vstack(all_points)

        # Centroid
        self.centroid_3d = np.mean(self.all_points, axis=0)

        # BBox
        self.bbox_min = np.min(self.all_points, axis=0)
        self.bbox_max = np.max(self.all_points, axis=0)

        self.bbox_3d = {
            'min': self.bbox_min,
            'max': self.bbox_max
        }

    def get_sample_points(self, max_points: int = 500) -> np.ndarray:
        """Get sampled points (for performance)"""
        if len(self.all_points) <= max_points:
            return self.all_points

        # Random sampling
        indices = np.random.choice(len(self.all_points), max_points, replace=False)
        return self.all_points[indices]

    def to_dict(self, cluster_id: int) -> Dict:
        """Convert cluster to dict for JSON"""
        return {
            'cluster_id': cluster_id,
            'n_masks': len(self.masks),
            'n_points': len(self.all_points),
            'masks': [
                {
                    'image_id': m['image_id'],
                    'mask_id': m['mask_id'],
                    'confidence': m['confidence']
                }
                for m in self.masks
            ],
            'centroid_3d': self.centroid_3d.tolist(),
            'bbox_3d': {
                'min': self.bbox_min.tolist(),
                'max': self.bbox_max.tolist()
            },
            'mean_confidence': float(np.mean([m['confidence'] for m in self.masks]))
        }


def compute_bbox_iou_3d(bbox1: Dict, bbox2: Dict) -> float:
    """
    Compute 3D BBox IoU (Intersection over Union)

    Args:
        bbox1, bbox2: {'min': [x,y,z], 'max': [x,y,z]}

    Returns:
        iou: 0.0 ~ 1.0
    """
    min1 = np.array(bbox1['min'])
    max1 = np.array(bbox1['max'])
    min2 = np.array(bbox2['min'])
    max2 = np.array(bbox2['max'])

    # Intersection bbox
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)

    # Check if intersection exists
    if np.any(inter_min >= inter_max):
        return 0.0

    # Intersection volume
    inter_size = inter_max - inter_min
    inter_volume = np.prod(inter_size)

    # Union volume
    volume1 = np.prod(max1 - min1)
    volume2 = np.prod(max2 - min2)
    union_volume = volume1 + volume2 - inter_volume

    if union_volume <= 0:
        return 0.0

    # IoU
    iou = inter_volume / union_volume

    return iou


def compute_overlap_score(mask_3d: Dict, cluster: Cluster, config: Dict) -> float:
    """
    Compute geometric overlap score between mask and cluster.

    Multi-stage rejection for performance:
    1. Centroid distance (O(1)) - quick rejection
    2. BBox overlap (O(1)) - medium rejection
    3. Point-level distance (O(n*m)) - final verification

    Args:
        mask_3d: 3D mask dict
        cluster: Cluster object
        config: Configuration dict

    Returns:
        score: 0.0 ~ 1.0
    """
    mask_centroid = np.array(mask_3d['centroid_3d'])
    mask_bbox = mask_3d['bbox_3d']

    # === STAGE 1: Centroid Distance (fastest) ===
    centroid_dist = np.linalg.norm(mask_centroid - cluster.centroid_3d)

    max_centroid_distance = config.get('max_centroid_distance', 0.5)

    if centroid_dist > max_centroid_distance:
        return 0.0  # Too far apart

    # === STAGE 2: BBox Overlap (fast) ===
    bbox_iou = compute_bbox_iou_3d(mask_bbox, cluster.bbox_3d)

    min_bbox_iou = config.get('min_bbox_iou', 0.05)

    if bbox_iou < min_bbox_iou:
        return 0.0  # No bbox overlap

    # === STAGE 3: Point-level Distance (slow, but only ~5% reach here) ===
    mask_points = np.array(mask_3d['points_3d'])

    # Sample for performance
    if len(mask_points) > 100:
        sample_indices = np.random.choice(len(mask_points), 100, replace=False)
        mask_sample = mask_points[sample_indices]
    else:
        mask_sample = mask_points

    cluster_sample = cluster.get_sample_points(max_points=500)

    # Pairwise distance (broadcasting)
    # Shape: (N_mask, 1, 3) - (1, N_cluster, 3) = (N_mask, N_cluster, 3)
    distances = np.linalg.norm(
        mask_sample[:, None, :] - cluster_sample[None, :, :],
        axis=2
    )

    # Minimum distance for each mask point
    min_distances = distances.min(axis=1)

    # Close ratio
    proximity_threshold = config.get('proximity_threshold', 0.05)
    close_ratio = np.mean(min_distances < proximity_threshold)

    # === FINAL SCORE: Weighted combination ===
    bbox_weight = config.get('bbox_weight', 0.3)
    point_weight = config.get('point_weight', 0.7)

    score = bbox_weight * bbox_iou + point_weight * close_ratio

    return score


def cluster_masks_geometric(masks_3d: List[Dict], config: Optional[Dict] = None) -> List[Cluster]:
    """
    Cluster 3D masks using geometric overlap.

    Greedy clustering algorithm:
    - Process masks sequentially
    - For each mask, find best matching cluster
    - Merge if overlap score > threshold, else create new cluster

    Args:
        masks_3d: List of 3D mask dicts
        config: Configuration dict

    Returns:
        clusters: List of Cluster objects
    """
    if config is None:
        config = {
            'max_centroid_distance': 0.5,
            'min_bbox_iou': 0.05,
            'proximity_threshold': 0.05,
            'overlap_threshold': 0.3,
            'bbox_weight': 0.3,
            'point_weight': 0.7
        }

    logger.info("Starting geometric clustering...")
    logger.info(f"  Config: {config}")

    clusters = []

    for mask_3d in tqdm(masks_3d, desc="Clustering masks"):
        best_cluster = None
        best_score = 0.0

        # Find best matching cluster
        for cluster in clusters:
            score = compute_overlap_score(mask_3d, cluster, config)

            if score > best_score:
                best_score = score
                best_cluster = cluster

        # Merge or create new
        overlap_threshold = config.get('overlap_threshold', 0.3)

        if best_score > overlap_threshold:
            # Merge to existing cluster
            best_cluster.add_mask(mask_3d)
        else:
            # Create new cluster
            new_cluster = Cluster(mask_3d)
            clusters.append(new_cluster)

    logger.info(f"Clustering complete: {len(clusters)} clusters")

    return clusters


def run_clustering(
    masks_3d_json: str,
    output_json: str,
    config: Optional[Dict] = None
):
    """
    Run 3D mask clustering.

    Args:
        masks_3d_json: Input masks_3d JSON
        output_json: Output clusters JSON
        config: Configuration dict
    """
    if config is None:
        config = {
            'max_centroid_distance': 0.5,
            'min_bbox_iou': 0.05,
            'proximity_threshold': 0.05,
            'overlap_threshold': 0.3,
            'bbox_weight': 0.3,
            'point_weight': 0.7
        }

    logger.info("=" * 80)
    logger.info("3D Mask Geometric Clustering")
    logger.info("=" * 80)

    # Load 3D masks
    logger.info(f"Loading 3D masks: {masks_3d_json}")
    with open(masks_3d_json, 'r') as f:
        data = json.load(f)

    masks_3d = data['masks']
    logger.info(f"  Loaded {len(masks_3d)} 3D masks")

    # Cluster
    clusters = cluster_masks_geometric(masks_3d, config)

    # Statistics
    logger.info("=" * 80)
    logger.info("Clustering Statistics")
    logger.info("=" * 80)
    logger.info(f"  Input masks: {len(masks_3d)}")
    logger.info(f"  Output clusters: {len(clusters)}")
    logger.info(f"  Reduction: {(1 - len(clusters)/len(masks_3d))*100:.1f}%")

    # Cluster size distribution
    cluster_sizes = [len(c.masks) for c in clusters]
    logger.info(f"  Mean masks per cluster: {np.mean(cluster_sizes):.1f}")
    logger.info(f"  Median masks per cluster: {np.median(cluster_sizes):.0f}")
    logger.info(f"  Max masks per cluster: {np.max(cluster_sizes)}")

    single_view_count = sum(1 for size in cluster_sizes if size == 1)
    logger.info(f"  Single-view clusters: {single_view_count} ({single_view_count/len(clusters)*100:.1f}%)")

    # Save
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clusters_data = [c.to_dict(i) for i, c in enumerate(clusters)]

    result = {
        'metadata': {
            'input_masks': len(masks_3d),
            'output_clusters': len(clusters),
            'reduction_percent': (1 - len(clusters)/len(masks_3d)) * 100,
            'single_view_clusters': single_view_count,
            'config': config
        },
        'clusters': clusters_data
    }

    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved clusters: {output_json}")
    logger.info("=" * 80)

    return clusters


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging

    parser = argparse.ArgumentParser(description='Cluster 3D masks using geometric overlap')
    parser.add_argument('--masks-3d', required=True,
                       help='Input masks_3d JSON')
    parser.add_argument('--output', required=True,
                       help='Output clusters JSON')
    parser.add_argument('--max-centroid-distance', type=float, default=0.5,
                       help='Maximum centroid distance (meters, default: 0.5)')
    parser.add_argument('--proximity-threshold', type=float, default=0.05,
                       help='Point proximity threshold (meters, default: 0.05)')
    parser.add_argument('--overlap-threshold', type=float, default=0.3,
                       help='Overlap score threshold (0-1, default: 0.3)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    config = {
        'max_centroid_distance': args.max_centroid_distance,
        'min_bbox_iou': 0.05,
        'proximity_threshold': args.proximity_threshold,
        'overlap_threshold': args.overlap_threshold,
        'bbox_weight': 0.3,
        'point_weight': 0.7
    }

    try:
        clusters = run_clustering(
            args.masks_3d,
            args.output,
            config
        )

        print(f"\n✅ Clustering complete!")
        print(f"   Input masks: {len(data['masks'])}")
        print(f"   Output clusters: {len(clusters)}")
        print(f"   Output: {args.output}")

    except Exception as e:
        logger.error(f"Clustering failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
