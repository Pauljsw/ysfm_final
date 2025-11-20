"""
DBSCAN Clustering for Crack Points

Clusters 3D crack points using DBSCAN algorithm.
Uses mapping info to aggregate source masks per cluster.

Usage:
    python -m src.cluster_crack_points_dbscan \
        --input outputs/crack_points.json \
        --output outputs/crack_clusters.json \
        --eps 0.05 \
        --min-samples 10
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


def generate_cluster_colors(n_clusters: int) -> List[Tuple[int, int, int]]:
    """
    Generate distinct colors for clusters using HSV color space.

    Args:
        n_clusters: Number of clusters

    Returns:
        List of RGB tuples
    """
    if n_clusters == 0:
        return []

    colors = []
    for i in range(n_clusters):
        hue = i / n_clusters

        # HSV to RGB conversion
        h = hue * 6
        c = 1.0
        x = 1 - abs(h % 2 - 1)

        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        colors.append((int(r * 255), int(g * 255), int(b * 255)))

    return colors


def save_ply_binary(filename: str, xyz: np.ndarray, rgb: np.ndarray):
    """Save point cloud as binary PLY."""
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(xyz)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode('ascii'))

        for i in range(len(xyz)):
            f.write(xyz[i].astype(np.float32).tobytes())
            f.write(rgb[i].astype(np.uint8).tobytes())


def compute_principal_axis(points: np.ndarray) -> np.ndarray:
    """
    Compute principal axis (main direction) using PCA.

    Args:
        points: (N, 3) 3D points

    Returns:
        axis: (3,) unit vector of principal direction
    """
    if len(points) < 2:
        return np.array([1.0, 0.0, 0.0])

    centroid = np.mean(points, axis=0)
    centered = points - centroid

    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        principal_axis = Vt[0]
        return principal_axis
    except np.linalg.LinAlgError:
        return np.array([1.0, 0.0, 0.0])


def run_dbscan_clustering(
    input_json: str,
    output_json: str,
    eps: float = 0.05,
    min_samples: int = 10,
    output_ply: str = None,
    show_noise: bool = False
):
    """
    Run DBSCAN clustering on crack points.

    Args:
        input_json: Input crack_points.json path
        output_json: Output crack_clusters.json path
        eps: DBSCAN epsilon (max distance between points in cluster)
        min_samples: DBSCAN minimum samples per cluster
        output_ply: Output PLY path for visualization (optional)
        show_noise: Whether to show noise points in PLY (gray)
    """
    logger.info("=" * 80)
    logger.info("DBSCAN Clustering for Crack Points")
    logger.info("=" * 80)

    # Load crack points
    logger.info(f"Loading crack points: {input_json}")
    with open(input_json, 'r') as f:
        data = json.load(f)

    crack_points = data['points']
    metadata = data['metadata']

    logger.info(f"  Loaded {len(crack_points)} crack points")

    if len(crack_points) == 0:
        logger.warning("No crack points to cluster!")
        return

    # Extract coordinates
    xyz = np.array([p['xyz'] for p in crack_points])

    # Run DBSCAN
    logger.info(f"Running DBSCAN (eps={eps}, min_samples={min_samples})...")
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
    labels = clustering.labels_

    # Count clusters
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)

    logger.info(f"  Found {n_clusters} clusters")
    logger.info(f"  Noise points: {n_noise} ({n_noise/len(crack_points)*100:.1f}%)")

    # Aggregate clusters
    logger.info("Aggregating cluster information...")
    clusters = []

    for cluster_id in sorted(unique_labels):
        if cluster_id == -1:
            continue  # Skip noise

        # Get points in this cluster
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_xyz = xyz[cluster_mask]

        # Get point data
        cluster_points = [crack_points[i] for i in cluster_indices]

        # Aggregate source masks
        source_masks_map = defaultdict(lambda: {
            'n_points': 0,
            'confidence_sum': 0.0
        })

        for point in cluster_points:
            for source in point['source_masks']:
                key = (source['image_id'], source['mask_id'])
                source_masks_map[key]['n_points'] += 1
                source_masks_map[key]['confidence_sum'] += source['confidence']

        # Convert to list
        source_masks = []
        for (image_id, mask_id), info in source_masks_map.items():
            source_masks.append({
                'image_id': image_id,
                'mask_id': mask_id,
                'n_points': info['n_points'],
                'avg_confidence': info['confidence_sum'] / info['n_points']
            })

        # Sort by n_points (most contributing masks first)
        source_masks.sort(key=lambda x: x['n_points'], reverse=True)

        # Compute cluster properties
        centroid = np.mean(cluster_xyz, axis=0)
        bbox_min = np.min(cluster_xyz, axis=0)
        bbox_max = np.max(cluster_xyz, axis=0)
        principal_axis = compute_principal_axis(cluster_xyz)

        # Average confidence
        avg_confidence = np.mean([p['avg_confidence'] for p in cluster_points])

        # Create cluster entry
        cluster_entry = {
            'cluster_id': int(cluster_id),
            'n_points': len(cluster_points),
            'point_ids': [p['point_id'] for p in cluster_points],
            'centroid_3d': centroid.tolist(),
            'bbox_3d': {
                'min': bbox_min.tolist(),
                'max': bbox_max.tolist()
            },
            'principal_axis': principal_axis.tolist(),
            'source_masks': source_masks,
            'n_source_masks': len(source_masks),
            'n_views': len(set(s['image_id'] for s in source_masks)),
            'avg_confidence': float(avg_confidence)
        }

        clusters.append(cluster_entry)

        logger.debug(f"  Cluster {cluster_id}: {len(cluster_points)} points, "
                     f"{len(source_masks)} source masks, {cluster_entry['n_views']} views")

    # Sort clusters by n_points
    clusters.sort(key=lambda x: x['n_points'], reverse=True)

    # Statistics
    logger.info("=" * 80)
    logger.info("Clustering Statistics")
    logger.info("=" * 80)
    logger.info(f"  Total clusters: {n_clusters}")
    logger.info(f"  Total noise points: {n_noise}")

    if clusters:
        points_per_cluster = [c['n_points'] for c in clusters]
        logger.info(f"  Points per cluster: min={min(points_per_cluster)}, "
                    f"max={max(points_per_cluster)}, mean={np.mean(points_per_cluster):.1f}")

        views_per_cluster = [c['n_views'] for c in clusters]
        logger.info(f"  Views per cluster: min={min(views_per_cluster)}, "
                    f"max={max(views_per_cluster)}, mean={np.mean(views_per_cluster):.1f}")

    # Save output
    output_data = {
        'metadata': {
            'total_crack_points': len(crack_points),
            'n_clusters': n_clusters,
            'noise_points': n_noise,
            'dbscan_eps': eps,
            'dbscan_min_samples': min_samples,
            'input_metadata': metadata
        },
        'clusters': clusters
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nSaved clusters: {output_json}")

    # Generate PLY visualization if requested
    if output_ply:
        logger.info(f"\nGenerating PLY visualization...")

        # Build point_id to xyz lookup
        point_lookup = {p['point_id']: np.array(p['xyz']) for p in crack_points}

        # Generate colors
        colors = generate_cluster_colors(len(clusters))

        # Collect all points with colors
        all_xyz = []
        all_rgb = []
        clustered_point_ids = set()

        for cluster_idx, cluster in enumerate(clusters):
            point_ids = cluster['point_ids']
            color = colors[cluster_idx % len(colors)] if colors else (255, 0, 0)

            for point_id in point_ids:
                if point_id in point_lookup:
                    all_xyz.append(point_lookup[point_id])
                    all_rgb.append(color)
                    clustered_point_ids.add(point_id)

        # Add noise points if requested
        if show_noise:
            noise_color = (128, 128, 128)
            noise_count = 0
            for point in crack_points:
                if point['point_id'] not in clustered_point_ids:
                    all_xyz.append(np.array(point['xyz']))
                    all_rgb.append(noise_color)
                    noise_count += 1
            logger.info(f"  Added {noise_count} noise points (gray)")

        # Convert and save
        all_xyz = np.array(all_xyz)
        all_rgb = np.array(all_rgb, dtype=np.uint8)

        save_ply_binary(output_ply, all_xyz, all_rgb)
        logger.info(f"  Saved PLY: {output_ply} ({len(all_xyz)} points)")

    logger.info("=" * 80)

    return clusters


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging

    parser = argparse.ArgumentParser(
        description='DBSCAN clustering for crack points with mapping info'
    )
    parser.add_argument('--input', required=True,
                       help='Input crack_points.json')
    parser.add_argument('--output', required=True,
                       help='Output crack_clusters.json')
    parser.add_argument('--eps', type=float, default=0.05,
                       help='DBSCAN epsilon - max distance between points (meters, default: 0.05)')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='DBSCAN min samples per cluster (default: 10)')
    parser.add_argument('--output-ply', default=None,
                       help='Output PLY path for cluster visualization (optional)')
    parser.add_argument('--show-noise', action='store_true',
                       help='Show noise points in gray in PLY output')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        clusters = run_dbscan_clustering(
            args.input,
            args.output,
            args.eps,
            args.min_samples,
            args.output_ply,
            args.show_noise
        )

        if clusters:
            print(f"\n✅ Clustering complete!")
            print(f"   Total clusters: {len(clusters)}")
            print(f"   Largest cluster: {clusters[0]['n_points']} points")
            print(f"   Output JSON: {args.output}")
            if args.output_ply:
                print(f"   Output PLY: {args.output_ply}")
        else:
            print(f"\n⚠️ No clusters found")

    except Exception as e:
        logger.error(f"Clustering failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
