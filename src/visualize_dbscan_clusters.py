"""
Visualize DBSCAN Clustering Results

Creates PLY file with each cluster in different color.

Usage:
    python -m src.visualize_dbscan_clusters \
        --clusters outputs/crack_clusters.json \
        --crack-points outputs/crack_points.json \
        --output outputs/clustered_cracks.ply
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def generate_cluster_colors(n_clusters: int) -> List[Tuple[int, int, int]]:
    """
    Generate distinct colors for clusters.

    Args:
        n_clusters: Number of clusters

    Returns:
        List of RGB tuples
    """
    if n_clusters == 0:
        return []

    # Use HSV color space for distinct colors
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


def visualize_clusters(
    clusters_json: str,
    crack_points_json: str,
    output_ply: str,
    show_noise: bool = False,
    noise_color: Tuple[int, int, int] = (128, 128, 128)
):
    """
    Visualize DBSCAN clusters as colored PLY.

    Args:
        clusters_json: Input crack_clusters.json
        crack_points_json: Input crack_points.json
        output_ply: Output PLY path
        show_noise: Whether to show noise points (gray)
        noise_color: RGB color for noise points
    """
    logger.info("=" * 80)
    logger.info("Visualize DBSCAN Clusters")
    logger.info("=" * 80)

    # Load clusters
    logger.info(f"Loading clusters: {clusters_json}")
    with open(clusters_json, 'r') as f:
        clusters_data = json.load(f)

    clusters = clusters_data['clusters']
    metadata = clusters_data['metadata']

    logger.info(f"  Loaded {len(clusters)} clusters")
    logger.info(f"  Noise points: {metadata['noise_points']}")

    # Load crack points
    logger.info(f"Loading crack points: {crack_points_json}")
    with open(crack_points_json, 'r') as f:
        points_data = json.load(f)

    crack_points = points_data['points']
    logger.info(f"  Loaded {len(crack_points)} crack points")

    # Build point_id to xyz lookup
    point_lookup = {p['point_id']: np.array(p['xyz']) for p in crack_points}

    # Generate colors
    colors = generate_cluster_colors(len(clusters))

    # Collect all points with colors
    all_xyz = []
    all_rgb = []

    clustered_point_ids = set()

    for cluster_idx, cluster in enumerate(clusters):
        cluster_id = cluster['cluster_id']
        point_ids = cluster['point_ids']
        color = colors[cluster_idx % len(colors)]

        for point_id in point_ids:
            if point_id in point_lookup:
                all_xyz.append(point_lookup[point_id])
                all_rgb.append(color)
                clustered_point_ids.add(point_id)

        logger.debug(f"  Cluster {cluster_id}: {len(point_ids)} points, color={color}")

    # Add noise points if requested
    if show_noise:
        noise_count = 0
        for point in crack_points:
            if point['point_id'] not in clustered_point_ids:
                all_xyz.append(np.array(point['xyz']))
                all_rgb.append(noise_color)
                noise_count += 1

        logger.info(f"  Added {noise_count} noise points (gray)")

    # Convert to arrays
    all_xyz = np.array(all_xyz)
    all_rgb = np.array(all_rgb, dtype=np.uint8)

    logger.info(f"  Total points: {len(all_xyz)}")

    # Save PLY
    save_ply_binary(output_ply, all_xyz, all_rgb)

    logger.info(f"\nSaved: {output_ply}")
    logger.info("=" * 80)

    return len(clusters), len(all_xyz)


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging

    parser = argparse.ArgumentParser(
        description='Visualize DBSCAN clustering results as colored PLY'
    )
    parser.add_argument('--clusters', required=True,
                       help='Input crack_clusters.json')
    parser.add_argument('--crack-points', required=True,
                       help='Input crack_points.json')
    parser.add_argument('--output', required=True,
                       help='Output PLY path')
    parser.add_argument('--show-noise', action='store_true',
                       help='Show noise points in gray')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        n_clusters, n_points = visualize_clusters(
            args.clusters,
            args.crack_points,
            args.output,
            args.show_noise
        )

        print(f"\n✅ Visualization complete!")
        print(f"   Clusters: {n_clusters}")
        print(f"   Points: {n_points}")
        print(f"   Output: {args.output}")

    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
