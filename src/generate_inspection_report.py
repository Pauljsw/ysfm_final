#!/usr/bin/env python3
"""
Generate 2D inspection diagram and measurement table from SfM point cloud and crack clusters.

This module creates:
1. 2D inspection diagram (PNG) showing wall outline with cracks
2. Measurement table (CSV) with crack dimensions
"""

import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ply(ply_path: Path) -> np.ndarray:
    """
    Load point cloud from PLY file (supports both ASCII and binary formats).

    Returns:
        Nx3 numpy array of XYZ coordinates
    """
    import struct

    # Property type sizes in bytes
    type_sizes = {
        'char': 1, 'uchar': 1, 'int8': 1, 'uint8': 1,
        'short': 2, 'ushort': 2, 'int16': 2, 'uint16': 2,
        'int': 4, 'uint': 4, 'int32': 4, 'uint32': 4,
        'float': 4, 'float32': 4,
        'double': 8, 'float64': 8
    }

    with open(ply_path, 'rb') as f:
        # Read header
        vertex_count = 0
        is_binary = False
        is_little_endian = True
        vertex_properties = []
        in_vertex_element = False

        while True:
            line = f.readline().decode('ascii').strip()

            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
                in_vertex_element = True
            elif line.startswith('element') and in_vertex_element:
                in_vertex_element = False
            elif line.startswith('property') and in_vertex_element:
                parts = line.split()
                if len(parts) >= 3:
                    prop_type = parts[1]
                    prop_name = parts[2]
                    vertex_properties.append((prop_name, prop_type))
            elif line.startswith('format'):
                if 'binary_little_endian' in line:
                    is_binary = True
                    is_little_endian = True
                elif 'binary_big_endian' in line:
                    is_binary = True
                    is_little_endian = False
            elif line == 'end_header':
                break

        # Calculate vertex size
        vertex_size = sum(type_sizes.get(prop[1], 4) for prop in vertex_properties)
        if vertex_size == 0:
            vertex_size = 12  # Default: just xyz floats

        if is_binary:
            # Binary format
            endian = '<' if is_little_endian else '>'
            points = []

            for _ in range(vertex_count):
                data = f.read(vertex_size)
                if len(data) < vertex_size:
                    break
                # First 3 floats are x, y, z
                x, y, z = struct.unpack(f'{endian}fff', data[:12])
                points.append([x, y, z])

            points = np.array(points)
        else:
            # ASCII format
            points = []
            for _ in range(vertex_count):
                line = f.readline().decode('ascii').strip()
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        points.append([x, y, z])
                    except ValueError:
                        continue
            points = np.array(points)

        # Filter NaN and Inf values
        if len(points) > 0:
            valid_mask = np.all(np.isfinite(points), axis=1)
            n_invalid = np.sum(~valid_mask)
            if n_invalid > 0:
                logger.warning(f"Removed {n_invalid} points with NaN/Inf values")
            points = points[valid_mask]

        logger.info(f"Loaded {len(points)} points from {ply_path}")
        return points


def filter_noise_statistical(points: np.ndarray, k: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """
    Statistical outlier removal based on mean distance to k nearest neighbors.

    Points with mean distance > mean + std_ratio * std are removed.
    """
    if len(points) < k + 1:
        return points

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points)
    distances, _ = nbrs.kneighbors(points)

    # Mean distance to k neighbors (exclude self)
    mean_distances = distances[:, 1:].mean(axis=1)

    # Filter outliers
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    threshold = global_mean + std_ratio * global_std

    mask = mean_distances <= threshold
    filtered = points[mask]

    logger.info(f"Statistical filtering: {len(points)} -> {len(filtered)} points "
                f"(removed {len(points) - len(filtered)})")
    return filtered


def filter_noise_density(points: np.ndarray, axis: int = 2,
                         grid_size: float = 0.1, min_count: int = 3) -> np.ndarray:
    """
    Density-based filtering along specified axis.

    Projects points to 2D (excluding specified axis) and filters by cell density.
    """
    # Project to 2D
    axes = [i for i in range(3) if i != axis]
    projected = points[:, axes]

    # Create grid
    min_vals = projected.min(axis=0)
    max_vals = projected.max(axis=0)

    # Assign points to grid cells
    cell_indices = ((projected - min_vals) / grid_size).astype(int)

    # Count points per cell
    cell_counts = {}
    for i, cell in enumerate(cell_indices):
        key = tuple(cell)
        if key not in cell_counts:
            cell_counts[key] = []
        cell_counts[key].append(i)

    # Keep points in cells with enough neighbors
    keep_indices = []
    for cell, indices in cell_counts.items():
        if len(indices) >= min_count:
            keep_indices.extend(indices)

    filtered = points[keep_indices]
    logger.info(f"Density filtering: {len(points)} -> {len(filtered)} points")
    return filtered


def fit_plane_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit plane to points using PCA.

    Returns:
        (centroid, normal, principal_axes)
        - centroid: center of the plane
        - normal: plane normal vector
        - principal_axes: 2x3 array of the two principal axes in the plane
    """
    centroid = points.mean(axis=0)
    centered = points - centroid

    pca = PCA(n_components=3)
    pca.fit(centered)

    # The smallest eigenvalue component is the normal
    normal = pca.components_[2]  # Smallest variance direction
    principal_axes = pca.components_[:2]  # Two largest variance directions

    logger.info(f"Plane fitted - Normal: {normal}")
    return centroid, normal, principal_axes


def project_to_plane(points: np.ndarray, centroid: np.ndarray,
                     principal_axes: np.ndarray) -> np.ndarray:
    """
    Project 3D points onto 2D plane defined by principal axes.

    Returns:
        Nx2 array of 2D coordinates
    """
    centered = points - centroid
    projected_2d = np.dot(centered, principal_axes.T)
    return projected_2d


def extract_boundary_alpha_shape(points_2d: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Extract boundary using alpha shape (concave hull).

    Simplified implementation using grid-based boundary detection.
    """
    if len(points_2d) < 3:
        return points_2d

    # Use convex hull as fallback, with shrinking based on density
    try:
        hull = ConvexHull(points_2d)
        boundary = points_2d[hull.vertices]
        return boundary
    except Exception as e:
        logger.warning(f"Convex hull failed: {e}")
        return points_2d


def extract_boundary_grid(points_2d: np.ndarray, grid_size: float = 0.05) -> np.ndarray:
    """
    Extract boundary using grid-based edge detection.

    Returns boundary points where grid cells touch empty space.
    """
    if len(points_2d) < 3:
        return points_2d

    min_vals = points_2d.min(axis=0)
    max_vals = points_2d.max(axis=0)

    # Create occupancy grid
    grid_shape = ((max_vals - min_vals) / grid_size + 1).astype(int)
    grid = np.zeros(grid_shape, dtype=bool)

    # Mark occupied cells
    cell_indices = ((points_2d - min_vals) / grid_size).astype(int)
    cell_indices = np.clip(cell_indices, 0, grid_shape - 1)

    for ci in cell_indices:
        grid[ci[0], ci[1]] = True

    # Find boundary cells (occupied cells with at least one empty neighbor)
    boundary_cells = []
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if not grid[i, j]:
                continue

            # Check 8-neighbors
            is_boundary = False
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= grid_shape[0] or nj < 0 or nj >= grid_shape[1]:
                        is_boundary = True
                        break
                    if not grid[ni, nj]:
                        is_boundary = True
                        break
                if is_boundary:
                    break

            if is_boundary:
                boundary_cells.append([i, j])

    if not boundary_cells:
        return points_2d

    # Convert back to coordinates
    boundary_coords = np.array(boundary_cells) * grid_size + min_vals + grid_size / 2

    # Order boundary points (approximate)
    if len(boundary_coords) > 3:
        try:
            hull = ConvexHull(boundary_coords)
            boundary_coords = boundary_coords[hull.vertices]
        except:
            pass

    return boundary_coords


def load_clusters(clusters_path: Path) -> List[Dict]:
    """Load crack clusters from JSON."""
    with open(clusters_path, 'r') as f:
        data = json.load(f)
    return data.get('clusters', data) if isinstance(data, dict) else data


def load_crack_points(crack_points_path: Path) -> Dict[int, Dict]:
    """Load crack points and create lookup by point_id."""
    with open(crack_points_path, 'r') as f:
        data = json.load(f)

    # Handle different JSON structures
    if isinstance(data, dict):
        if 'crack_points' in data:
            points = data['crack_points']
        elif 'points' in data:
            points = data['points']
        else:
            # Assume the dict values are the points
            points = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else []
    elif isinstance(data, list):
        points = data
    else:
        logger.error(f"Unexpected crack_points format: {type(data)}")
        return {}

    # Build lookup
    lookup = {}
    for p in points:
        if isinstance(p, dict) and 'point_id' in p:
            lookup[p['point_id']] = p

    logger.info(f"Loaded {len(lookup)} crack points")
    return lookup


def load_measurements(measurements_path: Path) -> Dict[int, Dict]:
    """Load measurements and create lookup by cluster_id."""
    with open(measurements_path, 'r') as f:
        data = json.load(f)

    measurements = data.get('measurements', data) if isinstance(data, dict) else data
    return {m['cluster_id']: m for m in measurements}


def get_cluster_line(cluster: Dict, crack_points_lookup: Dict,
                     centroid: np.ndarray, principal_axes: np.ndarray) -> Optional[np.ndarray]:
    """
    Get 2D line representation of a cluster.

    Projects cluster points to 2D and fits a line through them.
    """
    point_ids = cluster.get('point_ids', [])

    # Get 3D coordinates
    xyz_list = []
    for pid in point_ids:
        if pid in crack_points_lookup:
            point = crack_points_lookup[pid]
            if 'xyz' in point:
                xyz_list.append(point['xyz'])

    if len(xyz_list) < 2:
        return None

    xyz = np.array(xyz_list)

    # Project to 2D
    points_2d = project_to_plane(xyz, centroid, principal_axes)

    # Fit line through points using PCA
    if len(points_2d) >= 2:
        pca = PCA(n_components=1)
        pca.fit(points_2d)

        center = points_2d.mean(axis=0)
        direction = pca.components_[0]

        # Project points onto line to find endpoints
        projections = np.dot(points_2d - center, direction)
        min_proj, max_proj = projections.min(), projections.max()

        start = center + min_proj * direction
        end = center + max_proj * direction

        return np.array([start, end])

    return None


def generate_inspection_diagram(
    wall_boundary: np.ndarray,
    crack_lines: List[Tuple[int, np.ndarray]],
    output_path: Path,
    title: str = "Inspection Diagram",
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Generate inspection diagram image.

    Args:
        wall_boundary: Nx2 array of boundary points
        crack_lines: List of (cluster_id, line_points) tuples
        output_path: Output image path
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Draw wall boundary
    if len(wall_boundary) >= 3:
        # Close the polygon
        boundary_closed = np.vstack([wall_boundary, wall_boundary[0]])
        ax.plot(boundary_closed[:, 0], boundary_closed[:, 1],
                'k-', linewidth=2, label='Wall boundary')
        ax.fill(wall_boundary[:, 0], wall_boundary[:, 1],
                alpha=0.1, color='gray')

    # Draw cracks
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(crack_lines))))

    for i, (cluster_id, line) in enumerate(crack_lines):
        if line is not None and len(line) >= 2:
            color = colors[i % len(colors)]
            ax.plot(line[:, 0], line[:, 1],
                    linewidth=3, color=color,
                    label=f'Crack {cluster_id + 1}')

            # Add label at midpoint
            mid = line.mean(axis=0)
            ax.annotate(f'{cluster_id + 1}', mid,
                       fontsize=10, fontweight='bold',
                       ha='center', va='bottom',
                       color=color)

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved inspection diagram: {output_path}")


def generate_measurement_table(
    measurements: Dict[int, Dict],
    output_path: Path
):
    """
    Generate measurement table CSV.

    Columns: No., Damage Type, Width (mm), Length (mm)
    """
    rows = []

    for cluster_id in sorted(measurements.keys()):
        m = measurements[cluster_id]
        rows.append({
            'No.': f'Crack {cluster_id + 1}',
            'Damage Type': 'Crack',
            'Width (mm)': round(m.get('avg_width_mm', 0), 2),
            'Length (mm)': round(m.get('total_length_mm', 0), 1),
        })

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['No.', 'Damage Type', 'Width (mm)', 'Length (mm)'])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved measurement table: {output_path}")

    # Also print to console
    print("\n" + "="*60)
    print("Measurement Table")
    print("="*60)
    print(f"{'No.':<12} {'Type':<10} {'Width(mm)':<12} {'Length(mm)':<12}")
    print("-"*60)
    for row in rows:
        print(f"{row['No.']:<12} {row['Damage Type']:<10} {row['Width (mm)']:<12} {row['Length (mm)']:<12}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate 2D inspection diagram and measurement table'
    )
    parser.add_argument('--point-cloud', type=str, required=True,
                        help='Path to SfM point cloud PLY file')
    parser.add_argument('--clusters', type=str, required=True,
                        help='Path to crack clusters JSON')
    parser.add_argument('--crack-points', type=str, required=True,
                        help='Path to crack points JSON')
    parser.add_argument('--measurements', type=str, required=True,
                        help='Path to cluster measurements JSON')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--diagram-name', type=str, default='inspection_diagram.png',
                        help='Output diagram filename')
    parser.add_argument('--table-name', type=str, default='measurement_table.csv',
                        help='Output table filename')

    # Filtering parameters
    parser.add_argument('--noise-k', type=int, default=20,
                        help='K neighbors for statistical filtering')
    parser.add_argument('--noise-std', type=float, default=2.0,
                        help='Std ratio for statistical filtering')
    parser.add_argument('--density-grid', type=float, default=0.1,
                        help='Grid size for density filtering')
    parser.add_argument('--density-min', type=int, default=3,
                        help='Min points per cell for density filtering')

    # Boundary parameters
    parser.add_argument('--boundary-grid', type=float, default=0.05,
                        help='Grid size for boundary extraction')

    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("="*60)
    logger.info("Inspection Report Generation")
    logger.info("="*60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load point cloud
    logger.info(f"Loading point cloud: {args.point_cloud}")
    points = load_ply(Path(args.point_cloud))

    # Filter noise
    logger.info("Filtering noise...")
    points = filter_noise_statistical(points, k=args.noise_k, std_ratio=args.noise_std)
    points = filter_noise_density(points, grid_size=args.density_grid, min_count=args.density_min)

    # Fit plane
    logger.info("Fitting plane...")
    centroid, normal, principal_axes = fit_plane_pca(points)

    # Project to 2D
    logger.info("Projecting to 2D...")
    points_2d = project_to_plane(points, centroid, principal_axes)

    # Extract boundary
    logger.info("Extracting boundary...")
    boundary = extract_boundary_grid(points_2d, grid_size=args.boundary_grid)

    # Load cluster data
    logger.info(f"Loading clusters: {args.clusters}")
    clusters = load_clusters(Path(args.clusters))

    logger.info(f"Loading crack points: {args.crack_points}")
    crack_points_lookup = load_crack_points(Path(args.crack_points))

    logger.info(f"Loading measurements: {args.measurements}")
    measurements = load_measurements(Path(args.measurements))

    # Get crack lines
    logger.info("Generating crack lines...")
    crack_lines = []
    for cluster in clusters:
        cluster_id = cluster.get('cluster_id', 0)
        line = get_cluster_line(cluster, crack_points_lookup, centroid, principal_axes)
        if line is not None:
            crack_lines.append((cluster_id, line))

    logger.info(f"Generated {len(crack_lines)} crack lines")

    # Generate outputs
    diagram_path = output_dir / args.diagram_name
    table_path = output_dir / args.table_name

    generate_inspection_diagram(boundary, crack_lines, diagram_path)
    generate_measurement_table(measurements, table_path)

    logger.info("="*60)
    logger.info("Done!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
