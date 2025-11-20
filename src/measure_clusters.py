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
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from skimage.morphology import skeletonize
from skimage.draw import polygon as draw_polygon

logger = logging.getLogger(__name__)


def generate_cluster_colors(n_clusters: int) -> List[Tuple[int, int, int]]:
    """
    Generate distinct colors for clusters using HSV color space.
    Must match cluster_crack_points_dbscan.py for consistency.
    Uses golden ratio and varying saturation/value to avoid duplicates.
    """
    if n_clusters == 0:
        return []

    colors = []
    for i in range(n_clusters):
        # Vary hue across the spectrum using golden ratio
        hue = (i * 0.618033988749895) % 1.0

        # Vary saturation and value for more distinction
        sat_idx = (i // 12) % 3
        saturation = [1.0, 0.7, 0.85][sat_idx]
        value = [1.0, 0.9, 0.95][sat_idx]

        # HSV to RGB conversion
        h = hue * 6
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c

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

        colors.append((int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)))

    return colors


def rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
    """
    Convert RGB to approximate color name for user convenience.
    """
    r, g, b = rgb

    # Simple hue-based naming
    if r >= 200 and g < 100 and b < 100:
        return "Red"
    elif r >= 200 and g >= 200 and b < 100:
        return "Yellow"
    elif r < 100 and g >= 200 and b < 100:
        return "Green"
    elif r < 100 and g >= 200 and b >= 200:
        return "Cyan"
    elif r < 100 and g < 100 and b >= 200:
        return "Blue"
    elif r >= 200 and g < 100 and b >= 200:
        return "Magenta"
    elif r >= 200 and g >= 100 and b < 100:
        return "Orange"
    elif r >= 100 and g < 100 and b >= 100:
        return "Purple"
    elif r < 150 and g >= 150 and b < 150:
        return "Lime"
    elif r < 100 and g >= 100 and b >= 150:
        return "Teal"
    else:
        # Return hex for ambiguous colors
        return f"#{r:02X}{g:02X}{b:02X}"


def setup_logging(level: str = 'INFO'):
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_scale_with_fallback(
    scale_map: np.ndarray,
    r: int,
    c: int,
    search_radius: int = 20
) -> float:
    """
    Get scale value at (r, c) with fallback to nearby valid pixels.

    If scale_map[r, c] is 0 or invalid, search in expanding squares
    for nearby valid scale values and return their mean.
    """
    H, W = scale_map.shape

    # First try direct value
    if 0 <= r < H and 0 <= c < W:
        val = scale_map[r, c]
        if val > 0:
            return float(val)

    # Search in expanding squares for valid neighbors
    for radius in range(1, search_radius + 1):
        valid_values = []

        # Check pixels at this radius (square boundary)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                # Only check boundary pixels of the square
                if abs(dr) != radius and abs(dc) != radius:
                    continue

                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    val = scale_map[nr, nc]
                    if val > 0:
                        valid_values.append(val)

        if valid_values:
            return float(np.mean(valid_values))

    return 0.0


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
    scale_map: np.ndarray
) -> float:
    """
    Calculate crack length from skeleton using direction-based method with per-pixel scale.

    Args:
        skeleton: Binary skeleton image (1px thick)
        scale_map: Per-pixel mm/px scale map (same size as skeleton)

    Returns:
        Length in mm
    """
    # Find skeleton pixels
    rows, cols = np.where(skeleton > 0)

    if len(rows) < 2:
        return 0.0

    # Create set for fast lookup
    skeleton_pixels = set(zip(rows, cols))

    total_length = 0.0
    visited_edges = set()

    for r, c in skeleton_pixels:
        # Get scale at this pixel with fallback to nearby valid pixels
        D = get_scale_with_fallback(scale_map, r, c)

        if D == 0:
            continue

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

                    # Use average scale of the two pixels (with fallback)
                    D_neighbor = get_scale_with_fallback(scale_map, nr, nc)
                    if D_neighbor == 0:
                        D_neighbor = D
                    D_avg = (D + D_neighbor) / 2.0

                    # Determine distance based on direction
                    if i < 4:  # H or V
                        total_length += D_avg
                    else:  # Diagonal
                        total_length += D_avg * np.sqrt(2)

    return total_length


def calculate_skeleton_width(
    skeleton: np.ndarray,
    binary_mask: np.ndarray,
    scale_map: np.ndarray,
    sample_interval: int = 5
) -> Tuple[float, float]:
    """
    Calculate crack width by measuring perpendicular to skeleton with per-pixel scale.

    Args:
        skeleton: Binary skeleton image
        binary_mask: Original binary mask
        scale_map: Per-pixel mm/px scale map
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

        # Get scale at this pixel with fallback
        D = get_scale_with_fallback(scale_map, r, c)
        if D == 0:
            continue

        # Estimate local direction from nearby skeleton pixels
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


def detect_crack_pixels_in_mask(
    grayscale: np.ndarray,
    mask: np.ndarray,
    method: str = 'adaptive'
) -> np.ndarray:
    """
    Detect actual crack pixels within YOLO mask region using intensity.

    Cracks are dark pixels - this finds them within the mask candidate region.

    Args:
        grayscale: Grayscale image
        mask: YOLO mask (candidate region)
        method: 'adaptive', 'otsu', or 'percentile'

    Returns:
        Binary image where 1 = actual crack pixel, 0 = background
    """
    # Apply mask to get ROI
    masked_gray = grayscale.copy()
    masked_gray[mask == 0] = 255  # Set non-mask areas to white

    if method == 'otsu':
        # Otsu's method - good for bimodal distribution
        mask_pixels = grayscale[mask > 0]
        if len(mask_pixels) == 0:
            return np.zeros_like(mask)

        threshold, _ = cv2.threshold(
            mask_pixels, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        crack_binary = (grayscale < threshold).astype(np.uint8) * 255

    elif method == 'adaptive':
        # Adaptive thresholding - handles varying illumination
        crack_binary = cv2.adaptiveThreshold(
            masked_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )

    elif method.startswith('percentile'):
        # Use darkest percentile within mask as crack
        mask_pixels = grayscale[mask > 0]
        if len(mask_pixels) == 0:
            return np.zeros_like(mask)

        # Extract percentile value from method string (e.g., 'percentile_25')
        percentile_val = 30  # default
        if '_' in method:
            try:
                percentile_val = float(method.split('_')[1])
            except:
                pass
        threshold = np.percentile(mask_pixels, percentile_val)
        crack_binary = (grayscale < threshold).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply original mask to limit to ROI
    crack_binary = cv2.bitwise_and(crack_binary, crack_binary, mask=mask.astype(np.uint8))

    # Morphological cleanup - remove noise
    kernel = np.ones((2, 2), np.uint8)
    crack_binary = cv2.morphologyEx(crack_binary, cv2.MORPH_OPEN, kernel)

    return crack_binary


def calculate_intensity_based_width(
    skeleton: np.ndarray,
    grayscale: np.ndarray,
    mask: np.ndarray,
    scale_map: np.ndarray,
    sample_interval: int = 5,
    detection_method: str = 'percentile'
) -> Tuple[float, float]:
    """
    Calculate crack width by detecting actual dark crack pixels within mask.

    This gives true crack width in pixels, not mask width.

    Args:
        skeleton: Binary skeleton image
        grayscale: Grayscale image
        mask: YOLO mask (candidate region)
        scale_map: Per-pixel mm/px scale map
        sample_interval: Sample every N skeleton pixels
        detection_method: Method for crack pixel detection

    Returns:
        (average_width_mm, max_width_mm)
    """
    # Detect actual crack pixels within mask
    crack_binary = detect_crack_pixels_in_mask(grayscale, mask, detection_method)

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

        # Get scale at this pixel with fallback
        D = get_scale_with_fallback(scale_map, r, c)
        if D == 0:
            continue

        # Estimate local direction from nearby skeleton pixels
        window = 3
        nearby_rows = rows[max(0, idx-window):min(n_pixels, idx+window+1)]
        nearby_cols = cols[max(0, idx-window):min(n_pixels, idx+window+1)]

        if len(nearby_rows) < 2:
            continue

        # Fit line to get direction
        dr = nearby_rows[-1] - nearby_rows[0]
        dc = nearby_cols[-1] - nearby_cols[0]

        # Perpendicular direction (normal)
        length = np.sqrt(dr**2 + dc**2)
        if length < 1e-6:
            continue

        nr, nc = -dc / length, dr / length

        # Measure width along normal using detected crack pixels
        width_pixels = measure_width_along_normal(
            crack_binary, r, c, nr, nc, max_distance=50
        )

        if width_pixels > 0:
            widths.append(width_pixels * D)

    if not widths:
        return 0.0, 0.0

    return np.mean(widths), np.max(widths)


def calculate_edge_based_width(
    skeleton: np.ndarray,
    grayscale: np.ndarray,
    roi_mask: np.ndarray,
    scale_map: np.ndarray,
    sample_interval: int = 5,
    canny_low: int = 50,
    canny_high: int = 150
) -> Tuple[float, float]:
    """
    Calculate crack width using edge detection on grayscale image with per-pixel scale.

    This measures the actual crack boundaries based on intensity changes,
    not the YOLO mask boundaries.

    Args:
        skeleton: Binary skeleton image
        grayscale: Grayscale image (ROI region)
        roi_mask: ROI mask to limit edge detection area
        scale_map: Per-pixel mm/px scale map
        sample_interval: Sample every N skeleton pixels
        canny_low: Canny edge detection low threshold
        canny_high: Canny edge detection high threshold

    Returns:
        (average_width_mm, max_width_mm)
    """
    # Find skeleton pixels
    rows, cols = np.where(skeleton > 0)

    if len(rows) < 3:
        return 0.0, 0.0

    # Apply edge detection within ROI
    # Preprocess: enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(grayscale)

    # Apply ROI mask
    enhanced_roi = cv2.bitwise_and(enhanced, enhanced, mask=roi_mask.astype(np.uint8))

    # Edge detection
    edges = cv2.Canny(enhanced_roi, canny_low, canny_high)

    # Sample skeleton pixels
    n_pixels = len(rows)
    sample_indices = range(0, n_pixels, sample_interval)

    widths = []

    for idx in sample_indices:
        r, c = rows[idx], cols[idx]

        # Get scale at this pixel with fallback
        D = get_scale_with_fallback(scale_map, r, c)
        if D == 0:
            continue

        # Estimate local direction from nearby skeleton pixels
        window = 3
        nearby_rows = rows[max(0, idx-window):min(n_pixels, idx+window+1)]
        nearby_cols = cols[max(0, idx-window):min(n_pixels, idx+window+1)]

        if len(nearby_rows) < 2:
            continue

        # Fit line to get direction
        dr = nearby_rows[-1] - nearby_rows[0]
        dc = nearby_cols[-1] - nearby_cols[0]

        # Perpendicular direction (normal)
        length = np.sqrt(dr**2 + dc**2)
        if length < 1e-6:
            continue

        nr, nc = -dc / length, dr / length

        # Measure width along normal direction using edges
        width_pixels = measure_edge_width_along_normal(
            edges, r, c, nr, nc
        )

        if width_pixels > 0:
            widths.append(width_pixels * D)

    if not widths:
        return 0.0, 0.0

    return np.mean(widths), np.max(widths)


def measure_edge_width_along_normal(
    edges: np.ndarray,
    r: int, c: int,
    nr: float, nc: float,
    max_distance: int = 100
) -> int:
    """
    Measure crack width by finding edge pixels along normal direction.

    Returns:
        Width in pixels (distance between two edges)
    """
    height, width = edges.shape

    # Search in both directions along normal for edge pixels
    positive_edge = -1
    negative_edge = -1

    # Positive direction - find first edge
    for d in range(1, max_distance):
        new_r = int(round(r + d * nr))
        new_c = int(round(c + d * nc))

        if not (0 <= new_r < height and 0 <= new_c < width):
            break

        if edges[new_r, new_c] > 0:
            positive_edge = d
            break

    # Negative direction - find first edge
    for d in range(1, max_distance):
        new_r = int(round(r - d * nr))
        new_c = int(round(c - d * nc))

        if not (0 <= new_r < height and 0 <= new_c < width):
            break

        if edges[new_r, new_c] > 0:
            negative_edge = d
            break

    # Width = distance between two edges
    if positive_edge > 0 and negative_edge > 0:
        return positive_edge + negative_edge
    elif positive_edge > 0:
        return positive_edge * 2  # Estimate: double one side
    elif negative_edge > 0:
        return negative_edge * 2
    else:
        return 0


def measure_segment_2d(
    masks_dir: Path,
    image_id: str,
    mask_id: int,
    scale_map: np.ndarray,
    segment_uvs: List[List[float]] = None,
    image_shape: Tuple[int, int] = (2160, 3840),
    margin: int = 50,
    rgb_dir: Path = None,
    use_edge_width: bool = True,
    detection_method: str = 'percentile_30',
    sample_interval: int = 5
) -> Dict:
    """
    Measure a segment using 2D skeleton method with per-pixel scale map.

    Args:
        masks_dir: Path to YOLO masks
        image_id: Image identifier
        mask_id: Mask index
        scale_map: Per-pixel mm/px scale map (H, W)
        segment_uvs: List of [u, v] coordinates for this segment (for cropping)
        image_shape: (height, width)
        margin: Margin around bounding box for cropping
        rgb_dir: Path to RGB images (for edge-based width measurement)
        use_edge_width: Use edge detection for width measurement

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

    # Crop mask to segment region if UV coordinates provided
    if segment_uvs and len(segment_uvs) > 0:
        uvs = np.array(segment_uvs)
        u_coords = uvs[:, 0]
        v_coords = uvs[:, 1]

        # Calculate bounding box with margin
        u_min = max(0, int(np.min(u_coords) - margin))
        u_max = min(image_shape[1], int(np.max(u_coords) + margin))
        v_min = max(0, int(np.min(v_coords) - margin))
        v_max = min(image_shape[0], int(np.max(v_coords) + margin))

        # Crop the mask to segment region
        # Only keep mask pixels within the bounding box
        cropped_mask = np.zeros_like(binary_mask)
        cropped_mask[v_min:v_max, u_min:u_max] = binary_mask[v_min:v_max, u_min:u_max]
        binary_mask = cropped_mask

        if binary_mask.sum() == 0:
            return {'length_mm': 0, 'avg_width_mm': 0, 'max_width_mm': 0}

    # Skeletonize
    skeleton = skeletonize(binary_mask > 0)

    # Calculate length using per-pixel scale
    length_mm = calculate_skeleton_length(skeleton, scale_map)

    # Calculate width
    avg_width_mm, max_width_mm = 0.0, 0.0
    width_method = 'mask'

    # Try intensity-based width measurement if RGB image available
    if use_edge_width and rgb_dir:
        # Load RGB image
        rgb_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = rgb_dir / f"{image_id}{ext}"
            if candidate.exists():
                rgb_path = candidate
                break

        if rgb_path:
            rgb_img = cv2.imread(str(rgb_path))
            if rgb_img is not None:
                # Convert to grayscale
                grayscale = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

                # Use intensity-based crack detection within mask
                # This measures actual dark crack pixels, not mask boundary
                # detection_method and sample_interval will be passed from caller
                avg_width_mm, max_width_mm = calculate_intensity_based_width(
                    skeleton, grayscale, binary_mask, scale_map,
                    sample_interval=sample_interval,
                    detection_method=detection_method
                )
                width_method = 'intensity'

    # Fallback to mask-based width if intensity detection failed or not used
    if avg_width_mm == 0.0 and max_width_mm == 0.0:
        avg_width_mm, max_width_mm = calculate_skeleton_width(
            skeleton, binary_mask, scale_map
        )
        width_method = 'mask'

    return {
        'length_mm': round(length_mm, 2),
        'avg_width_mm': round(avg_width_mm, 2),
        'max_width_mm': round(max_width_mm, 2),
        'width_method': width_method
    }


# =============================================================================
# Main Measurement Function
# =============================================================================

def measure_cluster(
    cluster: Dict,
    crack_points_lookup: Dict[int, Dict],
    masks_dir: Path,
    scale_maps_dir: Path,
    image_shape: Tuple[int, int],
    n_segments: int = 5,
    rgb_dir: Path = None,
    use_edge_width: bool = True,
    detection_method: str = 'percentile_30',
    sample_interval: int = 5
) -> Dict:
    """
    Measure a single cluster.

    Args:
        cluster: Cluster dict from crack_clusters.json
        crack_points_lookup: point_id -> point dict
        masks_dir: Path to YOLO masks
        scale_maps_dir: Path to scale map .npy files
        image_shape: (height, width)
        n_segments: Number of segments
        rgb_dir: Path to RGB images (for edge-based width)
        use_edge_width: Use edge detection for width measurement

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

        # Load scale map for this image
        # Extract timestamp part from image_id (e.g., camera_RGB_1761702052_213355008 -> 1761702052_213355008)
        timestamp_key = image_id
        for prefix in ['camera_RGB_', 'camera_DPT_']:
            if image_id.startswith(prefix):
                timestamp_key = image_id[len(prefix):]
                break

        # Try to find scale map file
        # d2c_and_pixel_scale.py saves as: scale_map_iso_camera_DPT_{timestamp}.npy
        scale_map = None
        scale_map_patterns = [
            f"scale_map_iso_camera_DPT_{timestamp_key}.npy",
            f"scale_map_iso_{image_id}.npy",
            f"scale_map_iso_{timestamp_key}.npy",
        ]

        for pattern in scale_map_patterns:
            scale_map_path = scale_maps_dir / pattern
            if scale_map_path.exists():
                scale_map = np.load(scale_map_path)
                break

        if scale_map is None:
            logger.warning(f"No scale map for {image_id}")
            continue

        # Extract UV coordinates for this segment from points that belong to best_mask
        segment_uvs = []
        for point in segment['points']:
            for source in point.get('source_masks', []):
                if source['image_id'] == image_id and source['mask_id'] == mask_id:
                    if 'uv' in source:
                        segment_uvs.append(source['uv'])
                    break

        # Measure in 2D with segment cropping and intensity-based width
        measurement = measure_segment_2d(
            masks_dir, image_id, mask_id, scale_map, segment_uvs, image_shape,
            margin=50, rgb_dir=rgb_dir, use_edge_width=use_edge_width,
            detection_method=detection_method, sample_interval=sample_interval
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
    scale_maps_dir: str,
    output_json: str,
    image_width: int = 3840,
    image_height: int = 2160,
    n_segments: int = 5,
    rgb_dir: str = None,
    use_edge_width: bool = True,
    detection_method: str = 'percentile',
    crack_percentile: float = 30.0,
    sample_interval: int = 5
):
    """
    Run measurement on all clusters using per-pixel scale maps.
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

    # Build lookup tables
    crack_points_lookup = {
        p['point_id']: p for p in crack_points_data['points']
    }

    masks_path = Path(masks_dir)
    scale_maps_path = Path(scale_maps_dir)
    rgb_path = Path(rgb_dir) if rgb_dir else None
    image_shape = (image_height, image_width)

    clusters = clusters_data.get('clusters', [])
    logger.info(f"Measuring {len(clusters)} clusters...")
    logger.info(f"Scale maps directory: {scale_maps_dir}")
    if rgb_path and use_edge_width:
        logger.info(f"Using intensity-based width measurement with RGB from: {rgb_dir}")
        logger.info(f"  Detection method: {detection_method}")
        if detection_method == 'percentile':
            logger.info(f"  Crack percentile: {crack_percentile}%")
        logger.info(f"  Sample interval: {sample_interval}")
    else:
        logger.info("Using mask-based width measurement")

    # Generate colors for clusters (same as visualization)
    cluster_colors = generate_cluster_colors(len(clusters))

    # Measure each cluster
    measurements = []

    for idx, cluster in enumerate(clusters):
        # Build detection method string with percentile value
        method_str = detection_method
        if detection_method == 'percentile':
            method_str = f'percentile_{crack_percentile}'

        measurement = measure_cluster(
            cluster,
            crack_points_lookup,
            masks_path,
            scale_maps_path,
            image_shape,
            n_segments,
            rgb_path,
            use_edge_width,
            method_str,
            sample_interval
        )

        # Add color information
        color_rgb = cluster_colors[idx] if idx < len(cluster_colors) else (128, 128, 128)
        color_name = rgb_to_color_name(color_rgb)
        measurement['color'] = {
            'rgb': list(color_rgb),
            'hex': f"#{color_rgb[0]:02X}{color_rgb[1]:02X}{color_rgb[2]:02X}",
            'name': color_name
        }

        measurements.append(measurement)

        logger.info(
            f"Cluster {measurement['cluster_id']} ({color_name}): "
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
    parser.add_argument('--scale-maps-dir', required=True,
                       help='Directory containing scale_map_iso_*.npy files')
    parser.add_argument('--output', required=True,
                       help='Output cluster_measurements.json')
    parser.add_argument('--image-width', type=int, default=3840)
    parser.add_argument('--image-height', type=int, default=2160)
    parser.add_argument('--n-segments', type=int, default=5,
                       help='Number of segments per cluster (default: 5)')
    parser.add_argument('--rgb-dir', default=None,
                       help='RGB images directory (for intensity-based width measurement)')
    parser.add_argument('--no-edge-width', action='store_true',
                       help='Disable intensity-based width measurement (use mask-based)')
    parser.add_argument('--detection-method', default='percentile',
                       choices=['percentile', 'adaptive', 'otsu'],
                       help='Crack detection method: percentile (darkest N%%), adaptive, otsu')
    parser.add_argument('--crack-percentile', type=float, default=30.0,
                       help='Percentile threshold for crack pixels (default: 30, lower=stricter)')
    parser.add_argument('--sample-interval', type=int, default=5,
                       help='Sample every N skeleton pixels for width (default: 5)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        measurements = run_measurement(
            args.clusters,
            args.crack_points,
            args.masks_dir,
            args.scale_maps_dir,
            args.output,
            args.image_width,
            args.image_height,
            args.n_segments,
            args.rgb_dir,
            not args.no_edge_width,
            args.detection_method,
            args.crack_percentile,
            args.sample_interval
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
