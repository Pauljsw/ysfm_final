"""
Point Cloud Mask Overlay

Overlays YOLO crack masks onto SFM sparse point cloud.
Uses COLMAP track information to map 2D pixels to 3D points.

Usage:
    python -m src.point_cloud_overlay \
        --sparse-dir data/sfm/sparse/0 \
        --masks-dir data/yolo_masks \
        --output outputs/sfm_masked_cloud.ply
"""
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon

from .colmap_io import read_points3D_binary, read_images_binary, read_cameras_binary

logger = logging.getLogger(__name__)


def load_yolo_mask(mask_path: Path) -> Dict:
    """Load YOLO mask JSON"""
    with open(mask_path, 'r') as f:
        return json.load(f)


def is_pixel_in_crack_mask(
    mask_json: Dict,
    pixel_xy: Tuple[float, float],
    image_stem: str = None,
    min_confidence: float = 0.0
) -> Tuple[bool, float]:
    """
    Check if pixel is inside any crack polygon with confidence validation.

    Args:
        mask_json: YOLO mask JSON data
        pixel_xy: Pixel coordinates (u, v)
        image_stem: Image identifier for logging (optional)
        min_confidence: Minimum YOLO confidence to consider (default: 0.0)

    Returns:
        (is_inside, confidence): True if pixel is in valid crack mask, and confidence score
    """
    if 'masks' not in mask_json:
        return False, 0.0

    point = ShapelyPoint(pixel_xy)

    for mask_idx, mask in enumerate(mask_json['masks']):
        if mask['class'] != 'crack':
            continue

        confidence = mask.get('score', 0.0)

        # Filter by minimum confidence
        if confidence < min_confidence:
            continue

        polygon_coords = mask['polygon']

        # Validate polygon has enough points
        if len(polygon_coords) < 3:
            if image_stem:
                logger.debug(f"[{image_stem}] Mask {mask_idx}: < 3 points, skipping")
            continue

        try:
            poly = Polygon(polygon_coords)

            # Auto-fix invalid polygons with buffer(0) trick
            if not poly.is_valid:
                try:
                    fixed_poly = poly.buffer(0)
                    
                    # 🆕 Verify the fix was successful
                    if not fixed_poly.is_valid or fixed_poly.is_empty:
                        if image_stem:
                            logger.debug(
                                f"[{image_stem}] Mask {mask_idx}: Cannot fix invalid polygon, skipping"
                            )
                        continue
                    
                    # Use fixed polygon
                    poly = fixed_poly
                    
                    if image_stem:
                        logger.debug(
                            f"[{image_stem}] Mask {mask_idx}: Auto-fixed invalid polygon "
                            f"(area={poly.area:.2f}, confidence={confidence:.3f})"
                        )
                        
                except Exception as e:
                    # If buffer(0) fails, skip this polygon
                    if image_stem:
                        logger.debug(
                            f"[{image_stem}] Mask {mask_idx}: Buffer fix failed ({e}), skipping"
                        )
                    continue

            # Check if polygon is too small (degenerate)
            if poly.area < 1.0:
                if image_stem:
                    logger.debug(
                        f"[{image_stem}] Mask {mask_idx}: Area too small ({poly.area:.2f}px²)"
                    )
                continue

            # 🆕 Final safety check before contains()
            if not poly.is_valid:
                if image_stem:
                    logger.warning(
                        f"[{image_stem}] Mask {mask_idx}: Still invalid after fix, skipping"
                    )
                continue

            if poly.contains(point):
                return True, confidence

        except Exception as e:
            if image_stem:
                logger.error(
                    f"[{image_stem}] Mask {mask_idx}: Unexpected error - {e}"
                )
            continue

    return False, 0.0


def overlay_masks_on_pointcloud(
    sparse_dir: str,
    masks_dir: str,
    output_ply: str,
    crack_color: Tuple[int, int, int] = (255, 0, 0),
    min_track_length: int = 2,
    vote_threshold: float = 0.5,
    min_confidence: float = 0.25
):
    """
    Overlay YOLO masks on SFM point cloud with improved voting mechanism.

    Args:
        sparse_dir: COLMAP sparse/0 directory
        masks_dir: YOLO masks directory
        output_ply: Output PLY path
        crack_color: RGB color for crack points (default: red)
        min_track_length: Minimum track length to include point
        vote_threshold: Minimum ratio of views that must agree (default: 0.5 = majority)
        min_confidence: Minimum YOLO confidence to consider (default: 0.25)
    """
    logger.info("=" * 80)
    logger.info("Point Cloud Mask Overlay")
    logger.info("=" * 80)

    sparse_path = Path(sparse_dir)
    masks_path = Path(masks_dir)

    # Read COLMAP data
    logger.info(f"Reading COLMAP data from: {sparse_dir}")
    points3D = read_points3D_binary(str(sparse_path / "points3D.bin"))
    images = read_images_binary(str(sparse_path / "images.bin"))
    cameras = read_cameras_binary(str(sparse_path / "cameras.bin"))

    logger.info(f"  3D Points: {len(points3D)}")
    logger.info(f"  Images: {len(images)}")
    logger.info(f"  Cameras: {len(cameras)}")

    # Build image name to ID mapping
    image_name_to_id = {img.name: img_id for img_id, img in images.items()}

    # Load all masks
    logger.info(f"Loading YOLO masks from: {masks_dir}")
    masks = {}
    for mask_file in masks_path.glob("*.json"):
        try:
            mask_data = load_yolo_mask(mask_file)
            # Match image name (without extension)
            img_stem = mask_file.stem
            masks[img_stem] = mask_data
        except Exception as e:
            logger.warning(f"Failed to load mask {mask_file}: {e}")

    logger.info(f"  Loaded {len(masks)} masks")

    # Process each 3D point with voting mechanism
    logger.info("Processing 3D points with voting...")
    logger.info(f"  Vote threshold: {vote_threshold} (need {vote_threshold*100:.0f}% agreement)")
    logger.info(f"  Min confidence: {min_confidence}")

    xyz_list = []
    rgb_list = []
    crack_count = 0
    skipped_count = 0

    # Statistics for quality metrics
    vote_counts = []
    confidence_scores = []
    view_counts = []

    for point_id, point in points3D.items():
        xyz = point.xyz
        original_rgb = point.rgb

        # Skip points with short tracks (likely noise)
        if len(point.image_ids) < min_track_length:
            skipped_count += 1
            continue

        # Voting mechanism: count votes across all views
        crack_votes = 0
        total_votes = 0
        confidence_sum = 0.0

        for img_id, point2D_idx in zip(point.image_ids, point.point2D_idxs):
            if img_id not in images:
                continue

            image = images[img_id]
            image_name = image.name
            image_stem = Path(image_name).stem

            # Get pixel coordinates
            if point2D_idx >= len(image.xys):
                continue

            pixel_xy = image.xys[point2D_idx]

            # Skip if no mask for this image
            if image_stem not in masks:
                continue

            total_votes += 1

            # Check mask with confidence filtering
            is_in_mask, mask_confidence = is_pixel_in_crack_mask(
                masks[image_stem],
                pixel_xy,
                image_stem,
                min_confidence
            )

            if is_in_mask:
                crack_votes += 1
                confidence_sum += mask_confidence

        # Decision based on voting
        if total_votes > 0:
            vote_ratio = crack_votes / total_votes
            avg_confidence = confidence_sum / crack_votes if crack_votes > 0 else 0.0

            is_crack = (vote_ratio >= vote_threshold and avg_confidence >= min_confidence)

            # Collect statistics
            view_counts.append(total_votes)
            if is_crack:
                vote_counts.append(crack_votes)
                confidence_scores.append(avg_confidence)
        else:
            is_crack = False

        # Assign color
        if is_crack:
            rgb = crack_color
            crack_count += 1
        else:
            rgb = original_rgb

        xyz_list.append(xyz)
        rgb_list.append(rgb)

    logger.info(f"  Total points: {len(xyz_list)}")
    logger.info(f"  Crack points: {crack_count} ({crack_count/len(xyz_list)*100:.1f}%)")
    logger.info(f"  Skipped (short track): {skipped_count}")

    # Compute quality metrics
    metrics = {
        'total_points': len(xyz_list),
        'crack_points': crack_count,
        'skipped_points': skipped_count,
        'crack_ratio': crack_count / len(xyz_list) if len(xyz_list) > 0 else 0,
    }

    # Voting statistics
    if vote_counts:
        metrics['avg_votes_per_crack'] = float(np.mean(vote_counts))
        metrics['median_votes_per_crack'] = float(np.median(vote_counts))
        metrics['min_votes'] = int(np.min(vote_counts))
        metrics['max_votes'] = int(np.max(vote_counts))

        logger.info(f"\n  Voting Statistics:")
        logger.info(f"    Avg votes per crack: {metrics['avg_votes_per_crack']:.1f}")
        logger.info(f"    Vote range: {metrics['min_votes']}-{metrics['max_votes']}")

    # Confidence statistics
    if confidence_scores:
        metrics['avg_confidence'] = float(np.mean(confidence_scores))
        metrics['min_confidence'] = float(np.min(confidence_scores))
        metrics['max_confidence'] = float(np.max(confidence_scores))

        logger.info(f"    Avg confidence: {metrics['avg_confidence']:.3f}")
        logger.info(f"    Confidence range: {metrics['min_confidence']:.3f}-{metrics['max_confidence']:.3f}")

    # View coverage
    if view_counts:
        metrics['avg_views_per_point'] = float(np.mean(view_counts))
        metrics['points_single_view'] = int(sum(1 for v in view_counts if v == 1))
        metrics['points_multi_view'] = int(sum(1 for v in view_counts if v >= 2))

        logger.info(f"    Avg views per point: {metrics['avg_views_per_point']:.1f}")

    # Quality warnings
    warnings = []

    if metrics.get('avg_confidence', 1.0) < 0.5:
        warnings.append(f"Low average confidence: {metrics['avg_confidence']:.3f}")

    if view_counts and metrics.get('points_single_view', 0) / len(view_counts) > 0.5:
        single_view_ratio = metrics['points_single_view'] / len(view_counts)
        warnings.append(f"High single-view ratio: {single_view_ratio*100:.1f}%")

    if metrics['crack_ratio'] > 0.3:
        warnings.append(f"Unusually high crack ratio: {metrics['crack_ratio']*100:.1f}%")

    if warnings:
        logger.warning("\n  Quality Warnings:")
        for warning in warnings:
            logger.warning(f"    ⚠️  {warning}")
        metrics['quality_warnings'] = warnings
    else:
        metrics['quality_warnings'] = []
        logger.info(f"\n  ✅ No quality warnings")

    # Save PLY
    logger.info(f"\nSaving point cloud to: {output_ply}")
    save_ply(output_ply, xyz_list, rgb_list)

    logger.info("=" * 80)
    logger.info("Point cloud overlay complete!")
    logger.info("=" * 80)

    return metrics


def save_ply(filename: str, xyz: List[np.ndarray], rgb: List[np.ndarray]):
    """
    Save point cloud as binary PLY.

    Args:
        filename: Output PLY path
        xyz: List of 3D coordinates
        rgb: List of RGB colors (uint8)
    """
    xyz = np.array(xyz)
    rgb = np.array(rgb, dtype=np.uint8)

    assert xyz.shape[0] == rgb.shape[0]
    assert xyz.shape[1] == 3
    assert rgb.shape[1] == 3

    # Create output directory
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write binary PLY
    with open(filename, 'wb') as f:
        # Header
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

        # Vertices
        for i in range(len(xyz)):
            # xyz as float32
            f.write(xyz[i].astype(np.float32).tobytes())
            # rgb as uint8
            f.write(rgb[i].tobytes())

    logger.info(f"Saved {len(xyz)} points to {filename}")


if __name__ == '__main__':
    import argparse
    from .utils import setup_logging

    parser = argparse.ArgumentParser(
        description='Overlay YOLO masks on SFM point cloud with voting mechanism'
    )
    parser.add_argument('--sparse-dir', required=True,
                       help='COLMAP sparse directory (e.g., data/sfm/sparse/0)')
    parser.add_argument('--masks-dir', required=True,
                       help='YOLO masks directory')
    parser.add_argument('--output', required=True,
                       help='Output PLY path')
    parser.add_argument('--crack-color', type=int, nargs=3, default=[255, 0, 0],
                       help='RGB color for crack points (default: 255 0 0)')
    parser.add_argument('--min-track-length', type=int, default=2,
                       help='Minimum track length (default: 2)')
    parser.add_argument('--vote-threshold', type=float, default=0.5,
                       help='Minimum vote ratio to mark as crack (default: 0.5 = majority)')
    parser.add_argument('--min-confidence', type=float, default=0.25,
                       help='Minimum YOLO confidence to consider (default: 0.25)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    setup_logging(args.log_level)

    try:
        result = overlay_masks_on_pointcloud(
            args.sparse_dir,
            args.masks_dir,
            args.output,
            tuple(args.crack_color),
            args.min_track_length,
            args.vote_threshold,
            args.min_confidence
        )

        print(f"\n✅ Point cloud overlay complete!")
        print(f"   Total points: {result['total_points']}")
        print(f"   Crack points: {result['crack_points']} ({result['crack_ratio']*100:.1f}%)")
        if 'avg_confidence' in result:
            print(f"   Avg confidence: {result['avg_confidence']:.3f}")
        if 'avg_votes_per_crack' in result:
            print(f"   Avg votes per crack: {result['avg_votes_per_crack']:.1f}")
        print(f"   Output: {args.output}")

    except Exception as e:
        logger.error(f"Overlay failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
