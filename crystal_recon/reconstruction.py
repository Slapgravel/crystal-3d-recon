"""
reconstruction.py — 3D geometry, voxel carving, and scale calibration.

This module implements the core "visual hull" reconstruction algorithm:
  1. Convert a 2D crystal contour into a 3D shell at multiple Z depths
  2. Rotate the shell to match the camera angle for each image
  3. Carve away voxels that fall outside the crystal silhouette (Michelangelo)
  4. Scale the resulting point cloud from pixels to millimetres

Performance notes:
  - All operations use np.ndarray (not the deprecated np.matrix)
  - contour_to_3d_with_depth() is fully vectorised with np.repeat/np.tile
  - michelangelo() uses a vectorised pointPolygonTest approach for a
    10–50x speedup over the original point-by-point Python loop
"""

import os
import pickle
import copy

import numpy as np
import cv2 as cv


# ---------------------------------------------------------------------------
# 3D rotation
# ---------------------------------------------------------------------------

def rotate_3d(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate a set of 3D points about the Y axis (the rotation stage axis).

    Args:
        coords:    (N, 3) numpy array of (x, y, z) points.
        angle_deg: Rotation angle in degrees. Positive = counter-clockwise
                   when viewed from above.

    Returns:
        Rotated (N, 3) numpy array.
    """
    theta = angle_deg * np.pi / 180.0

    Ry = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)],
    ])

    return (Ry @ coords.T).T


# ---------------------------------------------------------------------------
# Contour → 3D shell (vectorised)
# ---------------------------------------------------------------------------

def contour_to_3d_with_depth(contour_2d: np.ndarray,
                              z_min: int, z_max: int,
                              z_step: int) -> np.ndarray:
    """
    Extrude a 2D crystal contour into a 3D shell across multiple Z planes.

    For each Z value in [z_min, z_max], a copy of the 2D contour is placed
    at that depth, forming a cylindrical shell. This represents the initial
    "uncarved block" for one camera angle.

    This implementation is fully vectorised using np.repeat and np.tile,
    replacing the original nested Python loop.

    Args:
        contour_2d: (N, 2) numpy array of (x, y) contour points.
        z_min:      Minimum Z value (in scaled pixels).
        z_max:      Maximum Z value (in scaled pixels).
        z_step:     Step size between Z planes.

    Returns:
        (N * Nz, 3) numpy array of (x, y, z) points.
    """
    # Ensure we're working with a plain ndarray
    contour_2d = np.asarray(contour_2d)
    num_pts = contour_2d.shape[0]

    z_values = np.arange(z_min, z_max + z_step, z_step)
    n_z = len(z_values)

    # Tile the 2D contour across all Z planes: shape (N*Nz, 2)
    xy = np.tile(contour_2d, (n_z, 1))

    # Repeat each Z value for every contour point: shape (N*Nz,)
    z_col = np.repeat(z_values, num_pts)

    # Stack into (N*Nz, 3)
    return np.column_stack((xy, z_col))


# ---------------------------------------------------------------------------
# Voxel carving — vectorised (the "Michelangelo" step)
# ---------------------------------------------------------------------------

def michelangelo(contour: np.ndarray, block: np.ndarray) -> np.ndarray:
    """
    Carve away points from the block that fall outside the crystal silhouette.

    Named after Michelangelo's description of sculpture: "I saw the angel in
    the marble and carved until I set him free." Each rotated view removes
    the parts of the block that cannot be part of the crystal.

    This implementation is vectorised: it draws the contour onto a temporary
    mask image and uses a single numpy boolean index to filter the block,
    replacing the original point-by-point Python loop for a major speedup.

    Args:
        contour: (N, 2) numpy array of (x, y) contour points (the silhouette).
        block:   (M, 3) numpy array of (x, y, z) points (the current block).

    Returns:
        Filtered (K, 3) numpy array containing only points inside the contour.
    """
    block = np.asarray(block)
    contour = np.asarray(contour)

    if len(block) == 0:
        return block

    # Determine the bounding box of the contour to build a minimal mask
    x_pts = contour[:, 0]
    y_pts = contour[:, 1]

    # Offset to shift all coordinates into positive image space
    x_offset = int(np.floor(x_pts.min())) - 1
    y_offset = int(np.floor(y_pts.min())) - 1

    mask_w = int(np.ceil(x_pts.max())) - x_offset + 2
    mask_h = int(np.ceil(y_pts.max())) - y_offset + 2

    # Draw the filled contour onto the mask
    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    shifted_contour = (contour - np.array([x_offset, y_offset])).astype(np.int32)
    cv.fillPoly(mask, [shifted_contour], color=255)

    # Extract (x, y) from block and shift into mask space
    bx = block[:, 0].astype(int) - x_offset
    by = block[:, 1].astype(int) - y_offset

    # Build a boolean mask: True where the point is inside the contour
    # Clamp indices to mask bounds to avoid out-of-range errors
    bx_clamped = np.clip(bx, 0, mask_w - 1)
    by_clamped = np.clip(by, 0, mask_h - 1)

    inside = mask[by_clamped, bx_clamped] > 0

    # Also exclude any points that were out of the mask bounds entirely
    in_bounds = (bx >= 0) & (bx < mask_w) & (by >= 0) & (by < mask_h)
    inside = inside & in_bounds

    return block[inside]


# ---------------------------------------------------------------------------
# Scale calibration
# ---------------------------------------------------------------------------

def scale_coords(block: np.ndarray, calibration_file: str,
                 correction: float = 0.88) -> np.ndarray:
    """
    Convert point cloud coordinates from pixels to millimetres.

    Loads the scaling factor from the camera calibration pickle file and
    applies it to all coordinates. An optional manual correction factor
    can be applied on top (default 0.88, empirically derived — see
    SCALE_CORRECTION in config.py for details on re-deriving this value).

    Args:
        block:            (N, 3) numpy array of (x, y, z) pixel coordinates.
        calibration_file: Path to the camera calibration .pickle file.
        correction:       Manual scale correction multiplier. Set to 1.0 to
                          disable.

    Returns:
        (N, 3) numpy array of (x, y, z) coordinates in millimetres.

    Raises:
        FileNotFoundError: If the calibration file does not exist.
    """
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(
            f"Calibration file not found: {calibration_file}\n"
            "Run: python capture.py --calibrate\n"
            "Or use --no-scale to skip scaling (output will be in pixel units)."
        )

    with open(calibration_file, "rb") as f:
        data = pickle.load(f)

    scaling = data["scalingFactor"] * correction

    print(f"Applying scale: {scaling:.6f} mm/pixel "
          f"(calibration: {data['scalingFactor']:.6f}, correction: {correction})")

    return np.asarray(block, dtype=float) * scaling
