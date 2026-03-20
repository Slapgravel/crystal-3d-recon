"""
reconstruction.py — 3D geometry, voxel carving, and scale calibration.

This module implements the core "visual hull" reconstruction algorithm:
  1. Convert a 2D crystal contour into a 3D shell at multiple Z depths
  2. Rotate the shell to match the camera angle for each image
  3. Carve away voxels that fall outside the crystal silhouette (Michelangelo)
  4. Scale the resulting point cloud from pixels to millimetres
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
        coords:    (N, 3) numpy matrix of (x, y, z) points.
        angle_deg: Rotation angle in degrees. Positive = counter-clockwise
                   when viewed from above.

    Returns:
        Rotated (N, 3) numpy matrix.
    """
    theta = angle_deg * np.pi / 180.0

    Ry = np.matrix([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)],
    ])

    rotated = Ry * coords.T
    return rotated.T


# ---------------------------------------------------------------------------
# Contour → 3D shell
# ---------------------------------------------------------------------------

def contour_to_3d_with_depth(contour_2d: np.ndarray,
                              z_min: int, z_max: int,
                              z_step: int) -> np.ndarray:
    """
    Extrude a 2D crystal contour into a 3D shell across multiple Z planes.

    For each Z value in [z_min, z_max], a copy of the 2D contour is placed
    at that depth, forming a cylindrical shell. This represents the initial
    "uncarved block" for one camera angle.

    Args:
        contour_2d: (N, 2) numpy matrix of (x, y) contour points.
        z_min:      Minimum Z value (in pixels).
        z_max:      Maximum Z value (in pixels).
        z_step:     Step size between Z planes.

    Returns:
        (N * Nz, 3) numpy matrix of (x, y, z) points.
    """
    num_pts = contour_2d.shape[0]

    # Build the base 3D contour at z=0
    z_zeros = np.matrix(np.zeros((num_pts, 1)))
    contour_3d = np.hstack((contour_2d[:, 0], contour_2d[:, 1], z_zeros))

    # Z values to extrude across
    z_values = np.arange(z_min, z_max + z_step, z_step)
    n_z = len(z_values)

    # Tile the contour across all Z planes
    contour_deep = np.tile(contour_3d, (n_z, 1))

    # Assign Z values
    i = 0
    for iz in range(n_z):
        for _ in range(num_pts):
            contour_deep[i, 2] = z_values[iz]
            i += 1

    return contour_deep


# ---------------------------------------------------------------------------
# Voxel carving (the "Michelangelo" step)
# ---------------------------------------------------------------------------

def michelangelo(contour: np.ndarray, block: np.ndarray) -> np.ndarray:
    """
    Carve away points from the block that fall outside the crystal silhouette.

    Named after Michelangelo's description of sculpture: "I saw the angel in
    the marble and carved until I set him free." Each rotated view removes
    the parts of the block that cannot be part of the crystal.

    The carving is done from the camera's perspective (along the Z axis),
    using OpenCV's pointPolygonTest to determine if each point lies inside
    the crystal contour.

    Args:
        contour: (N, 2) numpy matrix of (x, y) contour points (the silhouette).
        block:   (M, 3) numpy matrix of (x, y, z) points (the current block).

    Returns:
        Filtered (K, 3) numpy matrix containing only points inside the contour.
    """
    # Convert contour to the format expected by pointPolygonTest
    contour_pts = np.array(contour, dtype=np.float32).reshape((-1, 1, 2))

    new_block = []
    for pt in block:
        x = float(pt[0, 0])
        y = float(pt[0, 1])
        z = float(pt[0, 2])

        # Test if (x, y) is inside the contour polygon
        result = cv.pointPolygonTest(contour_pts, (x, y), measureDist=False)
        if result >= 0:
            new_block.append((x, y, z))

    return np.matrix(new_block)


# ---------------------------------------------------------------------------
# Scale calibration
# ---------------------------------------------------------------------------

def scale_coords(block: np.ndarray, calibration_file: str,
                 correction: float = 0.88) -> np.ndarray:
    """
    Convert point cloud coordinates from pixels to millimetres.

    Loads the scaling factor from the camera calibration pickle file and
    applies it to all coordinates. An optional manual correction factor
    can be applied on top (default 0.88, empirically derived).

    Args:
        block:            (N, 3) numpy matrix of (x, y, z) pixel coordinates.
        calibration_file: Path to the camera calibration .pickle file.
        correction:       Manual scale correction multiplier. Set to 1.0 to
                          disable. See config.py SCALE_CORRECTION.

    Returns:
        (N, 3) numpy matrix of (x, y, z) coordinates in millimetres.

    Raises:
        FileNotFoundError: If the calibration file does not exist.
    """
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(
            f"Calibration file not found: {calibration_file}\n"
            "Run camera_capture.py with --calibrate first, or provide the "
            "path to an existing calibration file via --calibration."
        )

    with open(calibration_file, "rb") as f:
        data = pickle.load(f)

    scaling = data["scalingFactor"]  # mm per pixel
    scaling *= correction

    print(f"Applying scale factor: {scaling:.6f} mm/pixel "
          f"(base: {data['scalingFactor']:.6f}, correction: {correction})")

    scaled = copy.deepcopy(block)
    scaled *= scaling
    return scaled
