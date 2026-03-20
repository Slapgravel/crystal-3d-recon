"""
reconstruct.py — Main entry point for the crystal 3D reconstruction pipeline.

Usage:
    python reconstruct.py --folder real_crystal_ring_white_light
    python reconstruct.py --folder real_crystal_ring_white_light --no-sam
    python reconstruct.py --folder my_crystal --scale 0.25 --z-step 2

Run with --help for all options.
"""

import argparse
import copy
import os
import sys

import numpy as np
from tqdm import tqdm

# Add parent directory to path so crystal_recon package is found
sys.path.insert(0, os.path.dirname(__file__))

from crystal_recon import config
from crystal_recon.image_utils import load_image
from crystal_recon.mesh import build_mesh, save_point_cloud
from crystal_recon.reconstruction import (
    contour_to_3d_with_depth,
    michelangelo,
    rotate_3d,
    scale_coords,
)
from crystal_recon.segmentation import (
    get_boundary,
    load_sam_model,
    make_mask_opencv,
    make_mask_sam,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Crystal 3D Reconstruction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with SAM segmentation (default)
  python reconstruct.py --folder real_crystal_ring_white_light

  # Use OpenCV thresholding instead of SAM (no model weights needed)
  python reconstruct.py --folder real_crystal_ring_white_light --no-sam

  # Higher quality reconstruction (slower)
  python reconstruct.py --folder my_crystal --scale 0.25 --z-step 2 --angle-step 1

  # Skip meshing, just generate the point cloud
  python reconstruct.py --folder my_crystal --no-mesh
        """
    )

    parser.add_argument(
        "--folder", "-f",
        required=True,
        help="Path to the folder containing crystal_XXXX.jpg images. "
             "Can be relative to notebooks/ or an absolute path."
    )
    parser.add_argument(
        "--no-sam",
        action="store_true",
        help="Use OpenCV thresholding instead of SAM for segmentation. "
             "Faster but less accurate. No model weights required."
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=config.SAM_CHECKPOINT,
        help=f"Path to SAM model checkpoint .pth file. "
             f"Default: {config.SAM_CHECKPOINT}"
    )
    parser.add_argument(
        "--sam-model",
        default=config.SAM_MODEL_TYPE,
        choices=["vit_b", "vit_l", "vit_h"],
        help=f"SAM model type. Default: {config.SAM_MODEL_TYPE}"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=config.IMAGE_SCALE_FACTOR,
        help=f"Image scale factor (e.g. 0.125 = 12.5%%). "
             f"Default: {config.IMAGE_SCALE_FACTOR}"
    )
    parser.add_argument(
        "--z-step",
        type=int,
        default=config.Z_AXIS_STEP_SIZE,
        help=f"Z-axis resolution of the 3D block (pixels). "
             f"Lower = finer detail, slower. Default: {config.Z_AXIS_STEP_SIZE}"
    )
    parser.add_argument(
        "--angle-step",
        type=int,
        default=config.RECONSTRUCTION_ANGLE_STEP,
        help=f"Degrees between reconstruction samples. "
             f"1 = all images, 5 = every 5th. Default: {config.RECONSTRUCTION_ANGLE_STEP}"
    )
    parser.add_argument(
        "--calibration",
        default=config.CALIBRATION_FILE,
        help=f"Path to camera calibration .pickle file. "
             f"Default: {config.CALIBRATION_FILE}"
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Skip pixel-to-mm scaling (output will be in pixel units)."
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help="Skip mesh generation. Only produce the .asc point cloud file."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open the Open3D viewer after meshing."
    )
    parser.add_argument(
        "--output-dir",
        default=config.OUTPUT_DIR,
        help=f"Directory for output files. Default: {config.OUTPUT_DIR}"
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        default=config.CAMERA_RESOLUTION,
        metavar=("WIDTH", "HEIGHT"),
        help=f"Camera resolution. Default: {config.CAMERA_RESOLUTION[0]} {config.CAMERA_RESOLUTION[1]}"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Resolve image folder path
# ---------------------------------------------------------------------------

def resolve_folder(folder: str) -> str:
    """
    Resolve the image folder path.

    Checks the given path directly, then looks inside notebooks/ as a fallback.
    """
    if os.path.isdir(folder):
        return folder

    # Try relative to notebooks/
    notebooks_path = os.path.join("notebooks", folder)
    if os.path.isdir(notebooks_path):
        return notebooks_path

    raise FileNotFoundError(
        f"Image folder not found: '{folder}'\n"
        f"Also tried: '{notebooks_path}'\n"
        f"Make sure the folder exists and contains crystal_XXXX.jpg files."
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):

    # --- Resolve folder ---
    folder = resolve_folder(args.folder)
    folder_name = os.path.basename(folder.rstrip("/\\"))
    print(f"\n{'='*60}")
    print(f"Crystal 3D Reconstruction Pipeline")
    print(f"{'='*60}")
    print(f"  Image folder:  {folder}")
    print(f"  Scale factor:  {args.scale}")
    print(f"  Z step:        {args.z_step}")
    print(f"  Angle step:    {args.angle_step}°")
    print(f"  Segmentation:  {'OpenCV (--no-sam)' if args.no_sam else 'SAM'}")
    print(f"{'='*60}\n")

    # --- Load SAM model once (if using SAM) ---
    sam = None
    if not args.no_sam:
        sam = load_sam_model(args.sam_checkpoint, args.sam_model)

    # --- Image dimensions ---
    w = int(args.resolution[0] * args.scale)
    h = int(args.resolution[1] * args.scale)

    # --- Initialise the 3D block ---
    z_min = -w
    z_max = w
    z_step = args.z_step
    angle_step = args.angle_step
    angle_max = 360

    block = None
    rotated_angle = 0

    # --- Pass 1: Build the initial block from all silhouettes ---
    print("Pass 1: Building 3D block from silhouettes...")
    angles = list(range(0, angle_max, angle_step))

    for angle in tqdm(angles, desc="Building", unit="img"):
        img = load_image(angle, folder, scale_factor=args.scale)
        if img is None:
            print(f"  WARNING: Skipping angle {angle}° (image not found)")
            continue

        # Segment
        if args.no_sam:
            mask = make_mask_opencv(img)
        else:
            mask = make_mask_sam(img, sam, config.SAM_BBOX)

        contour = get_boundary(mask, show=False)

        # Carve existing block
        if block is not None:
            block = michelangelo(contour, block)

        # Extrude contour into 3D shell and add to block
        shell = contour_to_3d_with_depth(contour, z_min, z_max, z_step)
        if block is None:
            block = shell.copy()
        else:
            block = np.vstack((block, shell))

        # Rotate block to next angle
        block = rotate_3d(block, -angle_step)
        rotated_angle -= angle_step

    # Rotate back to world orientation
    if abs(rotated_angle) != 360:
        block = rotate_3d(block, -rotated_angle)
    rotated_angle = 0

    # --- Pass 2: Second carving pass for cleaner result ---
    print("\nPass 2: Refining block with second carving pass...")
    for angle in tqdm(angles, desc="Refining", unit="img"):
        img = load_image(angle, folder, scale_factor=args.scale)
        if img is None:
            continue

        if args.no_sam:
            mask = make_mask_opencv(img)
        else:
            mask = make_mask_sam(img, sam, config.SAM_BBOX)

        contour = get_boundary(mask, show=False)

        block = michelangelo(contour, block)
        block = rotate_3d(block, -angle_step)
        rotated_angle -= angle_step

    # Sit the block on the Y=0 plane
    block[:, 1] -= np.min(block[:, 1])

    print(f"\nBlock complete: {len(block):,} points")

    # --- Scale to millimetres ---
    if args.no_scale:
        scaled_block = copy.deepcopy(block)
        print("Skipping scale calibration (--no-scale). Output in pixel units.")
    else:
        print("\nApplying scale calibration...")
        scaled_block = scale_coords(block, args.calibration, config.SCALE_CORRECTION)

    # --- Save point cloud ---
    os.makedirs(args.output_dir, exist_ok=True)
    asc_filename = f"{folder_name}-z={z_step},s={args.scale:.3f}.asc"
    asc_path = os.path.join(args.output_dir, asc_filename)
    save_point_cloud(scaled_block, asc_path)

    # --- Build mesh ---
    if not args.no_mesh:
        print("\nGenerating surface mesh...")
        output_base = os.path.join(args.output_dir,
                                   os.path.splitext(asc_filename)[0])
        results = build_mesh(
            np.asarray(scaled_block),
            output_base_path=output_base,
            visualize=args.visualize
        )

        print(f"\n{'='*60}")
        print("Reconstruction complete!")
        print(f"{'='*60}")
        print(f"  Point cloud:   {asc_path}")
        print(f"  Raw mesh:      {results['mesh_file']}")
        print(f"  Repaired mesh: {results['fixed_mesh_file']}")
        print(f"  Watertight:    {results['is_watertight']}")
        if results["volume_mm3"] is not None:
            print(f"  Volume:        {results['volume_mm3']:.2f} mm³")
        print(f"  Surface area:  {results['surface_area_mm2']:.2f} mm²")
    else:
        print(f"\nReconstruction complete! Point cloud: {asc_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    try:
        run_pipeline(args)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
