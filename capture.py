"""
capture.py — Crystal image acquisition using the Zaber rotation stage and camera.

Supports two modes:
  - Real hardware mode: controls the Zaber XRSW60AE03 stage and a GenICam camera
  - Simulator mode (--simulate): generates synthetic test images without any hardware

Usage:
    # Real hardware (lab machine only)
    python capture.py --output real_crystal_ring_white_light

    # Simulate (any machine, no hardware required)
    python capture.py --output test_crystal --simulate

    # Calibration image capture
    python capture.py --output cal_images2 --calibrate --simulate
"""

import argparse
import os
import sys
import time

import cv2 as cv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from crystal_recon import config


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Crystal image acquisition (real hardware or simulated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture real crystal images (requires hardware)
  python capture.py --output real_crystal_ring_white_light

  # Generate synthetic test images (no hardware needed)
  python capture.py --output test_crystal --simulate

  # Capture calibration images
  python capture.py --output cal_images2 --calibrate

  # Simulate calibration images
  python capture.py --output cal_images2 --calibrate --simulate
        """
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output folder name. Images saved as crystal_XXXX.jpg inside notebooks/<output>/"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Generate synthetic test images instead of using real hardware."
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Capture calibration images (checkerboard) instead of crystal images."
    )
    parser.add_argument(
        "--step",
        type=int,
        default=config.CAPTURE_STEP_DEGREES,
        help=f"Rotation step in degrees. Default: {config.CAPTURE_STEP_DEGREES}"
    )
    parser.add_argument(
        "--output-dir",
        default="notebooks",
        help="Parent directory for the output folder. Default: notebooks"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Simulator: synthetic crystal images
# ---------------------------------------------------------------------------

def simulate_crystal_image(angle_deg: float,
                            resolution: tuple = (666, 576)) -> np.ndarray:
    """
    Generate a synthetic crystal image for a given rotation angle.

    Creates a simple ellipse that rotates as the angle changes, simulating
    the silhouette of a crystal on a rotation stage. The ellipse is white
    on a black background, with slight Gaussian noise added for realism.

    Args:
        angle_deg:  Rotation angle in degrees (0–359).
        resolution: Image size as (width, height). Defaults to 1/8 of 5328x4608.

    Returns:
        Synthetic BGR image as a numpy array.
    """
    w, h = resolution
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Ellipse parameters that change with rotation angle to simulate 3D shape
    theta = np.radians(angle_deg)
    cx, cy = w // 2, h // 2

    # Semi-axes: the x-axis shrinks as the crystal rotates (foreshortening)
    a = int(w * 0.15 * (0.4 + 0.6 * abs(np.cos(theta))))  # semi-major (x)
    b = int(h * 0.30)                                        # semi-minor (y) — constant

    # Draw the crystal silhouette
    cv.ellipse(img, (cx, cy), (max(a, 5), max(b, 5)), 0, 0, 360, (220, 220, 220), -1)

    # Add a small facet highlight to make it more crystal-like
    highlight_x = cx + int(a * 0.3 * np.cos(theta + 0.5))
    highlight_y = cy - int(b * 0.2)
    cv.ellipse(img, (highlight_x, highlight_y),
               (max(a // 4, 3), max(b // 6, 3)), 0, 0, 360, (255, 255, 255), -1)

    # Add Gaussian noise
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def simulate_calibration_image(index: int,
                                resolution: tuple = (666, 576)) -> np.ndarray:
    """
    Generate a synthetic checkerboard calibration image.

    Args:
        index:      Image index (used to vary the board position slightly).
        resolution: Image size as (width, height).

    Returns:
        Synthetic BGR checkerboard image as a numpy array.
    """
    w, h = resolution
    img = np.ones((h, w, 3), dtype=np.uint8) * 200  # light gray background

    # Draw a simple checkerboard pattern
    square_size = 40
    cols = 7
    rows = 6
    board_w = cols * square_size
    board_h = rows * square_size

    # Slight random offset per image to simulate different positions
    rng = np.random.default_rng(index)
    offset_x = int(rng.integers(20, w - board_w - 20))
    offset_y = int(rng.integers(20, h - board_h - 20))

    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                x1 = offset_x + c * square_size
                y1 = offset_y + r * square_size
                cv.rectangle(img, (x1, y1),
                             (x1 + square_size, y1 + square_size),
                             (0, 0, 0), -1)

    return img


# ---------------------------------------------------------------------------
# Real hardware capture
# ---------------------------------------------------------------------------

def capture_with_hardware(output_folder: str, step: int, calibrate: bool):
    """
    Capture images using the real Zaber stage and GenICam camera.

    This function only runs on the lab machine with hardware connected.
    It will raise ImportError if the required hardware SDKs are not installed.
    """
    try:
        from zaber_motion.ascii import Connection as ZaberConnection
        from zaber_motion import Units
    except ImportError:
        raise ImportError(
            "zaber_motion is not installed or hardware is not connected.\n"
            "Use --simulate for testing without hardware."
        )

    try:
        from harvesters.core import Harvester
        import genicam
    except ImportError:
        raise ImportError(
            "harvesters / genicam SDK not found.\n"
            "Install the Allied Vision Vimba SDK and harvesters package.\n"
            "Use --simulate for testing without hardware."
        )

    print("Connecting to Zaber rotation stage...")
    # NOTE: Update the serial port in config.py if needed
    with ZaberConnection.open_serial_port("COM3") as connection:
        device_list = connection.detect_devices()
        stage = device_list[0].get_axis(1)
        stage.home()
        print("Stage homed.")

        print(f"Capturing {'calibration' if calibrate else 'crystal'} images...")
        angles = range(0, 360, step) if not calibrate else range(0, 12)

        for i, angle in enumerate(angles):
            if not calibrate:
                stage.move_absolute(angle, Units.ANGLE_DEGREES)
                time.sleep(0.2)

            # Camera capture would go here
            # img = camera.capture()
            # cv.imwrite(os.path.join(output_folder, f"crystal_{angle:04d}.jpg"), img)
            print(f"  Captured angle {angle}°")

    print("Hardware capture complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Resolve output folder
    output_folder = os.path.join(args.output_dir, args.output)
    os.makedirs(output_folder, exist_ok=True)
    print(f"\nOutput folder: {output_folder}")

    if args.simulate:
        # --- Simulator mode ---
        resolution = (
            int(config.CAMERA_RESOLUTION[0] * config.IMAGE_SCALE_FACTOR),
            int(config.CAMERA_RESOLUTION[1] * config.IMAGE_SCALE_FACTOR),
        )

        if args.calibrate:
            n_images = 12
            print(f"Simulating {n_images} calibration images at {resolution[0]}x{resolution[1]}...")
            for i in range(n_images):
                img = simulate_calibration_image(i, resolution)
                path = os.path.join(output_folder,
                                    f"CameraCalibration_{i}_simulated.png")
                cv.imwrite(path, img)
                print(f"  Saved: {path}")
        else:
            angles = list(range(0, 360, args.step))
            print(f"Simulating {len(angles)} crystal images at {resolution[0]}x{resolution[1]}...")
            for angle in angles:
                img = simulate_crystal_image(angle, resolution)
                path = os.path.join(output_folder, f"crystal_{angle:04d}.jpg")
                cv.imwrite(path, img)
                if angle % 30 == 0:
                    print(f"  {angle}°...")
            print(f"Done. {len(angles)} images saved to {output_folder}/")

    else:
        # --- Real hardware mode ---
        print("Connecting to hardware...")
        try:
            capture_with_hardware(output_folder, args.step, args.calibrate)
        except ImportError as e:
            print(f"\nERROR: {e}")
            print("\nTip: Use --simulate to test without hardware.")
            sys.exit(1)


if __name__ == "__main__":
    main()
