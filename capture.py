"""
capture.py — Crystal image acquisition using the Zaber rotation stage and camera.

Supports three modes:
  - Real hardware mode: controls the Zaber XRSW60AE03 stage and an Allied
    Vision GenICam camera. Both are auto-discovered at startup.
  - Camera-only mode (--no-stage): uses the real camera but skips all Zaber
    stage hardware. The crystal is not rotated; one image is captured per
    nominal angle step. Use this to verify the camera is working before the
    stage is connected.
  - Simulator mode (--simulate): generates synthetic test images without
    any hardware connected. Works on any machine.

Usage:
    # List all detected cameras and Zaber stages (no capture)
    python capture.py --list-cameras

    # Real hardware capture (auto-discovers camera and stage)
    python capture.py --output real_crystal_ring_white_light

    # Camera only — verify camera works before stage is connected
    python capture.py --output camera_test --no-stage

    # Camera only, single shot (step=360 captures just one image)
    python capture.py --output camera_test --no-stage --step 360

    # Simulate (any machine, no hardware required)
    python capture.py --output test_crystal --simulate

    # Simulate calibration images
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
  # Check what hardware is connected before capturing
  python capture.py --list-cameras

  # Capture real crystal images (auto-discovers camera and stage)
  python capture.py --output real_crystal_ring_white_light

  # Specify camera and port explicitly
  python capture.py --output my_crystal --camera 0 --port COM3

  # Generate synthetic test images (no hardware needed)
  python capture.py --output test_crystal --simulate

  # Simulate calibration images
  python capture.py --output cal_images2 --calibrate --simulate
        """
    )
    parser.add_argument(
        "--output", "-o",
        help="Output folder name. Images saved as crystal_XXXX.jpg inside "
             "notebooks/<output>/. Not required when using --list-cameras."
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
        "--camera",
        type=int,
        default=None,
        help="Camera index to use (from --list-cameras). "
             "If not specified, auto-selects if only one camera is found."
    )
    parser.add_argument(
        "--port",
        default=config.ZABER_PORT,
        help="Serial port for the Zaber stage (e.g. COM3 or /dev/ttyUSB0). "
             "If not specified, auto-discovers the stage."
    )
    parser.add_argument(
        "--cti",
        default=config.CAMERA_CTI_PATH,
        help="Path to the GenICam CTI file for the Allied Vision camera. "
             "If not specified, searches common installation paths."
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List all detected cameras and Zaber stages, then exit."
    )
    parser.add_argument(
        "--no-stage",
        action="store_true",
        help="Use the real camera but skip all Zaber stage hardware. "
             "The crystal is not rotated; one image is captured per nominal "
             "angle step. Use this to verify the camera before the stage is "
             "connected."
    )
    parser.add_argument(
        "--no-home",
        action="store_true",
        help="Skip homing the Zaber stage before capture. "
             "Use only if the stage is already at the home position."
    )
    parser.add_argument(
        "--output-dir",
        default=config.DATA_DIR,
        help=f"Parent directory for the output folder. Default: {config.DATA_DIR}"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hardware discovery
# ---------------------------------------------------------------------------

def find_cti_file(cti_path: str | None) -> str:
    """
    Find the GenICam CTI file for the Allied Vision camera.

    Checks the provided path first, then searches common installation
    locations on Windows and Linux.

    Args:
        cti_path: Path from config or CLI argument. None = auto-search.

    Returns:
        Path to the CTI file.

    Raises:
        FileNotFoundError: If no CTI file can be found.
    """
    if cti_path and os.path.exists(cti_path):
        return cti_path

    candidates = [
        # Windows — Vimba X (cti\ subfolder — USB and GigE)
        r"C:\Program Files\Allied Vision\Vimba X\cti\VimbaUSBTL.cti",
        r"C:\Program Files\Allied Vision\Vimba X\cti\VimbaGigETL.cti",
        r"C:\Program Files\Allied Vision\Vimba X\cti\VimbaCameraSimulatorTL.cti",
        # Windows — Vimba X (legacy api\bin\ path)
        r"C:\Program Files\Allied Vision\Vimba X\api\bin\VimbaC.cti",
        r"C:\Program Files\Allied Vision\VimbaX\api\bin\VimbaC.cti",
        # Windows — Vimba 6
        r"C:\Program Files\Allied Vision\Vimba_6\VimbaC\Bin\Win64\VimbaC.cti",
        # macOS — Vimba X
        "/Library/Frameworks/VimbaX.framework/Resources/cti/VimbaUSBTL.cti",
        "/Library/Frameworks/VimbaX.framework/Resources/cti/VimbaGigETL.cti",
        # Linux
        "/opt/VimbaX/cti/VimbaUSBTL.cti",
        "/opt/VimbaX/cti/VimbaGigETL.cti",
        "/opt/VimbaX/api/lib/libVimbaC.so",
        "/opt/Vimba_6/VimbaC/DynamicLib/x86_64bit/libVimbaC.so",
    ]

    for path in candidates:
        if os.path.exists(path):
            print(f"  Found CTI file: {path}")
            return path

    raise FileNotFoundError(
        "Could not find the Allied Vision GenICam CTI file.\n"
        "Install the Vimba X SDK from: https://www.alliedvision.com/en/products/vimba-sdk/\n"
        "Or specify the path with: --cti <path_to_VimbaC.cti>"
    )


def discover_cameras(cti_path: str) -> list:
    """
    Discover all available GenICam cameras using the Harvester library.

    Args:
        cti_path: Path to the CTI file.

    Returns:
        List of dicts with camera info: [{"index": 0, "model": "...", "serial": "..."}]

    Raises:
        ImportError: If harvesters is not installed.
    """
    try:
        from harvesters.core import Harvester
    except ImportError:
        raise ImportError(
            "harvesters is not installed.\n"
            "Install with: pip install harvesters\n"
            "Also install the Allied Vision Vimba X SDK."
        )

    h = Harvester()
    h.add_file(cti_path)
    h.update()

    cameras = []
    for i, info in enumerate(h.device_info_list):
        cameras.append({
            "index": i,
            "model": getattr(info, "model", "Unknown"),
            "serial": getattr(info, "serial_number", "Unknown"),
            "vendor": getattr(info, "vendor", "Unknown"),
        })

    h.reset()
    return cameras


def discover_zaber_stages(port: str | None) -> list:
    """
    Discover Zaber rotation stages on available serial ports.

    If a port is specified, only that port is checked. Otherwise, all
    available serial ports are scanned.

    Args:
        port: Specific port to check, or None to scan all ports.

    Returns:
        List of dicts: [{"port": "COM3", "device": "...", "axis": 1}]

    Raises:
        ImportError: If zaber_motion is not installed.
    """
    try:
        from zaber_motion.ascii import Connection as ZaberConnection
        import serial.tools.list_ports
    except ImportError:
        raise ImportError(
            "zaber_motion or pyserial is not installed.\n"
            "Install with: pip install zaber-motion pyserial"
        )

    ports_to_check = [port] if port else [
        p.device for p in serial.tools.list_ports.comports()
    ]

    found = []
    for p in ports_to_check:
        try:
            with ZaberConnection.open_serial_port(p) as conn:
                devices = conn.detect_devices()
                for dev in devices:
                    found.append({
                        "port": p,
                        "device": dev.name,
                        "axes": dev.axis_count,
                    })
        except Exception:
            pass  # Port not a Zaber device or unavailable

    return found


# ---------------------------------------------------------------------------
# List hardware and exit
# ---------------------------------------------------------------------------

def list_hardware(args):
    """Print all detected cameras and Zaber stages, then exit."""
    print("\n=== Detecting hardware ===\n")

    # Cameras
    print("Cameras:")
    try:
        cti = find_cti_file(args.cti)
        cameras = discover_cameras(cti)
        if cameras:
            for cam in cameras:
                print(f"  [{cam['index']}] {cam['vendor']} {cam['model']} "
                      f"(serial: {cam['serial']})")
        else:
            print("  No cameras found.")
    except (FileNotFoundError, ImportError) as e:
        print(f"  Could not detect cameras: {e}")

    # Zaber stages
    print("\nZaber stages:")
    try:
        stages = discover_zaber_stages(args.port)
        if stages:
            for s in stages:
                print(f"  Port: {s['port']}  Device: {s['device']}  "
                      f"Axes: {s['axes']}")
        else:
            print("  No Zaber stages found.")
    except (ImportError) as e:
        print(f"  Could not detect Zaber stages: {e}")

    print()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Real hardware capture
# ---------------------------------------------------------------------------

def capture_with_hardware(output_folder: str, step: int, calibrate: bool,
                          camera_index: int | None, port: str | None,
                          cti_path: str | None, home: bool):
    """
    Capture images using the real Zaber stage and Allied Vision camera.

    Auto-discovers hardware if camera_index or port are not specified.
    Prompts for confirmation before homing the stage.

    Args:
        output_folder:  Path to save images into.
        step:           Rotation step in degrees.
        calibrate:      If True, capture calibration images instead.
        camera_index:   Camera index (from --list-cameras), or None to auto-select.
        port:           Zaber serial port, or None to auto-discover.
        cti_path:       Path to CTI file, or None to auto-find.
        home:           If True, home the stage before capture.
    """
    try:
        from harvesters.core import Harvester
        from zaber_motion.ascii import Connection as ZaberConnection
        from zaber_motion import Units
    except ImportError as e:
        raise ImportError(
            f"{e}\nUse --simulate for testing without hardware."
        )

    # --- Find CTI file ---
    cti = find_cti_file(cti_path)

    # --- Discover cameras ---
    cameras = discover_cameras(cti)
    if not cameras:
        raise RuntimeError(
            "No cameras detected. Check that the camera is connected and "
            "the Vimba SDK is installed."
        )

    if camera_index is None:
        if len(cameras) == 1:
            camera_index = 0
            print(f"Auto-selected camera: {cameras[0]['vendor']} "
                  f"{cameras[0]['model']}")
        else:
            print("Multiple cameras detected:")
            for cam in cameras:
                print(f"  [{cam['index']}] {cam['vendor']} {cam['model']} "
                      f"(serial: {cam['serial']})")
            camera_index = int(input("Enter camera index to use: ").strip())

    # --- Discover Zaber stage ---
    stages = discover_zaber_stages(port)
    if not stages:
        raise RuntimeError(
            "No Zaber stage detected. Check that the stage is connected "
            "and the correct serial port is available.\n"
            "Use --port to specify the port explicitly."
        )

    zaber_port = stages[0]["port"]
    if len(stages) > 1:
        print("Multiple Zaber stages detected:")
        for i, s in enumerate(stages):
            print(f"  [{i}] Port: {s['port']}  Device: {s['device']}")
        idx = int(input("Enter stage index to use: ").strip())
        zaber_port = stages[idx]["port"]

    print(f"Using Zaber stage on port: {zaber_port}")

    # --- Connect to Zaber stage ---
    with ZaberConnection.open_serial_port(zaber_port) as conn:
        devices = conn.detect_devices()
        stage = devices[0].get_axis(1)

        if home:
            confirm = input(
                "\nAbout to HOME the rotation stage. "
                "Make sure the crystal is clear. Continue? [y/N]: "
            ).strip().lower()
            if confirm != "y":
                print("Homing cancelled. Exiting.")
                sys.exit(0)
            print("Homing stage...")
            stage.home()
            print("Stage homed.")

        # --- Connect to camera ---
        h = Harvester()
        h.add_file(cti)
        h.update()

        with h.create(search_key={"index": camera_index}) as ia:
            ia.remote_device.node_map.ExposureTime.value = config.CAMERA_EXPOSURE_US
            ia.remote_device.node_map.Gain.value = config.CAMERA_GAIN_DB
            ia.start()

            angles = (list(range(0, 360, step))
                      if not calibrate else list(range(12)))
            total = len(angles)

            print(f"\nCapturing {total} "
                  f"{'calibration' if calibrate else 'crystal'} images...")

            for i, angle in enumerate(angles):
                if not calibrate:
                    stage.move_absolute(angle, Units.ANGLE_DEGREES)
                    time.sleep(0.3)  # Allow stage to settle

                with ia.fetch() as buffer:
                    component = buffer.payload.components[0]
                    h_img = component.height
                    w_img = component.width
                    img_data = component.data.reshape(h_img, w_img)

                    # Convert mono16 to 8-bit for saving
                    img_8bit = (img_data / 256).astype(np.uint8)
                    img_bgr = cv.cvtColor(img_8bit, cv.COLOR_GRAY2BGR)

                if calibrate:
                    filename = f"CameraCalibration_{i:04d}.jpg"
                else:
                    filename = f"crystal_{angle:04d}.jpg"

                path = os.path.join(output_folder, filename)
                cv.imwrite(path, img_bgr)

                if i % 10 == 0 or i == total - 1:
                    print(f"  {i+1}/{total} — {filename}")

            ia.stop()

        h.reset()

    print(f"\nCapture complete. {total} images saved to {output_folder}/")


# ---------------------------------------------------------------------------
# Camera-only capture (no stage)
# ---------------------------------------------------------------------------

def capture_camera_only(output_folder: str, step: int, calibrate: bool,
                        camera_index: int | None, cti_path: str | None):
    """
    Capture images using the real camera with no stage movement.

    The crystal is NOT rotated between shots. Each image in the sequence is
    taken from the same position, labelled with the nominal angle that would
    have been used if the stage were connected. This lets you verify that the
    camera is working, the exposure and gain settings are correct, and that
    images are being saved properly — all before the Zaber stage is connected.

    Typical usage:
        # Capture a full 360-image sequence (all from the same position)
        python capture.py --output camera_test --no-stage

        # Capture just one image to check the camera quickly
        python capture.py --output camera_test --no-stage --step 360

    Args:
        output_folder:  Path to save images into.
        step:           Nominal rotation step in degrees (controls how many
                        images are taken, not actual movement).
        calibrate:      If True, capture calibration images instead.
        camera_index:   Camera index (from --list-cameras), or None to
                        auto-select.
        cti_path:       Path to CTI file, or None to auto-find.
    """
    try:
        from harvesters.core import Harvester
    except ImportError as e:
        raise ImportError(
            f"{e}\nThe harvesters package is required for camera capture.\n"
            "Install with: pip install harvesters\n"
            "Also ensure the Allied Vision Vimba X SDK is installed."
        )

    # --- Find CTI file ---
    cti = find_cti_file(cti_path)

    # --- Discover cameras ---
    cameras = discover_cameras(cti)
    if not cameras:
        raise RuntimeError(
            "No cameras detected. Check that the camera is connected and "
            "the Vimba SDK is installed."
        )
    if camera_index is None:
        if len(cameras) == 1:
            camera_index = 0
            print(f"Auto-selected camera: {cameras[0]['vendor']} "
                  f"{cameras[0]['model']}")
        else:
            print("Multiple cameras detected:")
            for cam in cameras:
                print(f"  [{cam['index']}] {cam['vendor']} {cam['model']} "
                      f"(serial: {cam['serial']})")
            camera_index = int(input("Enter camera index to use: ").strip())

    print("\nNOTE: --no-stage mode — stage is NOT connected. "
          "Crystal will not rotate between shots.")

    # --- Connect to camera and capture ---
    h = Harvester()
    h.add_file(cti)
    h.update()
    with h.create(search_key={"index": camera_index}) as ia:
        ia.remote_device.node_map.ExposureTime.value = config.CAMERA_EXPOSURE_US
        ia.remote_device.node_map.Gain.value = config.CAMERA_GAIN_DB
        ia.start()

        angles = (list(range(0, 360, step))
                  if not calibrate else list(range(12)))
        total = len(angles)
        print(f"\nCapturing {total} "
              f"{'calibration' if calibrate else 'crystal'} images "
              f"(no stage movement)...")

        for i, angle in enumerate(angles):
            with ia.fetch() as buffer:
                component = buffer.payload.components[0]
                h_img = component.height
                w_img = component.width
                img_data = component.data.reshape(h_img, w_img)
                # Convert mono16 to 8-bit for saving
                img_8bit = (img_data / 256).astype(np.uint8)
                img_bgr = cv.cvtColor(img_8bit, cv.COLOR_GRAY2BGR)

            if calibrate:
                filename = f"CameraCalibration_{i:04d}.jpg"
            else:
                filename = f"crystal_{angle:04d}.jpg"
            path = os.path.join(output_folder, filename)
            cv.imwrite(path, img_bgr)
            if i % 10 == 0 or i == total - 1:
                print(f"  {i+1}/{total} — {filename}  "
                      f"{w_img}x{h_img}px")

        ia.stop()
    h.reset()

    print(f"\nCapture complete. {total} images saved to {output_folder}/")
    print("Tip: open the images to verify focus, exposure, and framing "
          "before connecting the stage.")


# ---------------------------------------------------------------------------
# Simulator: synthetic crystal images
# ---------------------------------------------------------------------------

def simulate_crystal_image(angle_deg: float,
                            resolution: tuple = (666, 576)) -> np.ndarray:
    """
    Generate a synthetic crystal image for a given rotation angle.

    Creates a simple ellipse that changes shape as the angle varies,
    simulating the silhouette of a crystal on a rotation stage.

    Args:
        angle_deg:  Rotation angle in degrees (0–359).
        resolution: Image size as (width, height).

    Returns:
        Synthetic BGR image as a numpy array.
    """
    w, h = resolution
    img = np.zeros((h, w, 3), dtype=np.uint8)

    theta = np.radians(angle_deg)
    cx, cy = w // 2, h // 2

    # Semi-axes vary with rotation to simulate 3D foreshortening
    a = int(w * 0.15 * (0.4 + 0.6 * abs(np.cos(theta))))
    b = int(h * 0.30)

    cv.ellipse(img, (cx, cy), (max(a, 5), max(b, 5)), 0, 0, 360,
               (220, 220, 220), -1)

    # Add a highlight facet
    hx = cx + int(a * 0.3 * np.cos(theta + 0.5))
    hy = cy - int(b * 0.2)
    cv.ellipse(img, (hx, hy), (max(a // 4, 3), max(b // 6, 3)),
               0, 0, 360, (255, 255, 255), -1)

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
    img = np.ones((h, w, 3), dtype=np.uint8) * 200

    square_size = 40
    cols, rows = 7, 6
    board_w = cols * square_size
    board_h = rows * square_size

    rng = np.random.default_rng(index)
    offset_x = int(rng.integers(20, max(21, w - board_w - 20)))
    offset_y = int(rng.integers(20, max(21, h - board_h - 20)))

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
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Handle --list-cameras before requiring --output
    if args.list_cameras:
        list_hardware(args)
        return  # list_hardware calls sys.exit, but just in case

    if not args.output:
        print("ERROR: --output is required unless using --list-cameras.")
        sys.exit(1)

    output_folder = os.path.join(args.output_dir, args.output)
    os.makedirs(output_folder, exist_ok=True)
    print(f"\nOutput folder: {output_folder}")

    if args.simulate:
        resolution = (
            int(config.CAMERA_RESOLUTION[0] * config.IMAGE_SCALE_FACTOR),
            int(config.CAMERA_RESOLUTION[1] * config.IMAGE_SCALE_FACTOR),
        )

        if args.calibrate:
            n_images = 12
            print(f"Simulating {n_images} calibration images "
                  f"at {resolution[0]}x{resolution[1]}...")
            for i in range(n_images):
                img = simulate_calibration_image(i, resolution)
                path = os.path.join(output_folder,
                                    f"CameraCalibration_{i:04d}_simulated.png")
                cv.imwrite(path, img)
                print(f"  Saved: {os.path.basename(path)}")
        else:
            angles = list(range(0, 360, args.step))
            print(f"Simulating {len(angles)} crystal images "
                  f"at {resolution[0]}x{resolution[1]}...")
            for angle in angles:
                img = simulate_crystal_image(angle, resolution)
                path = os.path.join(output_folder, f"crystal_{angle:04d}.jpg")
                cv.imwrite(path, img)
                if angle % 30 == 0:
                    print(f"  {angle}°...")
            print(f"Done. {len(angles)} images saved to {output_folder}/")

    elif args.no_stage:
        print("Connecting to camera (stage skipped)...")
        try:
            capture_camera_only(
                output_folder=output_folder,
                step=args.step,
                calibrate=args.calibrate,
                camera_index=args.camera,
                cti_path=args.cti,
            )
        except (ImportError, RuntimeError, FileNotFoundError) as e:
            print(f"\nERROR: {e}")
            print("\nTip: Use --list-cameras to check what cameras are "
                  "detected, or --simulate to test without any hardware.")
            sys.exit(1)

    else:
        print("Connecting to hardware...")
        try:
            capture_with_hardware(
                output_folder=output_folder,
                step=args.step,
                calibrate=args.calibrate,
                camera_index=args.camera,
                port=args.port,
                cti_path=args.cti,
                home=not args.no_home,
            )
        except (ImportError, RuntimeError, FileNotFoundError) as e:
            print(f"\nERROR: {e}")
            print("\nTip: Use --simulate to test without hardware, "
                  "or --list-cameras to check what is connected.")
            sys.exit(1)


if __name__ == "__main__":
    main()
