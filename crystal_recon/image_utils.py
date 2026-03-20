"""
image_utils.py — Image loading, resizing, display, and dataset validation utilities.
"""

import glob
import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def load_image(degrees: int, folder: str,
               scale_factor: float = 1.0) -> np.ndarray | None:
    """
    Load a crystal image for a given rotation angle.

    Images are expected to be named crystal_XXXX.jpg (zero-padded to 4 digits)
    inside the specified folder.

    Args:
        degrees:      Rotation angle in degrees (0–359).
        folder:       Path to the folder containing the image files.
        scale_factor: Fraction to resize the image by (e.g. 0.125 = 12.5%).
                      A value of 1.0 returns the image at its native resolution.

    Returns:
        The loaded (and optionally resized) image as a numpy array,
        or None if the file does not exist or cannot be read.
    """
    file_path = os.path.join(folder, f"crystal_{degrees:04d}.jpg")

    if not os.path.exists(file_path):
        return None

    img = cv.imread(file_path)

    if img is None:
        print(f"WARNING: Could not read image (may be corrupted): {file_path}")
        return None

    if scale_factor != 1.0:
        h, w = img.shape[:2]
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)

    return img


def resize_image(img: np.ndarray, new_resolution: tuple) -> np.ndarray:
    """
    Resize an image to the given (width, height) resolution.

    Args:
        img:            Input image as a numpy array.
        new_resolution: Target resolution as (width, height).

    Returns:
        Resized image as a numpy array.
    """
    width = int(new_resolution[0])
    height = int(new_resolution[1])
    return cv.resize(img.copy(), (width, height), interpolation=cv.INTER_AREA)


def show_image(title: str, img: np.ndarray,
               save_path: str | None = None) -> None:
    """
    Display or save an OpenCV image.

    Handles BGR-to-RGB conversion and grayscale images automatically.
    When running in a headless environment (e.g. a server), use save_path
    to write the image to disk instead of opening a GUI window.

    Args:
        title:     Window/figure title.
        img:       Image to display (BGR colour or grayscale numpy array).
        save_path: If provided, save the figure to this path instead of
                   displaying it interactively. Supports .png, .jpg, etc.
    """
    fig, ax = plt.subplots()

    if len(img.shape) == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    ax.set_title(title)
    ax.axis("off")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()


def image_count(folder: str) -> int:
    """
    Count the number of crystal_XXXX.jpg images in a folder.

    Args:
        folder: Path to the image folder.

    Returns:
        Number of matching image files found.
    """
    pattern = os.path.join(folder, "crystal_????.jpg")
    return len(glob.glob(pattern))


def validate_folder(folder: str, angle_step: int = 1,
                    total_degrees: int = 360) -> bool:
    """
    Validate that an image folder is ready for reconstruction.

    Checks:
      - The folder exists
      - At least one image is present
      - No gaps in the expected angle sequence

    Args:
        folder:        Path to the image folder.
        angle_step:    Expected step between image angles (degrees).
        total_degrees: Total rotation covered (almost always 360).

    Returns:
        True if the folder passes all checks, False otherwise.
        Prints a detailed report of any issues found.
    """
    if not os.path.isdir(folder):
        print(f"ERROR: Image folder not found: {folder}")
        return False

    count = image_count(folder)
    if count == 0:
        print(f"ERROR: No crystal_XXXX.jpg images found in: {folder}")
        return False

    expected_angles = list(range(0, total_degrees, angle_step))
    missing = []
    for angle in expected_angles:
        path = os.path.join(folder, f"crystal_{angle:04d}.jpg")
        if not os.path.exists(path):
            missing.append(angle)

    if missing:
        print(f"WARNING: {len(missing)} image(s) missing from {folder}:")
        # Show at most 10 missing angles to avoid flooding the terminal
        shown = missing[:10]
        print(f"  Missing angles: {shown}"
              + (" ..." if len(missing) > 10 else ""))
        print(f"  Reconstruction will skip these angles automatically.")

    total_expected = len(expected_angles)
    print(f"Dataset: {count}/{total_expected} images found in '{os.path.basename(folder)}'",
          end="")
    print(" (complete)" if not missing else f" ({len(missing)} missing)")

    return True
