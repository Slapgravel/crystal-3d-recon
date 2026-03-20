"""
image_utils.py — Image loading, resizing, and display utilities.
"""

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def load_image(degrees: int, folder: str, scale_factor: float = 1.0) -> np.ndarray | None:
    """
    Load a crystal image for a given rotation angle.

    Images are expected to be named crystal_XXXX.jpg (zero-padded to 4 digits)
    inside the specified folder.

    Args:
        degrees:      Rotation angle in degrees (0–359).
        folder:       Path to the folder containing the image files.
        scale_factor: Fraction to resize the image by (e.g. 0.125 = 12.5%).

    Returns:
        The loaded (and optionally resized) image as a numpy array,
        or None if the file does not exist.
    """
    file_path = os.path.join(folder, f"crystal_{degrees:04d}.jpg")

    if not os.path.exists(file_path):
        print(f"ERROR: Image file not found: {file_path}")
        return None

    img = cv.imread(file_path)

    if img is None:
        print(f"ERROR: Could not read image: {file_path}")
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


def show_image(title: str, img: np.ndarray) -> None:
    """
    Display an OpenCV image using matplotlib.

    Handles BGR-to-RGB conversion and grayscale images automatically.

    Args:
        title: Window title.
        img:   Image to display (BGR colour or grayscale).
    """
    plt.figure()
    if len(img.shape) == 2:
        # Grayscale image
        plt.imshow(img, cmap="gray")
    else:
        # OpenCV stores images in BGR order — convert to RGB for matplotlib
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
