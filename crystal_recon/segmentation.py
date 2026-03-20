"""
segmentation.py — Crystal segmentation and contour extraction.

Provides two segmentation backends:
  - SAM (Segment Anything Model) — accurate, requires model weights
  - OpenCV threshold — fast fallback, no model weights needed

The SAM predictor is created once and passed in to make_mask_sam(),
avoiding the major performance issue in the original notebooks where
the model was reloaded from disk for every single image.
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from . import config


# ---------------------------------------------------------------------------
# SAM model + predictor loader
# ---------------------------------------------------------------------------

def load_sam_predictor(checkpoint_path: str, model_type: str):
    """
    Load the SAM model and return a ready-to-use SamPredictor.

    Call this once at startup and pass the returned predictor to
    make_mask_sam(). Creating the predictor here (rather than inside
    make_mask_sam) avoids re-initialising it on every image call.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        model_type:      Model variant — "vit_b", "vit_l", or "vit_h".

    Returns:
        A SamPredictor instance, ready to call .set_image() on.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ImportError:       If segment_anything is not installed.
    """
    import os

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint_path}\n"
            f"Download it with:\n"
            f"  wget https://dl.fbaipublicfiles.com/segment_anything/"
            f"sam_vit_b_01ec64.pth"
        )

    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        raise ImportError(
            "segment_anything is not installed.\n"
            "Install it with:\n"
            "  pip install git+https://github.com/facebookresearch/"
            "segment-anything.git"
        )

    print(f"Loading SAM model ({model_type}) from {checkpoint_path}...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    predictor = SamPredictor(sam)
    print("SAM model loaded.")
    return predictor


# ---------------------------------------------------------------------------
# SAM segmentation
# ---------------------------------------------------------------------------

def make_mask_sam(img: np.ndarray, predictor, bbox_fractions: list) -> np.ndarray:
    """
    Use the Segment Anything Model to create a binary mask of the crystal.

    The predictor is prompted with a centre-point (foreground) and a bounding
    box. Adjust bbox_fractions in config.py if the crystal is being clipped
    or if background is being included in the mask.

    Args:
        img:             Input image (BGR, numpy array).
        predictor:       Pre-loaded SamPredictor (from load_sam_predictor()).
        bbox_fractions:  [x_left, x_right, y_top, y_bottom] as fractions of
                         image width/height. e.g. [0.2, 0.8, 0.2, 0.85]

    Returns:
        Binary mask as a grayscale numpy array (255 = crystal, 0 = background).
    """
    h, w = img.shape[:2]

    # Centre-point prompt — assumes crystal is roughly centred in the frame
    center_x = int(w / 2)
    center_y = int(h / 2)
    input_point = np.array([[center_x, center_y]])
    input_label = np.array([1])  # 1 = foreground

    # Bounding box prompt derived from fractional config values
    x1 = int(w * bbox_fractions[0])
    x2 = int(w * bbox_fractions[1])
    y1 = int(h * bbox_fractions[2])
    y2 = int(h * bbox_fractions[3])
    input_box = np.array([x1, y1, x2, y2])

    # Convert BGR to RGB for SAM
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
    )

    mask = masks[0]

    # Convert boolean mask to binary grayscale image
    gray = np.zeros((h, w), dtype=np.uint8)
    gray[mask] = 255

    return gray


# ---------------------------------------------------------------------------
# OpenCV fallback segmentation (no SAM required)
# ---------------------------------------------------------------------------

def make_mask_opencv(img: np.ndarray) -> np.ndarray:
    """
    Create a binary mask using OpenCV thresholding (no SAM required).

    Attempts Otsu's global thresholding first. If the result looks poor
    (very few foreground pixels), falls back to adaptive thresholding,
    which handles uneven lighting better.

    This is a fallback for when SAM is not available. It works best for
    crystals photographed against a plain, contrasting background.

    Args:
        img: Input image (BGR, numpy array).

    Returns:
        Binary mask as a grayscale numpy array (255 = crystal, 0 = background).
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (11, 11), 0)

    # Try Otsu's global threshold first
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Check if the result is reasonable — at least 2% of pixels should be foreground
    foreground_ratio = np.count_nonzero(thresh) / thresh.size
    if foreground_ratio < 0.02 or foreground_ratio > 0.95:
        # Otsu failed — fall back to adaptive thresholding
        thresh = cv.adaptiveThreshold(
            blur, 255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            blockSize=51,
            C=-10
        )

    # Morphological cleanup — fill small holes and remove noise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=1)

    return thresh


# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

def get_boundary(mask: np.ndarray, show: bool = False,
                 title: str = "Contours") -> np.ndarray:
    """
    Trace the outer boundary of the crystal mask using contour detection.

    Returns the contour as (x, y) pairs centred about the image midpoint,
    with the Y axis inverted so that Y increases upward (standard math
    convention, matching the 3D coordinate system used in reconstruction).

    Args:
        mask:  Binary grayscale mask (255 = crystal, 0 = background).
        show:  If True, display the contour overlay using matplotlib.
        title: Plot title (used when show=True).

    Returns:
        Contour as an (N, 2) numpy matrix of (x, y) integer pairs,
        centred about the image midpoint.

    Raises:
        ValueError: If no valid contours are found, or if the largest
                    contour is smaller than MIN_CONTOUR_AREA (likely noise).
    """
    contours, _ = cv.findContours(
        image=mask,
        mode=cv.RETR_TREE,
        method=cv.CHAIN_APPROX_NONE
    )

    if not contours:
        raise ValueError(
            "No contours found in mask. The segmentation may have failed.\n"
            "Check the image quality or adjust SAM_BBOX / segmentation settings."
        )

    # Use the largest contour (the crystal outline)
    largest = max(contours, key=cv.contourArea)
    area = cv.contourArea(largest)

    if area < config.MIN_CONTOUR_AREA:
        raise ValueError(
            f"Largest contour area ({area:.0f} px²) is below MIN_CONTOUR_AREA "
            f"({config.MIN_CONTOUR_AREA} px²).\n"
            f"This image may be corrupted, overexposed, or the crystal may be "
            f"out of frame. Skipping."
        )

    x = largest[:, 0, 0]
    y = largest[:, 0, 1]

    if show:
        plt.figure()
        plt.imshow(mask, cmap="gray")
        plt.plot(x, y, "r.", markersize=1)
        plt.title(title)
        plt.axis("off")
        plt.show()

    h, w = mask.shape

    # Centre x about 0 (the axis of rotation)
    x = np.array(x - w / 2, dtype=int)

    # Invert y so it increases upward (matching 3D convention)
    y = h - y

    return np.matrix(np.column_stack((x, y)))
