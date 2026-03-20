"""
segmentation.py — Crystal segmentation and contour extraction.

Provides two segmentation backends:
  - SAM (Segment Anything Model) — accurate, requires model weights
  - OpenCV threshold — fast fallback, no model weights needed

The SAM model is loaded once and passed in, avoiding the major performance
issue in the original notebooks where it was reloaded for every single image.
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# SAM model loader
# ---------------------------------------------------------------------------

def load_sam_model(checkpoint_path: str, model_type: str):
    """
    Load the SAM model from a checkpoint file.

    Call this once at startup and pass the returned model to make_mask_sam().

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        model_type:      Model variant — "vit_b", "vit_l", or "vit_h".

    Returns:
        Loaded SAM model object.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ImportError:       If segment_anything is not installed.
    """
    import os
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint_path}\n"
            f"Download it with:\n"
            f"  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        )

    try:
        from segment_anything import sam_model_registry
    except ImportError:
        raise ImportError(
            "segment_anything is not installed.\n"
            "Install it with:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    print(f"Loading SAM model ({model_type}) from {checkpoint_path}...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    print("SAM model loaded.")
    return sam


# ---------------------------------------------------------------------------
# SAM segmentation
# ---------------------------------------------------------------------------

def make_mask_sam(img: np.ndarray, sam, bbox_fractions: list) -> np.ndarray:
    """
    Use the Segment Anything Model to create a binary mask of the crystal.

    The SAM model is prompted with a centre-point (foreground) and a bounding
    box. Adjust bbox_fractions in config.py if the crystal is being clipped
    or if background is being included in the mask.

    Args:
        img:             Input image (BGR, numpy array).
        sam:             Pre-loaded SAM model (from load_sam_model()).
        bbox_fractions:  [x1_frac, x2_frac, y1_frac, y2_frac] as fractions
                         of image width/height. e.g. [0.2, 0.8, 0.2, 0.85]

    Returns:
        Binary mask as a grayscale numpy array (255 = crystal, 0 = background).
    """
    from segment_anything import SamPredictor

    h, w = img.shape[:2]

    # Centre-point prompt — assumes crystal is roughly centred
    center_x = int(w / 2)
    center_y = int(h / 2)
    input_point = np.array([[center_x, center_y]])
    input_label = np.array([1])  # 1 = foreground

    # Bounding box prompt
    x1 = int(w * bbox_fractions[0])
    x2 = int(w * bbox_fractions[1])
    y1 = int(h * bbox_fractions[2])
    y2 = int(h * bbox_fractions[3])
    input_box = np.array([x1, y1, x2, y2])

    predictor = SamPredictor(sam)
    predictor.set_image(img)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
    )

    mask = masks[0]

    # Convert boolean mask to binary grayscale image
    result = img.copy()
    result[mask] = (255, 255, 255)
    result[~mask] = (0, 0, 0)
    gray = cv.cvtColor(result, cv.COLOR_RGB2GRAY)

    return gray


# ---------------------------------------------------------------------------
# OpenCV fallback segmentation (no SAM required)
# ---------------------------------------------------------------------------

def make_mask_opencv(img: np.ndarray) -> np.ndarray:
    """
    Create a binary mask using simple OpenCV thresholding (Otsu's method).

    This is a fallback for when SAM is not available. It works well for
    crystals photographed against a plain, contrasting background.

    Args:
        img: Input image (BGR, numpy array).

    Returns:
        Binary mask as a grayscale numpy array (255 = crystal, 0 = background).
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

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
    with the Y axis inverted so that Y increases upward (standard math convention).

    Args:
        mask:  Binary grayscale mask (255 = crystal, 0 = background).
        show:  If True, display the contour overlay using matplotlib.
        title: Plot title (used when show=True).

    Returns:
        Contour as an (N, 2) numpy matrix of (x, y) integer pairs,
        centred about the image midpoint.

    Raises:
        ValueError: If no contours are found in the mask.
    """
    contours, _ = cv.findContours(
        image=mask,
        mode=cv.RETR_TREE,
        method=cv.CHAIN_APPROX_NONE
    )

    if not contours:
        raise ValueError("No contours found in mask. Check segmentation output.")

    # Use the largest contour (the crystal outline)
    largest = max(contours, key=len)

    x = largest[:, 0, 0]
    y = largest[:, 0, 1]

    if show:
        plt.figure()
        plt.imshow(mask, cmap="gray")
        plt.plot(x, y, "r.", markersize=1)
        plt.title(title)
        plt.axis("off")

    h, w = mask.shape

    # Centre x about 0 (centre of rotation)
    x = np.array(x - w / 2, dtype=int)

    # Invert y so it increases upward
    y = h - y

    return np.matrix(np.column_stack((x, y)))
