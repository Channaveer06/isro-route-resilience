import numpy as np
from skimage.morphology import skeletonize as ski_skeletonize

def extract_skeleton(binary_mask):
    """
    Converts a binary road mask into a single-pixel wide skeleton.
    Args:
        binary_mask (np.ndarray): 2D array where roads are 1 and background is 0.
    Returns:
        np.ndarray: Skeletonized binary mask.
    """
    # Ensure binary format (boolean required by skimage)
    bool_mask = binary_mask > 0.5
    skeleton = ski_skeletonize(bool_mask)
    return skeleton.astype(np.uint8)
