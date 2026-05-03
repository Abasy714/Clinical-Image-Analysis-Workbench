"""
Core binary morphological operations: erosion and dilation, implemented from scratch.
Uses explicit neighborhood traversal — no scipy.ndimage or cv2 morphological functions.
"""

# numpy — array ops, padding, neighborhood extraction

# FUNCTIONS
# def erode(binary_image: np.ndarray, se: np.ndarray) -> np.ndarray:
#   — pad image -> for each pixel: check if SE fits entirely within foreground -> set output
# def dilate(binary_image: np.ndarray, se: np.ndarray) -> np.ndarray:
#   — pad image -> for each pixel: check if SE hits any foreground pixel -> set output

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale
# from utils.error_handler import wrap_errors
#
# validate_grayscale(binary_image)    # call at top of erode and dilate
# @wrap_errors                        # decorate erode and dilate
# Note: input must already be binarized (0/1) — call binarize() in the GUI panel before calling these
# Note: output is binary 0/1 — caller calls normalize_to_uint8 only for display purposes