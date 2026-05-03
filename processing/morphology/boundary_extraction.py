"""
Morphological boundary extraction for outlining anatomical structures in binary masks.
Subtracts the eroded image from the original binary image to isolate the boundary pixels.
"""

# numpy — array subtraction
# processing.morphology.erosion_dilation — erode

# FUNCTIONS
# def extract_boundary(binary_image: np.ndarray, se: np.ndarray) -> np.ndarray:
#   — eroded = erode(binary_image, se) -> return binary_image - eroded

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale
# from utils.error_handler import wrap_errors
#
# validate_grayscale(binary_image)    # call before erode
# @wrap_errors                        # decorate extract_boundary
# Note: result = binary_image - erode(binary_image, se) — output is binary 0/1
# Note: normalize_to_uint8 called by morphology_panel.py for display only