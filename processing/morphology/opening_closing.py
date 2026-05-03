"""
Compound morphological operations built on top of the erosion and dilation functions.
Opening removes small noise; closing fills gaps in binary medical image masks.
"""

# processing.morphology.erosion_dilation — erode, dilate

# FUNCTIONS
# def opening(binary_image: np.ndarray, se: np.ndarray) -> np.ndarray:
#   — erode -> dilate (removes small foreground noise)
# def closing(binary_image: np.ndarray, se: np.ndarray) -> np.ndarray:
#   — dilate -> erode (fills small background holes)

# --- UTIL USAGE GUIDE ---
# from utils.error_handler import wrap_errors
#
# @wrap_errors                        # decorate opening and closing
# Note: no direct image_utils calls needed — erode and dilate handle their own validation
# Note: output is binary 0/1 — morphology_panel.py calls normalize_to_uint8 before display