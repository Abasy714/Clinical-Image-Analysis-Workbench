"""
Statistical analysis of a rectangular ROI drawn on the image.
Computes mean, variance, and local histogram of the pixel values within the selected region.
"""

# numpy — array slicing, mean, variance computation
# processing.histogram.histogram_utils — compute_histogram

# FUNCTIONS
# def compute_roi_stats(image: np.ndarray, roi: tuple[int,int,int,int]) -> dict:
#   — extract ROI pixels -> compute mean, variance, histogram
#   — return dict with keys: 'mean', 'variance', 'histogram', 'pixel_count'
# def extract_roi(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
#   — safely slice image with boundary clamping

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call at top of compute_roi_stats
# @wrap_errors                        # decorate compute_roi_stats and extract_roi
# Note: extract_roi must clamp coordinates to image bounds — never let x+w exceed image width