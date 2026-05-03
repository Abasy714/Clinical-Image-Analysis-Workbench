"""
Histogram computation from scratch without using numpy.histogram or cv2.calcHist.
Supports 8-bit and 16-bit grayscale images.
"""

# numpy — array iteration and bin accumulation

# FUNCTIONS
# def compute_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
#   — iterate pixel values -> accumulate counts into bins array -> return counts
# def compute_cdf(histogram: np.ndarray) -> np.ndarray:
#   — compute cumulative distribution function from histogram -> normalize to [0,1]

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call at top of compute_histogram
# @wrap_errors                        # decorate compute_histogram and compute_cdf
# Note: compute_histogram returns raw counts — caller decides whether to normalize
# Note: compute_cdf is called by local_equalization.py and histogram_panel.py