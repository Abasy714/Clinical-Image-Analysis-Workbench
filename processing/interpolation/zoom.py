"""
Zoom dispatcher that routes zoom requests to either nearest-neighbor or bilinear interpolation.
Maintains the zoom factor and selected interpolation mode.
"""

# numpy — array passthrough
# processing.interpolation.nearest_neighbor — nearest_neighbor_resize
# processing.interpolation.bilinear — bilinear_resize

# CONSTANTS
# INTERPOLATION_MODES = ['nearest', 'bilinear']

# FUNCTIONS
# def apply_zoom(image: np.ndarray, zoom_factor: float, mode: str) -> np.ndarray:
#   — validate mode -> compute new dimensions -> dispatch to correct interpolation function

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# normalize_to_uint8(result)          # call on the resized array before returning to caller
# @wrap_errors                        # decorate apply_zoom — bad zoom_factor will crash