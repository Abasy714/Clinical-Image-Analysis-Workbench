"""
Handles exporting the current processed image to disk in JPEG or BMP format.
Converts numpy arrays back to image files using Pillow.
"""

# numpy — image array input
# Pillow (PIL.Image) — converting array to image file and saving
# os, pathlib — path construction and directory validation
# utils.error_handler — wrap_errors

# FUNCTIONS
# def save_image(image: np.ndarray, filepath: str) -> bool:
#   — normalize array to uint8 -> convert to PIL Image -> save to filepath
#   — return True on success, False on failure
# def _normalize_for_export(image: np.ndarray) -> np.ndarray:
#   — clip and scale float arrays to 0-255 uint8 range

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# normalize_to_uint8(image)           # call at the top of save_image before Pillow conversion
# @wrap_errors                        # decorate save_image — path may be invalid or disk full