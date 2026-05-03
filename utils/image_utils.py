"""
Shared image utility functions used across all processing and GUI modules.
Handles dtype normalization, channel validation, numpy-to-Qt conversion, and binarization.
"""

# numpy — array dtype conversion, clipping, normalization
# PyQt6.QtGui — QImage, QPixmap: for converting numpy arrays to Qt display objects
# PyQt6.QtCore — Qt

# FUNCTIONS
# def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
#   — scale float or high-bit-depth arrays to 0-255 uint8
# def to_grayscale(image: np.ndarray) -> np.ndarray:
#   — convert RGB to grayscale using luminance weights if needed
# def validate_grayscale(image: np.ndarray) -> bool:
#   — return True if image is 2D (single channel), raise ValueError otherwise
# def to_qpixmap(image: np.ndarray) -> QPixmap:
#   — normalize -> create QImage from bytes -> wrap in QPixmap for display
# def binarize(image: np.ndarray, threshold: int) -> np.ndarray:
#   — return binary 0/1 array where image > threshold
