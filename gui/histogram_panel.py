"""
Panel for local histogram equalization and ROI histogram display.
User inputs block size; algorithm equalizes local regions to enhance contrast in medical images.
"""

# PyQt6.QtWidgets — QWidget, QVBoxLayout, QHBoxLayout, QSpinBox, QPushButton, QLabel
# PyQt6.QtCore — pyqtSignal
# matplotlib.backends.backend_qtagg — FigureCanvasQTAgg: embed matplotlib plot in PyQt6
# matplotlib.figure — Figure
# numpy — for histogram data preparation
# processing.histogram.local_equalization — local_histogram_equalization
# processing.histogram.histogram_utils — compute_histogram

# FUNCTIONS / CLASSES
# class HistogramPanel(QWidget):
#   def __init__: build UI — block size input, apply button, matplotlib canvas for histogram
#   def on_apply_clicked: call local_histogram_equalization → emit result
#   def display_histogram: compute_histogram on current ROI or full image → plot on canvas
#   def update_roi: receive ROI from ImageViewer → recompute and redisplay histogram
# signal: equalization_applied(np.ndarray)

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale, normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call before passing image to local_histogram_equalization
# normalize_to_uint8(result)          # call on equalized output before emitting signal
# @wrap_errors                        # decorate on_apply_clicked and display_histogram