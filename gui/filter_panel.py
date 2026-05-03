"""
Control panel for spatial filtering operations.
Allows the user to select kernel size and apply average, Gaussian, Sobel/Prewitt, or median filters.
"""

# PyQt6.QtWidgets — QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox, QLabel, QRadioButton
# PyQt6.QtCore — pyqtSignal
# processing.spatial.smoothing — average_filter, gaussian_filter
# processing.spatial.edge_detection — sobel, prewitt, combined_magnitude
# processing.spatial.median_filter — median_filter
# utils.image_utils — validate_grayscale: ensure image is single-channel before filtering

# FUNCTIONS / CLASSES
# class FilterPanel(QWidget):
#   def __init__: build UI — filter type selector, kernel size spinner, sigma input (Gaussian only)
#   def on_apply_clicked: read UI state → dispatch to correct processing function → emit result
#   def _toggle_sigma_input: show/hide sigma spinner based on selected filter type
# signal: filter_applied(np.ndarray) — emitted with the filtered image result

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale, normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call at the top of on_apply_clicked before dispatching
# normalize_to_uint8(result)          # call on the filter output before emitting the signal
# @wrap_errors                        # decorate on_apply_clicked — kernel ops can raise on bad input