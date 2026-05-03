"""
Panel for synthetic noise injection and ROI-based statistical analysis.
Allows the user to inject Gaussian or uniform noise and view local statistics of a drawn ROI.
"""

# PyQt6.QtWidgets — QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QDoubleSpinBox, QPushButton, QLabel, QGroupBox
# PyQt6.QtCore — pyqtSignal
# matplotlib.backends.backend_qtagg — FigureCanvasQTAgg
# matplotlib.figure — Figure
# processing.noise.noise_injection — add_gaussian_noise, add_uniform_noise
# processing.noise.roi_stats — compute_roi_stats
# utils.image_utils — validate_grayscale

# FUNCTIONS / CLASSES
# class NoisePanel(QWidget):
#   def __init__: build UI — noise type selector, parameter inputs, inject button, stats display area
#   def on_inject_clicked: call noise injection function → emit noisy image
#   def update_roi_stats: receive ROI → call compute_roi_stats → display mean, variance, histogram
# signal: noise_applied(np.ndarray)

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale, normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call before noise injection and before ROI stats
# normalize_to_uint8(noisy)           # call on noise output before emitting noise_applied
# @wrap_errors                        # decorate on_inject_clicked and update_roi_stats