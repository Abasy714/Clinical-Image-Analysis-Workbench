"""
Frequency domain panel for periodic noise removal via interactive notch filtering.
Displays the log-scaled FFT magnitude spectrum and allows the user to click on noise spikes.
"""

# PyQt6.QtWidgets — QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QSpinBox, QLabel
# PyQt6.QtCore — pyqtSignal, QPoint
# matplotlib.backends.backend_qtagg — FigureCanvasQTAgg
# matplotlib.figure — Figure
# numpy — spectrum data handling
# processing.frequency.spectrum — compute_spectrum, spectrum_to_display
# processing.frequency.notch_filter — create_notch_filter, apply_notch_filter

# FUNCTIONS / CLASSES
# class FourierPanel(QWidget):
#   def __init__: build UI — spectrum canvas, filter shape selector, radius spinner, apply button
#   def set_image: compute and display spectrum for loaded image
#   def on_spectrum_clicked: capture (u,v) click → generate notch + conjugate → preview mask on spectrum
#   def on_apply_clicked: multiply mask with FFT → IFFT → emit cleaned image
#   def _display_spectrum: render log-magnitude to matplotlib canvas
# signal: notch_applied(np.ndarray)

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale, normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call in set_image before computing spectrum
# normalize_to_uint8(cleaned)         # call on ifft result before emitting notch_applied
# @wrap_errors                        # decorate on_spectrum_clicked and on_apply_clicked
# Note: spectrum_to_display() from processing.frequency.spectrum handles its own normalization
#       only call normalize_to_uint8 on the final reconstructed spatial-domain image