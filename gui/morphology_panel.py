"""
Panel for binary morphological operations on thresholded medical images.
User can binarize the image, choose structuring element shape and size, and apply erosion, dilation, opening, closing, or boundary extraction.
"""

# PyQt6.QtWidgets — QWidget, QVBoxLayout, QHBoxLayout, QSlider, QSpinBox, QComboBox, QPushButton, QLabel, QGroupBox
# PyQt6.QtCore — pyqtSignal
# processing.morphology.structuring_element — get_square_se, get_cross_se
# processing.morphology.erosion_dilation — erode, dilate
# processing.morphology.opening_closing — opening, closing
# processing.morphology.boundary_extraction — extract_boundary
# utils.image_utils — binarize: threshold numpy array to 0/1

# FUNCTIONS / CLASSES
# class MorphologyPanel(QWidget):
#   def __init__: build UI — threshold slider, SE shape selector, SE size spinner, op buttons
#   def on_threshold_changed: binarize image at current threshold → display preview
#   def on_operation_clicked: read SE settings → apply selected op → emit result
# signal: morphology_applied(np.ndarray)

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale, binarize, normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call before binarize — morphology only works on grayscale
# binarize(image, threshold)          # call in on_threshold_changed using slider value
# normalize_to_uint8(result)          # call on morphology output before emitting signal
# @wrap_errors                        # decorate on_threshold_changed and on_operation_clicked
# Note: pass the binarized array (not the original) to all erosion/dilation/opening/closing calls