"""
Image display widget with zoom controls and interactive ROI drawing.
Powered entirely by custom interpolation — no built-in zoom libraries used.
"""

# PyQt6.QtWidgets — QWidget, QLabel, QSlider, QPushButton, QVBoxLayout, QHBoxLayout, QRubberBand
# PyQt6.QtGui — QPixmap, QImage, QPainter, QPen
# PyQt6.QtCore — Qt, QRect, QPoint, pyqtSignal
# numpy — for image array manipulation before conversion to QPixmap
# processing.interpolation.zoom — apply_zoom: dispatches to nearest-neighbor or bilinear
# utils.image_utils — to_qpixmap: converts numpy array to QPixmap for display

# CONSTANTS
# MIN_ZOOM = 0.25
# MAX_ZOOM = 4.0
# DEFAULT_ZOOM = 1.0

# FUNCTIONS / CLASSES
# class ImageViewer(QWidget): — main image display widget
#   def set_image: accept numpy array, store original, refresh display
#   def zoom_in / zoom_out: increment zoom level, re-render via interpolation
#   def set_interpolation_mode: toggle between nearest-neighbor and bilinear
#   def mousePressEvent / mouseMoveEvent / mouseReleaseEvent: rubber-band ROI selection
#   def get_roi: return (x, y, w, h) of currently selected ROI in image coordinates
#   def _render: apply zoom → convert to QPixmap → update display label
# signal: roi_selected(QRect) — emitted when user finishes drawing ROI

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import to_qpixmap, normalize_to_uint8
#
# normalize_to_uint8(image)           # call inside _render before converting to QPixmap
# to_qpixmap(image)                   # call inside _render to convert numpy array for display
# normalize_to_uint8(zoomed)          # call after apply_zoom returns before displaying result