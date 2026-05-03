"""
Main application window for the Clinical Image Analysis Workbench.
Manages the tabbed interface, global pipeline state, and communication between panels.
"""

# PyQt6.QtWidgets — QMainWindow, QTabWidget, QVBoxLayout, QAction, QFileDialog, QMessageBox
# PyQt6.QtCore — Qt signals and slots for inter-panel communication
# utils.pipeline_state — PipelineState: manages the sequential op stack (undo/reset)
# utils.error_handler — wrap_errors: decorator for graceful crash handling
# processing.io.image_loader — load_image: loads DICOM/JPEG/BMP into numpy array + metadata
# processing.io.image_saver — save_image: exports the current processed image to disk

# CONSTANTS
# APP_TITLE = "Clinical Image Analysis Workbench"
# APP_VERSION = "1.0.0"

# FUNCTIONS / CLASSES
# class MainWindow(QMainWindow): — main window, owns all panels and pipeline state
#   def __init__: set up tabs, menu bar, status bar, connect signals
#   def load_image: open file dialog → call image_loader → push to viewer + metadata panel
#   def save_image: call image_saver on current processed image
#   def on_operation_applied: receive result from any panel → push to pipeline state → refresh viewer
#   def update_status_bar: show current image info and last operation name

# --- UTIL USAGE GUIDE ---
# from utils.pipeline_state import PipelineState
# from utils.error_handler import wrap_errors, show_error_dialog
# from utils.image_utils import to_qpixmap
#
# pipeline.set_original(image)        # call after image_loader returns — sets the base image
# pipeline.push(op_name, result)      # call every time any panel emits a processed image
# pipeline.current()                  # call to get the image currently shown in the viewer
# to_qpixmap(image)                   # call before passing any numpy array to ImageViewer
# @wrap_errors                        # decorate load_image and save_image methods
# show_error_dialog(title, msg)       # call when file dialog returns an unsupported format