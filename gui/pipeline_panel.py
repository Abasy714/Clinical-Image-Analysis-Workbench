"""
Pipeline management panel for sequential image enhancement operations.
Provides undo last step and reset to original buttons, and displays the current op stack.
"""

# PyQt6.QtWidgets — QWidget, QVBoxLayout, QListWidget, QPushButton, QLabel
# PyQt6.QtCore — pyqtSignal
# utils.pipeline_state — PipelineState

# FUNCTIONS / CLASSES
# class PipelinePanel(QWidget):
#   def __init__: build UI — op history list, undo button, reset button
#   def refresh_stack: read PipelineState and repopulate the list widget
#   def on_undo_clicked: call pipeline_state.undo → emit current image
#   def on_reset_clicked: call pipeline_state.reset → emit original image
# signal: pipeline_changed(np.ndarray) — emits the image after undo or reset

# --- UTIL USAGE GUIDE ---
# from utils.pipeline_state import PipelineState
# from utils.image_utils import to_qpixmap
# from utils.error_handler import wrap_errors
#
# pipeline.undo()                     # call inside on_undo_clicked — returns previous image
# pipeline.reset()                    # call inside on_reset_clicked — returns original image
# pipeline.get_stack_names()          # call inside refresh_stack to populate the list widget
# to_qpixmap(image)                   # call on the returned image before emitting pipeline_changed
# @wrap_errors                        # decorate on_undo_clicked and on_reset_clicked