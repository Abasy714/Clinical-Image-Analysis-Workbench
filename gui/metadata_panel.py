"""
Panel for displaying image metadata extracted from DICOM, JPEG, and BMP files.
Shows width, height, bit depth, and DICOM-specific tags when available.
"""

# PyQt6.QtWidgets — QWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QLabel
# PyQt6.QtCore — Qt

# FUNCTIONS / CLASSES
# class MetadataPanel(QWidget):
#   def update_metadata: accept metadata dict → populate table rows
#   def clear: reset table to empty state
# DICOM tags to display: Modality, PatientName, PatientAge, BodyPartExamined

# --- UTIL USAGE GUIDE ---
# from utils.error_handler import wrap_errors
#
# @wrap_errors                        # decorate update_metadata — metadata dict may be malformed
# No image_utils needed here — this panel only displays text, no array manipulation