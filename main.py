# STATUS: IMPLEMENTED
"""
Entry point for the Clinical Image Analysis Workbench.
Initializes the PyQt6 application and launches the main window.
"""

# PyQt6 — for creating and running the desktop GUI application
# sys — for passing command-line arguments to QApplication and clean exit

# --- UTIL USAGE GUIDE ---
# from utils.pipeline_state import PipelineState
# from utils.error_handler import setup_logger, wrap_errors
# from utils.image_utils import normalize_to_uint8
#
# pipeline = PipelineState()          # instantiate once at app startup, pass to all panels
# logger = setup_logger()             # call once at startup before anything else runs
# pipeline.set_original(image)        # call immediately after image_loader returns an array
# pipeline.current()                  # call whenever you need the currently displayed image

import sys
from PyQt6.QtWidgets import QApplication
from utils import setup_logger
from gui.styles import apply_styles
from gui import MainWindow


def main():
    logger = setup_logger()
    logger.info("Starting %s", "Clinical Image Analysis Workbench")

    app = QApplication(sys.argv)
    app.setApplicationName("Clinical Image Analysis Workbench")
    app.setOrganizationName("CIAW")

    apply_styles(app)

    window = MainWindow()
    window.show()

    logger.info("Application ready")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
