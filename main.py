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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import sys
from PyQt6.QtWidgets import QApplication
from utils import setup_logger
from gui.styles import apply_styles
from gui.main_window import MainWindow


class _LogHandler(logging.Handler):
    _LEVEL_KEYS = {
        logging.DEBUG:    'MUTED',
        logging.INFO:     'TEXT',
        logging.WARNING:  'AMBER',
        logging.ERROR:    'RED',
        logging.CRITICAL: 'RED',
    }

    def __init__(self, widget):
        super().__init__()
        self._w = widget

    def emit(self, record):
        msg = self.format(record)
        key = self._LEVEL_KEYS.get(record.levelno, 'TEXT')
        from gui.theme import get as _get_theme
        color = _get_theme().get(key, '#cccccc')
        from PyQt6.QtCore import QMetaObject, Qt, Q_ARG
        QMetaObject.invokeMethod(
            self._w, '_append_colored',
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, msg),
            Q_ARG(str, color))


def main():
    logger = setup_logger()
    logger.info("Starting %s", "Clinical Image Analysis Workbench")

    app = QApplication(sys.argv)
    app.setApplicationName("Clinical Image Analysis Workbench")
    app.setOrganizationName("CIAW")

    apply_styles(app)

    window = MainWindow()
    window.show()

    _h = _LogHandler(window.log_widget)
    _h.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'))
    _h.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(_h)

    logger.info("Application ready")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
