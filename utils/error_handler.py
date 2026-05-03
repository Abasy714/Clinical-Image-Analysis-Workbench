# STATUS: IMPLEMENTED
"""
Centralized error handling for the workbench application.
Provides a decorator and standalone function for catching and displaying errors gracefully without crashing.
"""

# functools — wraps: preserve decorated function metadata
# PyQt6.QtWidgets — QMessageBox: display error dialogs to the user
# traceback — format_exc: capture full stack trace for logging
# logging — write errors to log file for debugging

# FUNCTIONS
# def wrap_errors(func): decorator — wraps any function, catches all exceptions,
#   shows QMessageBox with friendly message, logs full traceback, returns None on error
# def show_error_dialog(title: str, message: str): standalone dialog helper
# def setup_logger() -> logging.Logger: configure file + console logging for the app

import functools
import logging
import traceback
from PyQt6.QtWidgets import QMessageBox

_DIALOG_STYLE = """
    QMessageBox {
        background-color: #111210;
        color: #eceee8;
        font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
        font-size: 11px;
    }
    QMessageBox QLabel { color: #eceee8; }
    QPushButton {
        background: #252623; color: #eceee8;
        border: 1px solid #353730; border-radius: 2px; padding: 4px 10px;
    }
    QPushButton:hover { background: #353730; }
"""


def setup_logger() -> logging.Logger:
    logger = logging.getLogger('ciaw')
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))

    try:
        file_h = logging.FileHandler('ciaw.log', encoding='utf-8')
        file_h.setLevel(logging.DEBUG)
        file_h.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d — %(message)s"
        ))
        logger.addHandler(file_h)
    except OSError:
        pass

    logger.addHandler(console)
    return logger


def wrap_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger = logging.getLogger('ciaw')
            logger.error("Exception in %s:\n%s", func.__name__, traceback.format_exc())
            dlg = QMessageBox()
            dlg.setIcon(QMessageBox.Icon.Critical)
            dlg.setWindowTitle(f"Error — {func.__name__}")
            dlg.setText(str(exc) + "\n\nSee ciaw.log for details.")
            dlg.setStyleSheet(_DIALOG_STYLE)
            dlg.exec()
            return None
    return wrapper


def show_error_dialog(title: str, message: str):
    dlg = QMessageBox()
    dlg.setIcon(QMessageBox.Icon.Critical)
    dlg.setWindowTitle(title)
    dlg.setText(message)
    dlg.setStyleSheet(_DIALOG_STYLE)
    dlg.exec()
