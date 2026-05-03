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
