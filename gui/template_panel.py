# STATUS: STUB — Phase 2
"""
Template matching panel using Fourier-domain cross-correlation.
User crops a template from the image, then finds its location in the full image.
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFrame)
from PyQt6.QtCore import pyqtSignal

try:
    from gui.styles import (HEADER_SS, FIELD_SS, APPLY_BTN_SS, ACCENT, BORDER)
except ImportError:
    HEADER_SS = FIELD_SS = APPLY_BTN_SS = ""
    ACCENT = "#c8f135"
    BORDER = "#2c2e2a"


class TemplatePanel(QWidget):
    """Template matching panel — crop from ROI, then locate via cross-correlation."""

    template_match_found = pyqtSignal(str, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._template = None
        self._current_image = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        header = QLabel("TEMPLATE MATCHING")
        header.setStyleSheet(HEADER_SS)
        layout.addWidget(header)

        step1 = QLabel("STEP 1 — Crop template from image")
        step1.setStyleSheet(FIELD_SS)
        layout.addWidget(step1)

        self._crop_btn = QPushButton("Crop from current ROI")
        self._crop_btn.setStyleSheet(APPLY_BTN_SS)
        self._crop_btn.clicked.connect(self._on_crop_clicked)
        layout.addWidget(self._crop_btn)

        self._template_status = QLabel("No template selected")
        self._template_status.setStyleSheet("color: #6b6f65; font-size: 9px;")
        layout.addWidget(self._template_status)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {BORDER};")
        layout.addWidget(sep)

        step2 = QLabel("STEP 2 — Find template in image")
        step2.setStyleSheet(FIELD_SS)
        layout.addWidget(step2)

        self._find_btn = QPushButton("Find Template")
        self._find_btn.setStyleSheet(APPLY_BTN_SS)
        self._find_btn.clicked.connect(self._on_find_clicked)
        layout.addWidget(self._find_btn)

        self._result_label = QLabel("No match found yet")
        self._result_label.setStyleSheet("color: #6b6f65; font-size: 9px;")
        layout.addWidget(self._result_label)

        self._confidence_label = QLabel("Peak confidence: —")
        self._confidence_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 11px; font-weight: bold;"
        )
        layout.addWidget(self._confidence_label)

        layout.addStretch()

    def set_current_image(self, image: np.ndarray):
        self._current_image = image

    def set_template_from_roi(self, image: np.ndarray, roi):
        if roi is None or roi.width() == 0 or roi.height() == 0:
            return
        x, y, w, h = roi.x(), roi.y(), roi.width(), roi.height()
        img_h, img_w = image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        self._template = image[y1:y2, x1:x2].copy()
        self._template_status.setText(f"Template: {x2 - x1}x{y2 - y1} px")

    def _on_crop_clicked(self):
        import logging
        logging.getLogger('ciaw').info(
            "Draw an ROI on the image first, then click Find Template"
        )

    def _on_find_clicked(self):
        if self._template is None:
            try:
                from utils import show_error_dialog
                show_error_dialog("No Template", "Draw an ROI first to define the template.")
            except Exception:
                pass
            return
        if self._current_image is None:
            try:
                from utils import show_error_dialog
                show_error_dialog("No Image", "Load an image first.")
            except Exception:
                pass
            return
        try:
            from processing.frequency import fourier_cross_correlate, find_best_match  # noqa: F401
            raise NotImplementedError("template matching — Phase 2")
        except NotImplementedError:
            import logging
            logging.getLogger('ciaw').error(
                "fourier_cross_correlate not implemented yet"
            )
