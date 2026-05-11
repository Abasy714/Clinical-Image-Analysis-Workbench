"""
Template matching panel.
User crops a template from the image ROI, then finds matching regions in the
current source image using normalized cross-correlation.
"""

import logging
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame,
    QDoubleSpinBox, QSpinBox, QFileDialog,
)
from PyQt6.QtCore import pyqtSignal

try:
    from gui.styles import (
        HEADER_SS, FIELD_SS, APPLY_BTN_SS, ACCENT, BORDER, SPINBOX_SS,
    )
except ImportError:
    HEADER_SS = FIELD_SS = APPLY_BTN_SS = SPINBOX_SS = ""
    ACCENT = "#c8f135"
    BORDER = "#2c2e2a"

from utils import show_error_dialog

_log = logging.getLogger('ciaw')


class TemplatePanel(QWidget):
    """Template matching panel - crop from ROI, then locate in the image."""

    template_match_found = pyqtSignal(str, np.ndarray)
    template_changed = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._template: np.ndarray | None = None
        self._current_image: np.ndarray | None = None
        self._last_roi = None
        self._last_roi_image: np.ndarray | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        header = QLabel("TEMPLATE MATCHING")
        header.setStyleSheet(HEADER_SS)
        layout.addWidget(header)

        step1 = QLabel("STEP 1 - Crop template from image")
        step1.setStyleSheet(FIELD_SS)
        layout.addWidget(step1)

        self._crop_btn = QPushButton("Crop from current ROI")
        self._crop_btn.setStyleSheet(APPLY_BTN_SS)
        self._crop_btn.clicked.connect(self._on_crop_clicked)
        layout.addWidget(self._crop_btn)

        self._load_template_btn = QPushButton("Load Template Image")
        self._load_template_btn.setStyleSheet(APPLY_BTN_SS)
        self._load_template_btn.clicked.connect(self._on_load_template_clicked)
        layout.addWidget(self._load_template_btn)

        self._template_status = QLabel("No template selected")
        self._template_status.setStyleSheet("color: #6b6f65; font-size: 9px;")
        layout.addWidget(self._template_status)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {BORDER};")
        layout.addWidget(sep)

        step2 = QLabel("STEP 2 - Find template in image")
        step2.setStyleSheet(FIELD_SS)
        layout.addWidget(step2)

        threshold_row = QWidget()
        trl = QHBoxLayout(threshold_row)
        trl.setContentsMargins(0, 0, 0, 0)
        trl.setSpacing(6)
        threshold_lbl = QLabel("Threshold")
        threshold_lbl.setStyleSheet(FIELD_SS)
        trl.addWidget(threshold_lbl)
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.10, 1.00)
        self._threshold_spin.setDecimals(2)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setValue(0.80)
        self._threshold_spin.setStyleSheet(SPINBOX_SS)
        trl.addWidget(self._threshold_spin)
        layout.addWidget(threshold_row)

        max_row = QWidget()
        mrl = QHBoxLayout(max_row)
        mrl.setContentsMargins(0, 0, 0, 0)
        mrl.setSpacing(6)
        max_lbl = QLabel("Max boxes")
        max_lbl.setStyleSheet(FIELD_SS)
        mrl.addWidget(max_lbl)
        self._max_matches_spin = QSpinBox()
        self._max_matches_spin.setRange(1, 25)
        self._max_matches_spin.setValue(5)
        self._max_matches_spin.setStyleSheet(SPINBOX_SS)
        mrl.addWidget(self._max_matches_spin)
        layout.addWidget(max_row)

        self._find_btn = QPushButton("Find Template")
        self._find_btn.setStyleSheet(APPLY_BTN_SS)
        self._find_btn.clicked.connect(self._on_find_clicked)
        layout.addWidget(self._find_btn)

        self._result_label = QLabel("No match found yet")
        self._result_label.setStyleSheet("color: #6b6f65; font-size: 9px;")
        layout.addWidget(self._result_label)

        self._confidence_label = QLabel("Best NCC score: -")
        self._confidence_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 11px; font-weight: bold;"
        )
        layout.addWidget(self._confidence_label)

        layout.addStretch()

    def set_current_image(self, image: np.ndarray):
        self._current_image = image

    def set_template_from_roi(self, image: np.ndarray, roi):
        self._last_roi = roi
        self._last_roi_image = image
        if roi is None or roi.width() == 0 or roi.height() == 0:
            return
        try:
            self._crop_template_from_roi(image, roi)
        except Exception as e:
            _log.error("template ROI crop failed: %s", e)

    def _crop_template_from_roi(self, image: np.ndarray, roi):
        if image is None:
            raise ValueError("Load an image before selecting a template.")
        x, y, w, h = roi.x(), roi.y(), roi.width(), roi.height()
        img_h, img_w = image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise ValueError("Template ROI must be at least 2x2 pixels.")
        self._template = image[y1:y2, x1:x2].copy()
        self._template_status.setText(f"Template: {x2 - x1}x{y2 - y1} px")
        self.template_changed.emit(self._template)

    def _on_crop_clicked(self):
        if self._last_roi is None or self._last_roi_image is None:
            show_error_dialog("No ROI", "Draw an ROI on the image first, then crop the template.")
            return
        try:
            self._crop_template_from_roi(self._last_roi_image, self._last_roi)
            _log.info("Template cropped from current ROI.")
        except Exception as e:
            _log.error("template crop error: %s", e)
            show_error_dialog("Template Crop Error", str(e))

    def _on_load_template_clicked(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Template Image", "",
            "Images (*.dcm *.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )
        if not filepath:
            return
        try:
            from processing.io import load_image
            loaded = load_image(filepath)
            if loaded is None:
                raise ValueError(f"Could not load template: {filepath}")
            image, _metadata = loaded
            if image is None:
                raise ValueError(f"Could not load template: {filepath}")
            self._template = image.copy()
            h, w = self._template.shape[:2]
            self._template_status.setText(f"Template file: {w}x{h} px")
            self.template_changed.emit(self._template)
        except Exception as e:
            _log.error("template load error: %s", e)
            show_error_dialog("Template Load Error", str(e))

    def _on_find_clicked(self):
        if self._template is None:
            show_error_dialog("No Template", "Draw an ROI first to define the template.")
            return
        if self._current_image is None:
            show_error_dialog("No Image", "Load an image first.")
            return

        try:
            from processing.frequency import match_template

            result, matches, _score_map = match_template(
                self._current_image,
                self._template,
                threshold=self._threshold_spin.value(),
                max_matches=self._max_matches_spin.value(),
            )
            best = matches[0]
            self._result_label.setText(
                f"{len(matches)} match(es), best row={best['row']}, col={best['col']}"
            )
            self._confidence_label.setText(f"Best NCC score: {best['score']:.3f}")
            self.template_match_found.emit(
                f"Template Match r={best['row']} c={best['col']}", result
            )
        except Exception as e:
            _log.error("template matching error: %s", e)
            show_error_dialog("Template Matching Error", str(e))
