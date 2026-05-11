"""
AI/CV panel.

The classifier controls are present but the trained model is not bundled with
the project. The enhancement suggestion controls are implemented locally using
the existing processing functions.
"""

import logging
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFrame, QProgressBar,
)
from PyQt6.QtCore import pyqtSignal

try:
    from gui.styles import HEADER_SS, APPLY_BTN_SS, ACCENT, BORDER
except ImportError:
    HEADER_SS = APPLY_BTN_SS = ""
    ACCENT = "#c8f135"
    BORDER = "#2c2e2a"

from utils import show_error_dialog, normalize_to_uint8

_log = logging.getLogger('ciaw')


class AIPanel(QWidget):
    """AI classification + enhancement suggestion panel."""

    classification_done = pyqtSignal(str, np.ndarray)
    suggestion_applied = pyqtSignal(str, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_image = None
        self._current_roi = None
        self._suggested_action = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        ai_header = QLabel("ROI CLASSIFIER")
        ai_header.setStyleSheet(HEADER_SS)
        layout.addWidget(ai_header)

        self._classify_btn = QPushButton("Classify ROI")
        self._classify_btn.setStyleSheet(APPLY_BTN_SS)
        self._classify_btn.clicked.connect(self._on_classify_clicked)
        layout.addWidget(self._classify_btn)

        self._result_label = QLabel("Draw an ROI then click Classify")
        self._result_label.setStyleSheet("color: #6b6f65; font-size: 9px;")
        layout.addWidget(self._result_label)

        self._confidence_label = QLabel("Confidence: -")
        self._confidence_label.setStyleSheet(
            f"color: {ACCENT}; font-size: 13px; font-weight: bold;"
        )
        layout.addWidget(self._confidence_label)

        self._confidence_bar = QProgressBar()
        self._confidence_bar.setRange(0, 100)
        self._confidence_bar.setValue(0)
        self._confidence_bar.setStyleSheet(f"""
            QProgressBar {{
                background: #252623; border: 1px solid #353730;
                border-radius: 2px; height: 8px;
            }}
            QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}
        """)
        layout.addWidget(self._confidence_bar)

        self._gradcam_btn = QPushButton("Show Grad-CAM overlay")
        self._gradcam_btn.setStyleSheet(APPLY_BTN_SS)
        self._gradcam_btn.setEnabled(False)
        self._gradcam_btn.clicked.connect(self._on_gradcam_clicked)
        layout.addWidget(self._gradcam_btn)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {BORDER};")
        layout.addWidget(sep)

        sug_header = QLabel("ENHANCEMENT SUGGESTION")
        sug_header.setStyleSheet(HEADER_SS)
        layout.addWidget(sug_header)

        self._analyze_btn = QPushButton("Analyze & Suggest")
        self._analyze_btn.setStyleSheet(APPLY_BTN_SS)
        self._analyze_btn.clicked.connect(self._on_analyze_clicked)
        layout.addWidget(self._analyze_btn)

        self._suggestion_label = QLabel("-")
        self._suggestion_label.setWordWrap(True)
        self._suggestion_label.setStyleSheet("color: #6b6f65; font-size: 9px;")
        layout.addWidget(self._suggestion_label)

        self._apply_suggestion_btn = QPushButton("Apply Suggested Pipeline")
        self._apply_suggestion_btn.setStyleSheet(APPLY_BTN_SS)
        self._apply_suggestion_btn.setEnabled(False)
        self._apply_suggestion_btn.clicked.connect(self._on_apply_suggestion_clicked)
        layout.addWidget(self._apply_suggestion_btn)

        layout.addStretch()

    def set_current_image(self, image: np.ndarray):
        self._current_image = image

    def set_current_roi(self, roi):
        self._current_roi = roi

    def _on_classify_clicked(self):
        _log.error("AI classifier requested, but no trained model is bundled.")
        show_error_dialog(
            "Classifier Unavailable",
            "The classifier model is not included in this project yet."
        )

    def _on_gradcam_clicked(self):
        _log.error("Grad-CAM requested, but no trained classifier is bundled.")
        show_error_dialog(
            "Grad-CAM Unavailable",
            "Grad-CAM requires a trained classifier model, which is not included yet."
        )

    def _on_analyze_clicked(self):
        if self._current_image is None:
            show_error_dialog("No Image", "Load an image before analyzing enhancement suggestions.")
            return
        try:
            img = self._current_image.astype(float)
            mean_val = float(img.mean())
            std_val = float(img.std())

            if std_val < 30:
                suggestion = "Low contrast detected -> Apply Local EQ 8x8"
                self._suggested_action = "local_eq"
            elif std_val > 80:
                suggestion = "High variation detected -> Apply Median 3x3 first"
                self._suggested_action = "median"
            elif mean_val < 60:
                suggestion = "Dark image -> Apply Local EQ 8x8"
                self._suggested_action = "local_eq"
            else:
                suggestion = "Image quality OK -> try Gaussian sigma=1.0 for smoothing"
                self._suggested_action = "gaussian"

            self._suggestion_label.setText(suggestion)
            self._apply_suggestion_btn.setEnabled(True)
        except Exception as e:
            _log.error("analyze failed: %s", e)
            show_error_dialog("Analysis Error", str(e))

    def _on_apply_suggestion_clicked(self):
        if self._current_image is None:
            show_error_dialog("No Image", "Load an image before applying a suggestion.")
            return
        if self._suggested_action is None:
            show_error_dialog("No Suggestion", "Analyze the image before applying a suggestion.")
            return
        try:
            if self._suggested_action == "local_eq":
                from processing.histogram import local_histogram_equalization
                result = local_histogram_equalization(self._current_image, 8)
                op_name = "Suggested Local EQ 8x8"
            elif self._suggested_action == "median":
                from processing.spatial import median_filter
                result = median_filter(self._current_image, 3)
                op_name = "Suggested Median 3x3"
            else:
                from processing.spatial import gaussian_filter
                result = gaussian_filter(self._current_image, 3, 1.0)
                op_name = "Suggested Gaussian 3x3"

            self.suggestion_applied.emit(op_name, normalize_to_uint8(result))
        except Exception as e:
            _log.error("apply suggestion failed: %s", e)
            show_error_dialog("Suggestion Error", str(e))
