# STATUS: STUB — Phase 2 bonus
"""
AI/CV panel for ROI classification using MobileNetV2.
Displays classification result, confidence, and Grad-CAM overlay.
Also provides an enhancement suggestion engine based on image statistics.
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                              QFrame, QProgressBar)
from PyQt6.QtCore import pyqtSignal

try:
    from gui.styles import (HEADER_SS, FIELD_SS, APPLY_BTN_SS, ACCENT, BORDER)
except ImportError:
    HEADER_SS = FIELD_SS = APPLY_BTN_SS = ""
    ACCENT = "#c8f135"
    BORDER = "#2c2e2a"


class AIPanel(QWidget):
    """AI classification + enhancement suggestion panel."""

    classification_done = pyqtSignal(str, np.ndarray)
    suggestion_applied = pyqtSignal(str, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_image = None
        self._current_roi = None
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

        self._confidence_label = QLabel("Confidence: —")
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

        self._suggestion_label = QLabel("—")
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
        import logging
        logging.getLogger('ciaw').error("AI classifier not implemented yet — Phase 2 bonus")

    def _on_gradcam_clicked(self):
        import logging
        logging.getLogger('ciaw').error("Grad-CAM not implemented yet — Phase 2 bonus")

    def _on_analyze_clicked(self):
        if self._current_image is None:
            return
        try:
            img = self._current_image.astype(float)
            mean_val = img.mean()
            std_val = img.std()
            suggestions = []
            if std_val < 30:
                suggestions.append("Low contrast detected → Apply Local EQ 8x8")
            if std_val > 80:
                suggestions.append("High noise detected → Apply Adaptive Median first")
            if mean_val < 60:
                suggestions.append("Dark image → Apply Histogram EQ")
            if not suggestions:
                suggestions.append("Image quality OK → try Gaussian sigma=1.0 for smoothing")
            self._suggestion_label.setText("\n".join(suggestions))
            self._apply_suggestion_btn.setEnabled(True)
        except Exception as e:
            import logging
            logging.getLogger('ciaw').error(f"analyze failed: {e}")

    def _on_apply_suggestion_clicked(self):
        import logging
        logging.getLogger('ciaw').error("apply suggestion not implemented yet")
