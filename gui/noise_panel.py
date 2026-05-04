# STATUS: IMPLEMENTED
"""
Panel for synthetic noise injection and ROI-based statistical analysis.
Allows the user to inject Gaussian or uniform noise and view local statistics of a drawn ROI.
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QComboBox, QDoubleSpinBox, QLabel, QFrame,
                              QGridLayout, QSizePolicy)
from PyQt6.QtCore import pyqtSignal, Qt, QRect
from PyQt6.QtGui import QPainter, QColor, QPen

from gui.styles import (BG, PANEL, PANEL2, INPUT, BORDER, BORDER2,
                        ACCENT, TEXT, MUTED, MUTED2, btn_style,
                        HEADER_SS, FIELD_SS, APPLY_BTN_SS, SPINBOX_SS, COMBO_SS)
from utils import (validate_grayscale, normalize_to_uint8, wrap_errors, show_error_dialog,)


class _HistCanvas(QWidget):
    """Minimal histogram canvas reused within NoisePanel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._histogram: np.ndarray | None = None
        self.setFixedHeight(72)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(f"background:{BG};border:1px solid {BORDER};border-radius:2px;")

    def set_histogram(self, histogram: np.ndarray):
        self._histogram = histogram
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(BG))
        if self._histogram is None or len(self._histogram) == 0:
            painter.setPen(QColor(MUTED2))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No ROI selected")
            painter.end()
            return
        w, h = self.width(), self.height()
        pad = 4
        canvas_w, canvas_h = w - pad * 2, h - pad * 2
        hist = self._histogram.astype(np.float64)
        hist_max = float(hist.max()) or 1.0
        n = len(hist)
        accent = QColor(ACCENT)
        pen = QPen(QColor(BORDER))
        painter.setPen(pen)
        painter.drawLine(pad, h - pad, w - pad, h - pad)
        for i, val in enumerate(hist):
            bar_h = int(val / hist_max * canvas_h)
            x = pad + int(i / n * canvas_w)
            bw = max(1, int(canvas_w / n))
            accent.setAlpha(max(80, int(255 * val / hist_max)))
            painter.fillRect(x, h - pad - bar_h, bw, bar_h, accent)
        painter.end()


class NoisePanel(QWidget):
    noise_applied = pyqtSignal(str, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_image: np.ndarray | None = None

        self.setStyleSheet(f"background:{PANEL};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        hdr = QLabel("NOISE INJECTION")
        hdr.setStyleSheet(HEADER_SS)
        layout.addWidget(hdr)

        # noise type
        type_lbl = QLabel("Noise type")
        type_lbl.setStyleSheet(FIELD_SS)
        layout.addWidget(type_lbl)
        self._type_combo = QComboBox()
        self._type_combo.addItems(["Gaussian", "Uniform"])
        self._type_combo.setStyleSheet(COMBO_SS)
        layout.addWidget(self._type_combo)

        # Gaussian params
        self._gauss_w = QWidget()
        gl = QGridLayout(self._gauss_w)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(4)
        mean_lbl = QLabel("Mean")
        mean_lbl.setStyleSheet(FIELD_SS)
        gl.addWidget(mean_lbl, 0, 0)
        self._gauss_mean = QDoubleSpinBox()
        self._gauss_mean.setRange(-128.0, 128.0)
        self._gauss_mean.setValue(0.0)
        self._gauss_mean.setStyleSheet(SPINBOX_SS)
        gl.addWidget(self._gauss_mean, 0, 1)
        sigma_lbl = QLabel("σ")
        sigma_lbl.setStyleSheet(FIELD_SS)
        gl.addWidget(sigma_lbl, 1, 0)
        self._gauss_sigma = QDoubleSpinBox()
        self._gauss_sigma.setRange(0.1, 100.0)
        self._gauss_sigma.setValue(25.0)
        self._gauss_sigma.setStyleSheet(SPINBOX_SS)
        gl.addWidget(self._gauss_sigma, 1, 1)
        layout.addWidget(self._gauss_w)

        # Uniform params
        self._uniform_w = QWidget()
        ul = QGridLayout(self._uniform_w)
        ul.setContentsMargins(0, 0, 0, 0)
        ul.setSpacing(4)
        low_lbl = QLabel("Low")
        low_lbl.setStyleSheet(FIELD_SS)
        ul.addWidget(low_lbl, 0, 0)
        self._uniform_low = QDoubleSpinBox()
        self._uniform_low.setRange(-255.0, 0.0)
        self._uniform_low.setValue(-30.0)
        self._uniform_low.setStyleSheet(SPINBOX_SS)
        ul.addWidget(self._uniform_low, 0, 1)
        high_lbl = QLabel("High")
        high_lbl.setStyleSheet(FIELD_SS)
        ul.addWidget(high_lbl, 1, 0)
        self._uniform_high = QDoubleSpinBox()
        self._uniform_high.setRange(0.0, 255.0)
        self._uniform_high.setValue(30.0)
        self._uniform_high.setStyleSheet(SPINBOX_SS)
        ul.addWidget(self._uniform_high, 1, 1)
        self._uniform_w.hide()
        layout.addWidget(self._uniform_w)

        # inject button
        self.inject_btn = QPushButton("Inject Noise")
        self.inject_btn.setStyleSheet(APPLY_BTN_SS)
        layout.addWidget(self.inject_btn)

        self._type_combo.currentTextChanged.connect(self._on_type_changed)

        # divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"background:{BORDER};max-height:1px;")
        layout.addWidget(div)

        # ROI stats
        roi_lbl = QLabel("ROI STATISTICS")
        roi_lbl.setStyleSheet(HEADER_SS)
        layout.addWidget(roi_lbl)

        self._hist_canvas = _HistCanvas()
        layout.addWidget(self._hist_canvas)

        stats_w = QWidget()
        stats_w.setStyleSheet(f"background:{INPUT};border-radius:2px;")
        sg = QGridLayout(stats_w)
        sg.setContentsMargins(8, 6, 8, 6)
        sg.setSpacing(4)

        self._stat_labels: dict = {}
        for row, (key, display) in enumerate([
            ("mean", "Mean"), ("var", "Variance"),
            ("min", "Min"), ("max", "Max"), ("count", "Pixels")
        ]):
            sg.addWidget(QLabel(display, styleSheet=f"color:{MUTED};font-size:9px;"), row, 0)
            v = QLabel("—")
            v.setStyleSheet(f"color:{TEXT};font-size:9px;font-weight:bold;")
            v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            sg.addWidget(v, row, 1)
            self._stat_labels[key] = v
        layout.addWidget(stats_w)

        hint = QLabel("Draw an ROI on the image\nto compute local statistics")
        hint.setStyleSheet(f"color:{MUTED2};font-size:9px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        layout.addStretch()

    def set_image(self, image: np.ndarray):
        self._current_image = image

    def _on_type_changed(self, name: str):
        self._gauss_w.setVisible(name == "Gaussian")
        self._uniform_w.setVisible(name == "Uniform")

    def on_inject_clicked(self):
        if self._current_image is None:
            return
        try:
            validate_grayscale(self._current_image)
            noise_type = self._type_combo.currentText()

            if noise_type == "Gaussian":
                from processing.noise.noise_injection import add_gaussian_noise
                mean = self._gauss_mean.value()
                sigma = self._gauss_sigma.value()
                result = add_gaussian_noise(self._current_image, mean, sigma)
                op_name = f"Gaussian Noise μ={mean:.0f} σ={sigma:.0f}"
            else:
                from processing.noise.noise_injection import add_uniform_noise
                low = self._uniform_low.value()
                high = self._uniform_high.value()
                result = add_uniform_noise(self._current_image, low, high)
                op_name = f"Uniform Noise [{low:.0f},{high:.0f}]"

            result = normalize_to_uint8(result)
            self.noise_applied.emit(op_name, result)
        except (ImportError, NotImplementedError, Exception) as e:
            show_error_dialog("Noise Error", f"Noise injection failed.\n{e}")

    def update_roi_stats(self, image: np.ndarray, roi: QRect):
        try:
            validate_grayscale(image)
            from processing.noise.roi_stats import compute_roi_stats, extract_roi
            from processing.histogram.histogram_utils import compute_histogram
            roi_pixels = extract_roi(image, roi.x(), roi.y(), roi.width(), roi.height())
            if roi_pixels.size == 0:
                return
            hist = compute_histogram(roi_pixels)
            self._hist_canvas.set_histogram(hist)
            self._stat_labels["mean"].setText(f"{roi_pixels.mean():.2f}")
            self._stat_labels["var"].setText(f"{roi_pixels.var():.2f}")
            self._stat_labels["min"].setText(f"{int(roi_pixels.min())}")
            self._stat_labels["max"].setText(f"{int(roi_pixels.max())}")
            self._stat_labels["count"].setText(f"{roi_pixels.size}")
        except ImportError as e:
            import logging
            logging.getLogger('ciaw').error(f"ROI stats not ready: {e}")
            return
        except Exception as e:
            import logging
            logging.getLogger('ciaw').error(f"update_roi_stats error: {e}")
            return
