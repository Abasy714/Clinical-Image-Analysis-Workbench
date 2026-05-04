# STATUS: IMPLEMENTED
"""
Panel for local histogram equalization and ROI histogram display.
User inputs block size; algorithm equalizes local regions to enhance contrast in medical images.
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QButtonGroup, QRadioButton, QGridLayout,
                              QSizePolicy, QFrame)
from PyQt6.QtCore import pyqtSignal, Qt, QRect
from PyQt6.QtGui import QPainter, QColor, QPen

from gui.styles import (BG, PANEL, PANEL2, INPUT, BORDER, BORDER2,
                        ACCENT, TEXT, MUTED, MUTED2, btn_style,
                        HEADER_SS, FIELD_SS, APPLY_BTN_SS)
from utils import (validate_grayscale, normalize_to_uint8, wrap_errors, show_error_dialog,)


class HistogramCanvas(QWidget):
    """Custom widget that draws a histogram using QPainter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._histogram: np.ndarray | None = None
        self.setFixedHeight(80)
        self.setStyleSheet(f"background:{BG};border:1px solid {BORDER};border-radius:2px;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

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
        padding = 4
        canvas_w = w - padding * 2
        canvas_h = h - padding * 2
        hist = self._histogram.astype(np.float64)
        hist_max = float(hist.max()) or 1.0
        n = len(hist)

        # baseline
        pen = QPen(QColor(BORDER))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(padding, h - padding, w - padding, h - padding)

        # bars
        accent_color = QColor(ACCENT)
        for i, val in enumerate(hist):
            bar_h = int(val / hist_max * canvas_h)
            x = padding + int(i / n * canvas_w)
            bar_w = max(1, int(canvas_w / n))
            alpha = max(80, int(255 * val / hist_max))
            accent_color.setAlpha(alpha)
            painter.fillRect(x, h - padding - bar_h, bar_w, bar_h, accent_color)

        painter.end()


class HistogramPanel(QWidget):
    equalization_applied = pyqtSignal(str, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{PANEL};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        hdr = QLabel("LOCAL HISTOGRAM EQ")
        hdr.setStyleSheet(HEADER_SS)
        layout.addWidget(hdr)

        # block size
        bsz_lbl = QLabel("Block size")
        bsz_lbl.setStyleSheet(FIELD_SS)
        layout.addWidget(bsz_lbl)

        bsz_row = QWidget()
        brl = QHBoxLayout(bsz_row)
        brl.setContentsMargins(0, 0, 0, 0)
        brl.setSpacing(4)
        self._bsz_group = QButtonGroup(self)
        for sz in (4, 8, 16, 32):
            rb = QRadioButton(str(sz))
            rb.setProperty("bsz", sz)
            rb.setStyleSheet(f"color:{TEXT};font-size:9px;")
            self._bsz_group.addButton(rb)
            brl.addWidget(rb)
            if sz == 8:
                rb.setChecked(True)
        brl.addStretch()
        layout.addWidget(bsz_row)

        self.apply_btn = QPushButton("Apply Local EQ")
        self.apply_btn.setStyleSheet(APPLY_BTN_SS)
        layout.addWidget(self.apply_btn)

        # divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"background:{BORDER};max-height:1px;")
        layout.addWidget(div)

        # ROI histogram
        roi_lbl = QLabel("ROI HISTOGRAM")
        roi_lbl.setStyleSheet(HEADER_SS)
        layout.addWidget(roi_lbl)

        self._canvas = HistogramCanvas()
        layout.addWidget(self._canvas)

        # stats grid
        stats_w = QWidget()
        stats_w.setStyleSheet(f"background:{INPUT};border-radius:2px;")
        sg = QGridLayout(stats_w)
        sg.setContentsMargins(8, 6, 8, 6)
        sg.setSpacing(4)

        self._stat_labels: dict = {}
        for row, (key, display) in enumerate([("mean", "Mean"), ("var", "Variance"),
                                               ("min", "Min"), ("max", "Max")]):
            k_lbl = QLabel(display)
            k_lbl.setStyleSheet(f"color:{MUTED};font-size:9px;")
            sg.addWidget(k_lbl, row, 0)
            v_lbl = QLabel("—")
            v_lbl.setStyleSheet(f"color:{TEXT};font-size:9px;font-weight:bold;")
            v_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            sg.addWidget(v_lbl, row, 1)
            self._stat_labels[key] = v_lbl

        layout.addWidget(stats_w)

        hint = QLabel("Draw an ROI on the image\nto compute local statistics")
        hint.setStyleSheet(f"color:{MUTED2};font-size:9px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        layout.addStretch()

    @wrap_errors
    def on_apply_clicked(self, image: np.ndarray):
        validate_grayscale(image)
        bsz = 8
        for btn in self._bsz_group.buttons():
            if btn.isChecked():
                bsz = btn.property("bsz")
                break
        try:
            from processing.histogram.local_equalization import local_histogram_equalization
            result = local_histogram_equalization(image, bsz)
            result = normalize_to_uint8(result)
            self.equalization_applied.emit(f"Local EQ {bsz}×{bsz}", result)
        except (ImportError, NotImplementedError, Exception) as e:
            show_error_dialog("Not Implemented", f"Local histogram equalization is not yet available.\n{e}")

    def update_roi(self, image: np.ndarray, roi: QRect):
        try:
            validate_grayscale(image)
            from processing.noise.roi_stats import extract_roi
            from processing.histogram.histogram_utils import compute_histogram
            roi_pixels = extract_roi(image, roi.x(), roi.y(), roi.width(), roi.height())
            if roi_pixels.size == 0:
                return
            hist = compute_histogram(roi_pixels)
            self._canvas.set_histogram(hist)
            self._stat_labels["mean"].setText(f"{roi_pixels.mean():.2f}")
            self._stat_labels["var"].setText(f"{roi_pixels.var():.2f}")
            self._stat_labels["min"].setText(f"{int(roi_pixels.min())}")
            self._stat_labels["max"].setText(f"{int(roi_pixels.max())}")
        except ImportError as e:
            import logging
            logging.getLogger('ciaw').error(f"ROI histogram not ready: {e}")
            return
        except Exception as e:
            import logging
            logging.getLogger('ciaw').error(f"update_roi error: {e}")
            return
