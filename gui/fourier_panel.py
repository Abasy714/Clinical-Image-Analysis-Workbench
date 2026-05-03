# STATUS: IMPLEMENTED
"""
Frequency domain panel for periodic noise removal via interactive notch filtering.
Displays the log-scaled FFT magnitude spectrum and allows the user to click on noise spikes.
"""

# PyQt6.QtWidgets — QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QSpinBox, QLabel
# PyQt6.QtCore — pyqtSignal, QPoint
# matplotlib.backends.backend_qtagg — FigureCanvasQTAgg
# matplotlib.figure — Figure
# numpy — spectrum data handling
# processing.frequency.spectrum — compute_spectrum, spectrum_to_display
# processing.frequency.notch_filter — create_notch_filter, apply_notch_filter

# FUNCTIONS / CLASSES
# class FourierPanel(QWidget):
#   def __init__: build UI — spectrum canvas, filter shape selector, radius spinner, apply button
#   def set_image: compute and display spectrum for loaded image
#   def on_spectrum_clicked: capture (u,v) click → generate notch + conjugate → preview mask on spectrum
#   def on_apply_clicked: multiply mask with FFT → IFFT → emit cleaned image
#   def _display_spectrum: render log-magnitude to matplotlib canvas
# signal: notch_applied(np.ndarray)

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale, normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call in set_image before computing spectrum
# normalize_to_uint8(cleaned)         # call on ifft result before emitting notch_applied
# @wrap_errors                        # decorate on_spectrum_clicked and on_apply_clicked
# Note: spectrum_to_display() from processing.frequency.spectrum handles its own normalization
#       only call normalize_to_uint8 on the final reconstructed spatial-domain image

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QSpinBox, QButtonGroup, QRadioButton,
                              QFrame, QSizePolicy)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QPainter, QColor, QPen, QPixmap, QImage

from gui.styles import (BG, PANEL, PANEL2, INPUT, BORDER, BORDER2,
                        ACCENT, RED, TEXT, MUTED, MUTED2, btn_style)
from utils import (validate_grayscale, normalize_to_uint8, to_qpixmap, wrap_errors,)


class SpectrumCanvas(QWidget):
    """Custom canvas showing FFT spectrum with clickable notch placement."""

    clicked_at = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._spectrum_pixmap: QPixmap | None = None
        self._notches: list = []
        self._spectrum_shape: tuple = (1, 1)
        self.setFixedHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(f"background:{BG};border:1px solid {BORDER};border-radius:2px;")
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_spectrum(self, log_magnitude: np.ndarray):
        self._spectrum_shape = log_magnitude.shape
        data = normalize_to_uint8(log_magnitude)
        self._spectrum_pixmap = to_qpixmap(data)
        self.update()

    def set_notches(self, notches: list):
        self._notches = notches
        self.update()

    def clear_notches(self):
        self._notches = []
        self.update()

    def mousePressEvent(self, event):
        if self._spectrum_pixmap is None:
            return
        h_sp, w_sp = self._spectrum_shape
        cw, ch = self.width(), self.height()
        u = int(event.position().x() / cw * w_sp)
        v = int(event.position().y() / ch * h_sp)
        u = max(0, min(u, w_sp - 1))
        v = max(0, min(v, h_sp - 1))
        self.clicked_at.emit(u, v)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(BG))

        if self._spectrum_pixmap is not None:
            scaled = self._spectrum_pixmap.scaled(
                self.width(), self.height(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
            painter.drawPixmap(0, 0, scaled)

        # crosshair
        pen = QPen(QColor(BORDER))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(self.width() // 2, 0, self.width() // 2, self.height())
        painter.drawLine(0, self.height() // 2, self.width(), self.height() // 2)

        # notch markers
        if self._spectrum_shape[0] > 1:
            h_sp, w_sp = self._spectrum_shape
            cw, ch = self.width(), self.height()
            for u, v in self._notches:
                cx = int(u / w_sp * cw)
                cy = int(v / h_sp * ch)
                # solid red circle
                painter.setPen(QPen(QColor(RED), 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(cx - 6, cy - 6, 12, 12)
                # dashed outer ring
                pen2 = QPen(QColor(RED), 1, Qt.PenStyle.DashLine)
                painter.setPen(pen2)
                painter.drawEllipse(cx - 10, cy - 10, 20, 20)

        painter.end()


class FourierPanel(QWidget):
    notch_applied = pyqtSignal(str, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._notch_points: list = []
        self._shifted_fft: np.ndarray | None = None
        self._spectrum_shape: tuple = (1, 1)

        self.setStyleSheet(f"background:{PANEL};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        hdr = QLabel("FREQUENCY / NOTCH FILTER")
        hdr.setStyleSheet(f"color:{ACCENT};font-size:9px;font-weight:bold;letter-spacing:.15em;")
        layout.addWidget(hdr)

        # spectrum canvas
        self._canvas = SpectrumCanvas()
        self._canvas.clicked_at.connect(self._on_canvas_clicked)
        layout.addWidget(self._canvas)

        # notch count
        self._notch_count_lbl = QLabel("0 notches placed")
        self._notch_count_lbl.setStyleSheet(f"color:{MUTED};font-size:9px;")
        layout.addWidget(self._notch_count_lbl)

        # filter shape
        shape_lbl = QLabel("Filter shape")
        shape_lbl.setStyleSheet(f"color:{MUTED};font-size:9px;")
        layout.addWidget(shape_lbl)

        shape_row = QWidget()
        srl = QHBoxLayout(shape_row)
        srl.setContentsMargins(0, 0, 0, 0)
        srl.setSpacing(4)
        self._shape_group = QButtonGroup(self)
        for label in ("Ideal", "Butterworth", "Gaussian"):
            rb = QRadioButton(label)
            rb.setStyleSheet(f"color:{TEXT};font-size:9px;")
            self._shape_group.addButton(rb)
            srl.addWidget(rb)
            if label == "Ideal":
                rb.setChecked(True)
        layout.addWidget(shape_row)

        # radius + order
        params_row = QWidget()
        prl = QHBoxLayout(params_row)
        prl.setContentsMargins(0, 0, 0, 0)
        prl.setSpacing(8)

        prl.addWidget(QLabel("D₀", styleSheet=f"color:{MUTED};font-size:9px;"))
        self._radius_spin = QSpinBox()
        self._radius_spin.setRange(1, 200)
        self._radius_spin.setValue(10)
        self._radius_spin.setFixedWidth(60)
        prl.addWidget(self._radius_spin)

        self._order_lbl = QLabel("n")
        self._order_lbl.setStyleSheet(f"color:{MUTED};font-size:9px;")
        prl.addWidget(self._order_lbl)
        self._order_spin = QSpinBox()
        self._order_spin.setRange(1, 10)
        self._order_spin.setValue(2)
        self._order_spin.setFixedWidth(50)
        self._order_spin.setEnabled(False)
        prl.addWidget(self._order_spin)
        prl.addStretch()
        layout.addWidget(params_row)

        self._shape_group.buttonClicked.connect(self._on_shape_changed)

        # clear notches
        clr_btn = QPushButton("Clear notches")
        clr_btn.setStyleSheet(btn_style('ghost'))
        clr_btn.clicked.connect(self._clear_notches)
        layout.addWidget(clr_btn)

        hint = QLabel("Click spectrum to place notch.\nAlt+click to remove nearest pair.")
        hint.setStyleSheet(f"color:{MUTED2};font-size:9px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        # apply button
        self.apply_btn = QPushButton("Apply Notch Filter")
        self.apply_btn.setStyleSheet(btn_style('primary'))
        layout.addWidget(self.apply_btn)

        layout.addStretch()

    # ------------------------------------------------------------------ public

    @wrap_errors
    def set_image(self, image: np.ndarray):
        validate_grayscale(image)
        from processing.frequency.spectrum import compute_spectrum
        shifted_fft, log_magnitude = compute_spectrum(image)
        self._shifted_fft = shifted_fft
        self._spectrum_shape = image.shape
        self._canvas.set_spectrum(log_magnitude)
        self._clear_notches()

    @wrap_errors
    def on_apply_clicked(self):
        if self._shifted_fft is None:
            return
        if not self._notch_points:
            return
        from processing.frequency.notch_filter import create_notch_filter, apply_notch_filter
        from processing.frequency.spectrum import inverse_spectrum

        shape = self._shifted_fft.shape
        checked = self._shape_group.checkedButton()
        kind = checked.text().lower() if checked else "ideal"
        radius = self._radius_spin.value()
        order = self._order_spin.value()

        combined = np.ones(shape, dtype=np.float64)
        for u, v in self._notch_points:
            mask = create_notch_filter(shape, u, v, radius, kind, order)
            combined *= mask

        filtered = apply_notch_filter(self._shifted_fft, combined)
        result = inverse_spectrum(filtered)
        result = normalize_to_uint8(result)
        self.notch_applied.emit(f"Notch {kind.title()} D₀={radius}", result)

    # ------------------------------------------------------------------ private

    def _on_canvas_clicked(self, u: int, v: int):
        h, w = self._spectrum_shape
        mirror_u = (h - u) % h
        mirror_v = (w - v) % w
        self._notch_points.append((u, v))
        if (mirror_u, mirror_v) != (u, v):
            self._notch_points.append((mirror_u, mirror_v))
        self._canvas.set_notches(self._notch_points)
        n_pairs = len(self._notch_points) // 2
        self._notch_count_lbl.setText(f"{n_pairs} notch pair{'s' if n_pairs != 1 else ''} placed")

    def _clear_notches(self):
        self._notch_points.clear()
        self._canvas.clear_notches()
        self._notch_count_lbl.setText("0 notches placed")

    def _on_shape_changed(self, btn):
        self._order_spin.setEnabled(btn.text() == "Butterworth")
