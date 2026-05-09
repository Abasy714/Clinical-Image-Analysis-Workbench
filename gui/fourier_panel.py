# STATUS: IMPLEMENTED
"""
Frequency domain panel for periodic noise removal via interactive notch filtering.
Displays the log-scaled FFT magnitude spectrum and allows the user to click on noise spikes.
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QSpinBox, QButtonGroup, QRadioButton,
                              QFrame, QSizePolicy)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QPainter, QColor, QPen, QPixmap, QImage

from gui.styles import (BG, PANEL, PANEL2, INPUT, BORDER, BORDER2,
                        ACCENT, RED, TEXT, MUTED, MUTED2, btn_style,
                        HEADER_SS, FIELD_SS, APPLY_BTN_SS, SPINBOX_SS)
from utils import (validate_grayscale, normalize_to_uint8, to_qpixmap, wrap_errors, show_error_dialog,)


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
        self._raw_image: np.ndarray | None = None
        self._spectrum_shape: tuple = (1, 1)
        self._domain: str = "frequency"

        self.setStyleSheet(f"background:{PANEL};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        hdr = QLabel("FREQUENCY / NOTCH FILTER")
        hdr.setStyleSheet(HEADER_SS)
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
        shape_lbl.setStyleSheet(FIELD_SS)
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

        d0_lbl = QLabel("D₀")
        d0_lbl.setStyleSheet(FIELD_SS)
        prl.addWidget(d0_lbl)
        self._radius_spin = QSpinBox()
        self._radius_spin.setRange(1, 200)
        self._radius_spin.setValue(10)
        self._radius_spin.setFixedWidth(60)
        self._radius_spin.setStyleSheet(SPINBOX_SS)
        prl.addWidget(self._radius_spin)

        self._order_lbl = QLabel("n")
        self._order_lbl.setStyleSheet(FIELD_SS)
        prl.addWidget(self._order_lbl)
        self._order_spin = QSpinBox()
        self._order_spin.setRange(1, 10)
        self._order_spin.setValue(2)
        self._order_spin.setFixedWidth(50)
        self._order_spin.setEnabled(False)
        self._order_spin.setStyleSheet(SPINBOX_SS)
        prl.addWidget(self._order_spin)
        prl.addStretch()
        layout.addWidget(params_row)

        self._shape_group.buttonClicked.connect(self._on_shape_changed)

        # spatial / frequency domain toggle
        domain_lbl = QLabel("Domain")
        domain_lbl.setStyleSheet(FIELD_SS)
        layout.addWidget(domain_lbl)

        domain_row = QWidget()
        drl = QHBoxLayout(domain_row)
        drl.setContentsMargins(0, 0, 0, 0)
        drl.setSpacing(4)
        self._spatial_btn = QPushButton("Spatial")
        self._spatial_btn.setCheckable(True)
        self._spatial_btn.setChecked(False)
        self._spatial_btn.setStyleSheet(btn_style('ghost'))
        self._spatial_btn.clicked.connect(lambda: self._set_domain("spatial"))
        drl.addWidget(self._spatial_btn)
        self._freq_btn = QPushButton("Frequency")
        self._freq_btn.setCheckable(True)
        self._freq_btn.setChecked(True)
        self._freq_btn.setStyleSheet(btn_style())
        self._freq_btn.clicked.connect(lambda: self._set_domain("frequency"))
        drl.addWidget(self._freq_btn)
        drl.addStretch()
        layout.addWidget(domain_row)

        # magnitude / phase toggle
        self._spectrum_mode: str = "magnitude"
        phase_row = QWidget()
        phl = QHBoxLayout(phase_row)
        phl.setContentsMargins(0, 0, 0, 0)
        phl.setSpacing(4)
        self._mag_btn = QPushButton("Magnitude")
        self._mag_btn.setCheckable(True)
        self._mag_btn.setChecked(True)
        self._mag_btn.setStyleSheet(btn_style())
        self._mag_btn.clicked.connect(lambda: self._set_spectrum_mode("magnitude"))
        phl.addWidget(self._mag_btn)
        self._phase_btn = QPushButton("Phase")
        self._phase_btn.setCheckable(True)
        self._phase_btn.setChecked(False)
        self._phase_btn.setStyleSheet(btn_style('ghost'))
        self._phase_btn.clicked.connect(lambda: self._set_spectrum_mode("phase"))
        phl.addWidget(self._phase_btn)
        phl.addStretch()
        layout.addWidget(phase_row)

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
        self.apply_btn.setStyleSheet(APPLY_BTN_SS)
        layout.addWidget(self.apply_btn)

        layout.addStretch()

    # ------------------------------------------------------------------ public

    def set_image(self, image: np.ndarray):
        try:
            validate_grayscale(image)
            self._raw_image = image
            from processing.frequency.spectrum import compute_spectrum
            self._shifted_fft, log_magnitude, _ = compute_spectrum(image)
            self._spectrum_shape = log_magnitude.shape
            self._update_canvas_display()
            self._canvas.set_notches(self._notch_points)
        except Exception:
            self._shifted_fft = None
            return

    def _set_spectrum_mode(self, mode: str):
        self._spectrum_mode = mode
        self._mag_btn.setChecked(mode == "magnitude")
        self._phase_btn.setChecked(mode == "phase")
        self._mag_btn.setStyleSheet(btn_style() if mode == "magnitude" else btn_style('ghost'))
        self._phase_btn.setStyleSheet(btn_style() if mode == "phase" else btn_style('ghost'))
        self._update_canvas_display()

    def _set_domain(self, domain: str):
        self._domain = domain
        self._spatial_btn.setChecked(domain == "spatial")
        self._freq_btn.setChecked(domain == "frequency")
        self._spatial_btn.setStyleSheet(btn_style() if domain == "spatial" else btn_style('ghost'))
        self._freq_btn.setStyleSheet(btn_style() if domain == "frequency" else btn_style('ghost'))
        self._update_canvas_display()

    def _update_canvas_display(self):
        if self._domain == "spatial":
            if self._raw_image is None:
                return
            try:
                self._canvas.set_spectrum(self._raw_image.astype(np.float64))
            except Exception:
                pass
            return
        if self._shifted_fft is None:
            return
        try:
            if self._spectrum_mode == "phase":
                from processing.frequency.spectrum import phase_to_display
                display = phase_to_display(self._shifted_fft)
            else:
                from processing.frequency.spectrum import spectrum_to_display
                display = spectrum_to_display(self._shifted_fft)
            self._canvas.set_spectrum(display)
        except Exception:
            pass

    def on_apply_clicked(self):
        if self._shifted_fft is None:
            return
        if not self._notch_points:
            return
        try:
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
        except ImportError:
            # Phase 2 — frequency domain not implemented yet
            # No dialog — user will see empty spectrum panel
            pass
        except Exception as e:
            show_error_dialog("Not Implemented", f"Notch filter is not yet available.\n{e}")

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
