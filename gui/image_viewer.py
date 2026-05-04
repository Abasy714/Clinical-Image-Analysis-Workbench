# STATUS: IMPLEMENTED
"""
Image display widget with zoom controls and interactive ROI drawing.
Powered entirely by custom interpolation — no built-in zoom libraries used.
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                              QPushButton, QScrollArea, QRubberBand, QSizePolicy)
from PyQt6.QtCore import Qt, QRect, QPoint, QSize, pyqtSignal, QEvent
from PyQt6.QtGui import QPainter, QPen, QColor

from gui.styles import (BG, PANEL, BORDER, BORDER2, ACCENT, MUTED, MUTED2, btn_style)
from utils import (to_qpixmap, normalize_to_uint8, wrap_errors,)


class _ImageLabel(QLabel):
    """QLabel that paints a semi-transparent histogram waveform on top of the image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._histogram: np.ndarray | None = None
        self._show_overlay: bool = True

    def set_histogram(self, histogram: np.ndarray | None):
        self._histogram = histogram
        self.update()

    def set_show_overlay(self, visible: bool):
        self._show_overlay = visible
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._show_overlay or self._histogram is None or len(self._histogram) == 0:
            return
        painter = QPainter(self)
        w, h = self.width(), self.height()
        overlay_h = max(40, h // 6)
        y_base = h - 4
        hist_max = float(self._histogram.max()) or 1.0
        n = len(self._histogram)
        painter.setOpacity(0.55)
        pen = QPen(QColor(ACCENT))
        pen.setWidth(1)
        painter.setPen(pen)
        for i, val in enumerate(self._histogram):
            bar_h = int(val / hist_max * overlay_h)
            x = int(i / n * w)
            painter.drawLine(x, y_base, x, y_base - bar_h)
        painter.end()


def _sep() -> QWidget:
    w = QWidget()
    w.setFixedSize(1, 20)
    w.setStyleSheet(f"background:{BORDER};")
    return w


class ImageViewer(QWidget):
    roi_selected = pyqtSignal(QRect)
    zoom_changed = pyqtSignal(int)
    coords_changed = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: np.ndarray | None = None
        self._zoom: int = 100
        self._interp_mode: str = 'nearest'
        self._roi: QRect | None = None
        self._origin: QPoint = QPoint()
        self._show_hist_overlay: bool = True

        self.setStyleSheet(f"background:{BG};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- toolbar ----
        toolbar = QWidget()
        toolbar.setFixedHeight(32)
        toolbar.setStyleSheet(f"background:{PANEL};border-bottom:1px solid {BORDER};")
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(6, 2, 6, 2)
        tl.setSpacing(4)

        self._btn_zout = QPushButton("−")
        self._btn_zout.setFixedSize(24, 24)
        self._btn_zout.setStyleSheet(btn_style('ghost'))
        self._btn_zout.clicked.connect(self.zoom_out)
        tl.addWidget(self._btn_zout)

        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(36)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._zoom_label.setStyleSheet(f"color:{MUTED};font-size:9px;")
        tl.addWidget(self._zoom_label)

        self._btn_zin = QPushButton("+")
        self._btn_zin.setFixedSize(24, 24)
        self._btn_zin.setStyleSheet(btn_style('ghost'))
        self._btn_zin.clicked.connect(self.zoom_in)
        tl.addWidget(self._btn_zin)

        self._btn_fit = QPushButton("Fit")
        self._btn_fit.setFixedHeight(24)
        self._btn_fit.setStyleSheet(btn_style('ghost'))
        self._btn_fit.clicked.connect(self.zoom_fit)
        tl.addWidget(self._btn_fit)

        tl.addWidget(_sep())

        self._btn_interp = QPushButton("NN")
        self._btn_interp.setFixedHeight(24)
        self._btn_interp.setStyleSheet(btn_style())
        self._btn_interp.clicked.connect(self._toggle_interp)
        tl.addWidget(self._btn_interp)

        tl.addWidget(_sep())

        self._btn_overlay = QPushButton("Hist ✓")
        self._btn_overlay.setFixedHeight(24)
        self._btn_overlay.setStyleSheet(btn_style())
        self._btn_overlay.clicked.connect(self._toggle_overlay)
        tl.addWidget(self._btn_overlay)

        tl.addStretch()

        self._coord_label = QLabel("x:—  y:—")
        self._coord_label.setStyleSheet(f"color:{MUTED2};font-size:9px;")
        tl.addWidget(self._coord_label)

        layout.addWidget(toolbar)

        # ---- scroll area ----
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setStyleSheet(f"QScrollArea{{background:{BG};border:none;}}")
        self._scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._img_label = _ImageLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setStyleSheet(f"background:{BG};color:{MUTED};font-size:11px;")
        self._img_label.setText("No image loaded")
        self._scroll.setWidget(self._img_label)
        layout.addWidget(self._scroll)

        # rubber band on viewport
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self._scroll.viewport())
        self._scroll.viewport().installEventFilter(self)
        self._scroll.viewport().setMouseTracking(True)

        # processing overlay
        self._overlay = QLabel("PROCESSING…", self._scroll.viewport())
        self._overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay.setStyleSheet(
            f"background:rgba(17,18,16,210);color:{ACCENT};font-size:13px;font-weight:bold;"
        )
        self._overlay.hide()

    # ------------------------------------------------------------------ public API

    def set_image(self, image: np.ndarray):
        self._image = image
        self._render()

    def zoom_in(self):
        self._zoom = min(self._zoom + 10, 800)
        self._render()

    def zoom_out(self):
        self._zoom = max(self._zoom - 10, 25)
        self._render()

    def zoom_fit(self):
        self._zoom = 100
        self._render()

    def set_interpolation_mode(self, mode: str):
        if mode in ('nearest', 'bilinear'):
            self._interp_mode = mode
            self._render()

    def get_roi(self) -> QRect | None:
        return self._roi

    def show_processing_overlay(self, visible: bool):
        if visible:
            vp = self._scroll.viewport()
            self._overlay.setGeometry(0, 0, vp.width(), vp.height())
            self._overlay.raise_()
            self._overlay.show()
        else:
            self._overlay.hide()

    def set_histogram_overlay(self, histogram: np.ndarray):
        self._img_label.set_histogram(histogram)

    # ------------------------------------------------------------------ events

    def eventFilter(self, obj, event):
        if obj is self._scroll.viewport():
            t = event.type()
            if t == QEvent.Type.MouseButtonPress:
                self.mousePressEvent(event)
            elif t == QEvent.Type.MouseMove:
                self.mouseMoveEvent(event)
            elif t == QEvent.Type.MouseButtonRelease:
                self.mouseReleaseEvent(event)
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin = event.position().toPoint()
            self._rubber_band.setGeometry(QRect(self._origin, QSize()))
            self._rubber_band.show()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if not self._origin.isNull():
            self._rubber_band.setGeometry(QRect(self._origin, pos).normalized())
        ix, iy = self._viewport_to_image(pos)
        self._coord_label.setText(f"x:{ix}  y:{iy}")
        self.coords_changed.emit(ix, iy)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self._origin.isNull():
            end = event.position().toPoint()
            vp_rect = QRect(self._origin, end).normalized()
            self._rubber_band.hide()
            self._origin = QPoint()
            self._roi = self._viewport_rect_to_image_rect(vp_rect)
            self.roi_selected.emit(self._roi)

    # ------------------------------------------------------------------ private

    @wrap_errors
    def _render(self):
        if self._image is None:
            return
        zoom_factor = self._zoom / 100.0
        try:
            from processing.interpolation.zoom import apply_zoom
            zoomed = apply_zoom(self._image, zoom_factor, self._interp_mode)
        except Exception:
            zoomed = self._image
        data = normalize_to_uint8(zoomed)
        pixmap = to_qpixmap(data)
        self._img_label.setPixmap(pixmap)
        self._img_label.setFixedSize(pixmap.width(), pixmap.height())
        self._zoom_label.setText(f"{self._zoom}%")
        self.zoom_changed.emit(self._zoom)

    def _viewport_to_image(self, point: QPoint) -> tuple:
        if self._image is None:
            return 0, 0
        hscroll = self._scroll.horizontalScrollBar().value()
        vscroll = self._scroll.verticalScrollBar().value()
        zoom_factor = self._zoom / 100.0
        ix = int((point.x() + hscroll) / zoom_factor)
        iy = int((point.y() + vscroll) / zoom_factor)
        h, w = self._image.shape[:2]
        return max(0, min(ix, w - 1)), max(0, min(iy, h - 1))

    def _viewport_rect_to_image_rect(self, vp_rect: QRect) -> QRect:
        x1, y1 = self._viewport_to_image(vp_rect.topLeft())
        x2, y2 = self._viewport_to_image(vp_rect.bottomRight())
        return QRect(QPoint(x1, y1), QPoint(x2, y2)).normalized()

    def _toggle_interp(self):
        if self._interp_mode == 'nearest':
            self.set_interpolation_mode('bilinear')
            self._btn_interp.setText("BL")
        else:
            self.set_interpolation_mode('nearest')
            self._btn_interp.setText("NN")

    def _toggle_overlay(self):
        self._show_hist_overlay = not self._show_hist_overlay
        self._img_label.set_show_overlay(self._show_hist_overlay)
        self._btn_overlay.setText("Hist ✓" if self._show_hist_overlay else "Hist")
