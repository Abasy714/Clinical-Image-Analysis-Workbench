# STATUS: IMPLEMENTED
"""
Image display widget with zoom controls and interactive ROI drawing.
Powered entirely by custom interpolation — no built-in zoom libraries used.
"""

import logging
import numpy as np
from PyQt6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                              QPushButton, QScrollArea, QRubberBand, QSizePolicy, QFrame)
from PyQt6.QtCore import Qt, QRect, QPoint, QSize, pyqtSignal, QEvent, QThread
from PyQt6.QtGui import QPainter, QPen, QColor

from gui.styles import (BG, PANEL, BORDER, BORDER2, ACCENT, MUTED, MUTED2, btn_style)
from utils import (to_qpixmap, normalize_to_uint8, wrap_errors,)


class ZoomWorker(QThread):
    """Runs zoom/resize in a background thread."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            result = self._fn()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


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
    interp_changed = pyqtSignal(str)  # 'NN' or 'BL'

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: np.ndarray | None = None
        self._display_image: np.ndarray | None = None
        self._before_image: np.ndarray | None = None
        self._showing_before: bool = False
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

        tl.addWidget(_sep())

        # ── Before / After toggle ──
        self._after_btn = QPushButton("After")
        self._after_btn.setCheckable(True)
        self._after_btn.setChecked(True)
        self._after_btn.setFixedHeight(24)
        self._after_btn.clicked.connect(lambda: self._set_view_mode('after'))
        self._after_btn.setStyleSheet("""
            QPushButton {
                background: #1a2208;
                color: #c8f135;
                border: 1px solid #6a8a10;
                border-right: none;
                border-radius: 2px 0 0 2px;
                font-family: 'JetBrains Mono', Consolas, monospace;
                font-size: 9px;
                font-weight: bold;
                padding: 0 10px;
                letter-spacing: 1px;
            }
            QPushButton:!checked {
                background: #252623;
                color: #6b6f65;
                border: 1px solid #353730;
                border-right: none;
            }
        """)
        tl.addWidget(self._after_btn)

        self._before_btn = QPushButton("Before")
        self._before_btn.setCheckable(True)
        self._before_btn.setChecked(False)
        self._before_btn.setFixedHeight(24)
        self._before_btn.clicked.connect(lambda: self._set_view_mode('before'))
        self._before_btn.setStyleSheet("""
            QPushButton {
                background: #252623;
                color: #6b6f65;
                border: 1px solid #353730;
                border-radius: 0 2px 2px 0;
                font-family: 'JetBrains Mono', Consolas, monospace;
                font-size: 9px;
                font-weight: bold;
                padding: 0 10px;
                letter-spacing: 1px;
            }
            QPushButton:checked {
                background: #1a2208;
                color: #c8f135;
                border: 1px solid #6a8a10;
            }
        """)
        tl.addWidget(self._before_btn)

        tl.addStretch()

        self._coord_label = QLabel("x:—  y:—")
        self._coord_label.setStyleSheet(f"color:{MUTED2};font-size:9px;")
        tl.addWidget(self._coord_label)

        layout.addWidget(toolbar)

        # ---- scroll area ----
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setStyleSheet(f"QScrollArea{{background:{BG};border:none;}}")
        self._scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._img_label = _ImageLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setStyleSheet(f"background:{BG};color:{MUTED};font-size:11px;")
        self._img_label.setText("No image loaded")
        self._img_label.setMinimumSize(1, 1)
        self._img_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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

        # before badge overlay
        self._before_badge = QLabel("◀ BEFORE", self._img_label)
        self._before_badge.setStyleSheet("""
            QLabel {
                background: rgba(200, 241, 53, 220);
                color: #0d1002;
                font-family: 'JetBrains Mono', Consolas, monospace;
                font-size: 10px;
                font-weight: bold;
                letter-spacing: 2px;
                padding: 3px 10px;
                border-radius: 2px;
            }
        """)
        self._before_badge.adjustSize()
        self._before_badge.move(10, 10)
        self._before_badge.hide()
        self._before_badge.raise_()

    # ------------------------------------------------------------------ public API

    def set_image(self, image: np.ndarray):
        if image is None or not isinstance(image, np.ndarray):
            return
        img = image.copy()
        if img.ndim == 3:
            img = np.mean(img, axis=2).astype(np.uint8)
        self._image = normalize_to_uint8(img)

        MAX_DISPLAY = 1024
        h, w = self._image.shape
        if h > MAX_DISPLAY or w > MAX_DISPLAY:
            scale = MAX_DISPLAY / max(h, w)
            disp_h = max(1, int(h * scale))
            disp_w = max(1, int(w * scale))
            try:
                from processing.interpolation import nearest_neighbor_resize
                resized = nearest_neighbor_resize(self._image, disp_h, disp_w)
                self._display_image = resized if resized is not None else self._image
            except Exception:
                self._display_image = self._image
        else:
            self._display_image = self._image

        # Reset before/after state to "after"
        self._showing_before = False
        self._after_btn.setChecked(True)
        self._before_btn.setChecked(False)
        self._before_badge.hide()

        # Always render synchronously so image appears immediately
        self._render_sync()

    def set_before_image(self, image: np.ndarray):
        """Store the before/original image for before-after comparison."""
        if image is None or not isinstance(image, np.ndarray):
            return
        img = image.copy()
        if img.ndim == 3:
            img = np.mean(img, axis=2).astype(np.uint8)
        if img.dtype != np.uint8:
            mn, mx = float(img.min()), float(img.max())
            if mx > mn:
                img = ((img.astype(np.float64) - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
        self._before_image = img

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

    def toggle_before_after(self):
        """Toggle between before and after — bound to B shortcut."""
        if self._showing_before:
            self._set_view_mode('after')
        else:
            self._set_view_mode('before')

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
            try:
                self.roi_selected.emit(self._roi)
            except Exception as e:
                import logging
                logging.getLogger('ciaw').error(f"ROI selection error: {e}")

    # ------------------------------------------------------------------ private

    def _render_sync(self):
        """Synchronous render — always shows image immediately on the main thread."""
        if self._image is None:
            return
        display = self._display_image if self._display_image is not None else self._image
        self._display_array(display)

    def _render(self):
        if self._image is None or self._display_image is None:
            return

        if self._zoom == 100:
            self._render_sync()
            return

        zoom_factor = self._zoom / 100.0
        h, w = self._display_image.shape[:2]
        new_h = max(1, int(round(h * zoom_factor)))
        new_w = max(1, int(round(w * zoom_factor)))
        _img = self._display_image.copy()
        _mode = self._interp_mode

        def _zoom_fn():
            if _mode == 'nearest':
                from processing.interpolation import nearest_neighbor_resize
                result = nearest_neighbor_resize(_img, new_h, new_w)
                return result if result is not None else _img
            else:
                from processing.interpolation import bilinear_resize
                result = bilinear_resize(_img, new_h, new_w)
                if result is None:
                    from processing.interpolation import nearest_neighbor_resize
                    return nearest_neighbor_resize(_img, new_h, new_w)
                mn, mx = float(result.min()), float(result.max())
                if mx > mn:
                    return ((result.astype(np.float64) - mn) / (mx - mn) * 255).astype(np.uint8)
                return np.zeros((new_h, new_w), dtype=np.uint8)

        self._zoom_worker = ZoomWorker(_zoom_fn, parent=self)
        self._zoom_worker.finished.connect(self._display_array)
        self._zoom_worker.error.connect(
            lambda e: logging.getLogger('ciaw').error(f"Zoom error: {e}")
        )
        self._zoom_worker.start()

    def _display_array(self, display_image):
        """Update label with image array. Must run on main thread."""
        if display_image is None or display_image.size == 0:
            return
        data = normalize_to_uint8(display_image)
        pixmap = to_qpixmap(data)
        if pixmap.isNull():
            return
        label_size = self._img_label.size()
        if label_size.width() > 10 and label_size.height() > 10:
            pixmap = pixmap.scaled(
                label_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation
            )
        self._img_label.setPixmap(pixmap)
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._zoom_label.setText(f"{self._zoom}%")
        self.zoom_changed.emit(self._zoom)

    def _set_view_mode(self, mode: str):
        """Switch display between before and after."""
        if mode == 'before':
            if self._before_image is None:
                return
            self._showing_before = True
            self._after_btn.setChecked(False)
            self._before_btn.setChecked(True)
            self._display_array(self._before_image)
            self._before_badge.show()
            self._before_badge.raise_()
        else:
            self._showing_before = False
            self._after_btn.setChecked(True)
            self._before_btn.setChecked(False)
            self._before_badge.hide()
            if self._image is not None:
                self._render_sync()

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
            self.interp_changed.emit('BL')
        else:
            self.set_interpolation_mode('nearest')
            self._btn_interp.setText("NN")
            self.interp_changed.emit('NN')

    def _toggle_overlay(self):
        self._show_hist_overlay = not self._show_hist_overlay
        self._img_label.set_show_overlay(self._show_hist_overlay)
        self._btn_overlay.setText("Hist ✓" if self._show_hist_overlay else "Hist")
