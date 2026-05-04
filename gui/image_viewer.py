# STATUS: IMPLEMENTED
"""
Image display widget with zoom controls and interactive ROI drawing.
Powered entirely by custom interpolation — no built-in zoom libraries used.
"""

import logging
import numpy as np
from PyQt6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                              QPushButton, QScrollArea, QRubberBand, QSizePolicy, QFrame)
from PyQt6.QtCore import Qt, QRect, QPoint, QSize, pyqtSignal, QEvent, QThread, QTimer

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
    """QLabel used for displaying the current image."""


def _sep() -> QWidget:
    w = QWidget()
    w.setFixedSize(1, 20)
    w.setStyleSheet(f"background:{BORDER};")
    return w


class ImageViewer(QWidget):
    _MIN_ZOOM = 1
    _MAX_ZOOM = 1000

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
        self._fit_to_window: bool = True
        self._interp_mode: str = 'nearest'
        self._roi: QRect | None = None
        self._origin: QPoint = QPoint()

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
        self._btn_fit.clicked.connect(self.fit_to_window)
        tl.addWidget(self._btn_fit)

        tl.addWidget(_sep())

        self._btn_interp = QPushButton("NN")
        self._btn_interp.setFixedHeight(24)
        self._btn_interp.setStyleSheet(btn_style())
        self._btn_interp.clicked.connect(self._toggle_interp)
        tl.addWidget(self._btn_interp)

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
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setStyleSheet(f"QScrollArea{{background:{BG};border:none;}}")
        self._scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._img_label = _ImageLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setStyleSheet(f"background:{BG};color:{MUTED};font-size:11px;")
        self._img_label.setText("No image loaded")
        self._img_label.setMinimumSize(1, 1)
        self._img_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
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

    def set_image(self, image: np.ndarray, fit_to_window: bool = False):
        if image is None or not isinstance(image, np.ndarray):
            return
        img = image.copy()
        if img.ndim == 3:
            img = np.mean(img, axis=2).astype(np.uint8)
        self._image = normalize_to_uint8(img)

        self._display_image = self._image

        # Reset before/after state to "after"
        self._showing_before = False
        self._fit_to_window = fit_to_window
        self._after_btn.setChecked(True)
        self._before_btn.setChecked(False)
        self._before_badge.hide()

        if fit_to_window:
            # Wait one event loop so the scroll viewport has its final size.
            QTimer.singleShot(0, self.fit_to_window)
        else:
            self._render()

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
        self._set_zoom(self._zoom + 10, fit_mode=False)

    def zoom_out(self):
        self._set_zoom(self._zoom - 10, fit_mode=False)

    def zoom_fit(self):
        self.fit_to_window()

    def fit_to_window(self):
        base = self._get_display_base()
        if base is None:
            return
        self._fit_to_window = True
        self._set_zoom(self._compute_fit_zoom(base), fit_mode=True)

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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_to_window and self._image is not None:
            QTimer.singleShot(0, self.fit_to_window)

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
        base = self._get_display_base()
        if base is None:
            return
        self._display_array(base)

    def _render(self):
        base = self._get_display_base()
        if base is None:
            return

        if self._zoom == 100:
            self._render_sync()
            return

        zoom_factor = self._zoom / 100.0
        h, w = base.shape[:2]
        new_h = max(1, int(round(h * zoom_factor)))
        new_w = max(1, int(round(w * zoom_factor)))
        _img = base.copy()
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

        requested_zoom = self._zoom
        self._zoom_worker = ZoomWorker(_zoom_fn, parent=self)
        self._zoom_worker.finished.connect(
            lambda result, zoom=requested_zoom: self._display_array(result)
            if zoom == self._zoom else None
        )
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
        self._img_label.setPixmap(pixmap)
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setFixedSize(pixmap.size())
        self._img_label.updateGeometry()
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
            self._render()
            self._before_badge.show()
            self._before_badge.raise_()
        else:
            self._showing_before = False
            self._after_btn.setChecked(True)
            self._before_btn.setChecked(False)
            self._before_badge.hide()
            if self._image is not None:
                self._render()

    def _set_zoom(self, zoom_percent: int, fit_mode: bool = False):
        self._fit_to_window = fit_mode
        self._zoom = max(self._MIN_ZOOM, min(int(zoom_percent), self._MAX_ZOOM))
        self._render()

    def _compute_fit_zoom(self, image: np.ndarray) -> int:
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return 100

        viewport = self._scroll.viewport().size()
        view_w = max(1, viewport.width() - 2)
        view_h = max(1, viewport.height() - 2)

        scale = min(view_w / w, view_h / h)
        zoom = int(scale * 100)
        return max(self._MIN_ZOOM, min(zoom, self._MAX_ZOOM))

    def _get_display_base(self) -> np.ndarray | None:
        if self._showing_before and self._before_image is not None:
            return self._before_image
        return self._display_image if self._display_image is not None else self._image

    def _viewport_to_image(self, point: QPoint) -> tuple:
        if self._image is None:
            return 0, 0
        label_pos = self._img_label.mapFrom(self._scroll.viewport(), point)
        zoom_factor = self._zoom / 100.0
        ix = int(label_pos.x() / zoom_factor)
        iy = int(label_pos.y() / zoom_factor)
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

