import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel,
    QPushButton, QSizePolicy, QRubberBand,
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect, QSize, QEvent
from PyQt6.QtGui import (
    QPixmap, QPainter, QColor, QFont, QKeySequence, QShortcut,
)

from utils.image_utils import to_qpixmap
from utils.error_handler import wrap_errors
from processing.interpolation.zoom import apply_zoom
from gui.workers import PipelineWorker
from gui.theme import get as _get_theme


class _ArrayState:
    """Minimal state adapter so PipelineWorker can retrieve an image."""
    def __init__(self, img: np.ndarray):
        self._img = img

    def get_base_image(self) -> np.ndarray:
        return self._img


class ImageViewer(QWidget):
    roi_selected = pyqtSignal(int, int, int, int)
    interp_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._array: np.ndarray | None = None
        self._before_array: np.ndarray | None = None
        self._showing_before = False
        self._zoom = 1.0
        self._interp = 'nearest'
        self._worker: PipelineWorker | None = None
        self._rubber_band: QRubberBand | None = None
        self._drag_origin = QPoint()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        bar = QHBoxLayout()
        bar.setContentsMargins(4, 4, 4, 4)
        bar.setSpacing(6)

        self._before_btn = QPushButton("BEFORE")
        self._before_btn.setCheckable(True)
        self._before_btn.setFixedHeight(24)
        self._before_btn.clicked.connect(self._on_before_toggled)

        self._interp_btn = QPushButton("NN")
        self._interp_btn.setFixedHeight(24)
        self._interp_btn.clicked.connect(self._toggle_interp)

        bar.addWidget(self._before_btn)
        bar.addWidget(self._interp_btn)
        bar.addStretch()
        layout.addLayout(bar)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._label.setScaledContents(False)
        self._scroll.setWidget(self._label)
        layout.addWidget(self._scroll)

        self._label.installEventFilter(self)
        self._scroll.viewport().installEventFilter(self)

        sc = QShortcut(QKeySequence("B"), self)
        sc.activated.connect(self._toggle_before)

        self._apply_theme()

    def _apply_theme(self):
        p = _get_theme()
        self._scroll.setStyleSheet(
            f"QScrollArea {{ background: {p['BG']}; border: none; }}"
        )
        self._label.setStyleSheet(f"QLabel {{ background: {p['BG']}; }}")
        btn_ss = (
            f"QPushButton {{ background: {p['INPUT']}; color: {p['MUTED']}; "
            f"border: 1px solid {p['BORDER2']}; border-radius: 2px; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; font-size: 9px; "
            f"font-weight: bold; letter-spacing: 1px; padding: 2px 8px; }}"
            f"QPushButton:hover {{ color: {p['TEXT']}; }}"
            f"QPushButton:checked {{ background: {p['PANEL2']}; color: {p['ACCENT']}; "
            f"border-color: {p['ACCENT_DIM']}; }}"
        )
        self._before_btn.setStyleSheet(btn_ss)
        self._interp_btn.setStyleSheet(btn_ss)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_image(self, array: np.ndarray):
        self._array = array
        self._showing_before = False
        self._before_btn.setChecked(False)
        self._redraw()

    def store_before(self, image: np.ndarray):
        self._before_array = image.copy()

    def theme_changed(self, palette: dict):
        self._apply_theme()
        self._redraw()

    # ------------------------------------------------------------------
    # Internal drawing
    # ------------------------------------------------------------------

    def _redraw(self):
        if self._array is None:
            return
        display = (
            self._before_array
            if (self._showing_before and self._before_array is not None)
            else self._array
        )
        pix = to_qpixmap(display)
        if self._zoom != 1.0:
            w = max(1, int(pix.width() * self._zoom))
            h = max(1, int(pix.height() * self._zoom))
            mode = (
                Qt.TransformationMode.FastTransformation
                if self._interp == 'nearest'
                else Qt.TransformationMode.SmoothTransformation
            )
            pix = pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, mode)
        if self._showing_before and self._before_array is not None:
            pix = self._draw_before_badge(pix)
        self._label.setPixmap(pix)
        self._label.resize(pix.size())

    def _draw_before_badge(self, pix: QPixmap) -> QPixmap:
        result = QPixmap(pix)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        badge = QRect(8, 8, 72, 20)
        painter.fillRect(badge, QColor("#f5c518"))
        painter.setPen(QColor("#0d1002"))
        font = QFont("JetBrains Mono", 8)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, "BEFORE")
        painter.end()
        return result

    # ------------------------------------------------------------------
    # Toggle handlers
    # ------------------------------------------------------------------

    def _toggle_before(self):
        checked = not self._before_btn.isChecked()
        self._before_btn.setChecked(checked)
        self._on_before_toggled(checked)

    def _on_before_toggled(self, checked: bool):
        self._showing_before = checked
        self._redraw()

    def _toggle_interp(self):
        self._interp = 'bilinear' if self._interp == 'nearest' else 'nearest'
        self._interp_btn.setText("BL" if self._interp == 'bilinear' else "NN")
        self.interp_changed.emit(self._interp)
        self._redraw()

    def set_interp(self, mode: str):
        if mode not in ('nearest', 'bilinear'):
            return
        self._interp = mode
        self._interp_btn.setText("BL" if mode == 'bilinear' else "NN")
        self._redraw()

    # ------------------------------------------------------------------
    # Event filter: wheel zoom + rubber-band ROI
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is self._label:
            t = event.type()
            if t == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._drag_origin = event.pos()
                self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self._label)
                self._rubber_band.setGeometry(QRect(self._drag_origin, QSize()))
                self._rubber_band.show()
                return True
            if t == QEvent.Type.MouseMove and self._rubber_band is not None:
                self._rubber_band.setGeometry(
                    QRect(self._drag_origin, event.pos()).normalized()
                )
                return True
            if t == QEvent.Type.MouseButtonRelease and self._rubber_band is not None:
                rect = QRect(self._drag_origin, event.pos()).normalized()
                self._rubber_band.hide()
                self._rubber_band = None
                self._emit_roi(rect)
                return True
        if obj is self._scroll.viewport():
            if event.type() == QEvent.Type.Wheel:
                self._on_wheel_zoom(event.angleDelta().y())
                return True
        return super().eventFilter(obj, event)

    def _emit_roi(self, widget_rect: QRect):
        if self._array is None or widget_rect.width() < 2 or widget_rect.height() < 2:
            return
        pix = self._label.pixmap()
        if pix is None:
            return
        lw, lh = self._label.width(), self._label.height()
        pw, ph = pix.width(), pix.height()
        ox = (lw - pw) // 2
        oy = (lh - ph) // 2
        x = max(0, int((widget_rect.x() - ox) / self._zoom))
        y = max(0, int((widget_rect.y() - oy) / self._zoom))
        w = max(1, int(widget_rect.width() / self._zoom))
        h = max(1, int(widget_rect.height() / self._zoom))
        self.roi_selected.emit(x, y, w, h)

    # ------------------------------------------------------------------
    # Wheel zoom via PipelineWorker
    # ------------------------------------------------------------------

    def _on_wheel_zoom(self, delta: int):
        if self._array is None:
            return
        factor = 1.1 if delta > 0 else 0.9
        self._zoom = max(0.1, min(8.0, self._zoom * factor))
        self._redraw()  # immediate Qt-scaled preview
        if self._worker and self._worker.isRunning():
            return
        state = _ArrayState(self._array)
        self._worker = PipelineWorker(
            apply_zoom, "zoom", state,
            zoom_factor=self._zoom, mode=self._interp,
        )
        self._worker.finished.connect(self._on_zoom_done)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_zoom_done(self, _op_name: str, result: np.ndarray):
        pix = to_qpixmap(result)
        if self._showing_before and self._before_array is not None:
            pix = self._draw_before_badge(pix)
        self._label.setPixmap(pix)
        self._label.resize(pix.size())

    def _on_worker_error(self, _msg: str):
        pass
