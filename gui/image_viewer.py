import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QLabel,
    QPushButton, QSizePolicy, QRubberBand, QGraphicsOpacityEffect,
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QPoint, QRect, QSize, QEvent,
    QPropertyAnimation, QAbstractAnimation,
)
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont, QKeySequence, QShortcut

from utils.image_utils import to_qpixmap
from processing.interpolation.zoom import apply_zoom
from gui.workers import PipelineWorker
from gui.theme import get as _get_theme
from gui.widgets import ProgressStrip, RoiOverlay


class _ArrayState:
    def __init__(self, img: np.ndarray):
        self._img = img

    def get_base_image(self) -> np.ndarray:
        return self._img


class ImageViewer(QWidget):
    roi_selected   = pyqtSignal(int, int, int, int)
    roi_cleared    = pyqtSignal()
    interp_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_image: np.ndarray | None = None
        self._before_image:  np.ndarray | None = None
        self._view_mode  = 'after'
        self._freq_mode  = False
        self._zoom       = 1.0
        self._interp     = 'nearest'
        self._state      = None
        self._workers: list = []
        self._worker: PipelineWorker | None = None
        self._rubber_band: QRubberBand | None = None
        self._drag_origin = QPoint()
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_state(self, state):
        self._state = state

    def set_image(self, array: np.ndarray):
        self._current_image = array.copy()
        self._view_mode = 'after'
        self._update_view_buttons()
        self._refresh_domain_display()
        self._update_zoom_label()
        self._update_info_strip()

    def store_before(self, image: np.ndarray):
        self._before_image = image.copy()

    def clear_roi(self, emit_signal: bool = True):
        if self._rubber_band is not None:
            self._rubber_band.hide()
            self._rubber_band = None
        self._roi_overlay.set_rect(None)
        if emit_signal:
            self.roi_cleared.emit()

    def set_interp(self, mode: str):
        if mode not in ('nearest', 'bilinear'):
            return
        self._interp = mode
        self._update_interp_btn()
        self._refresh_domain_display()

    def theme_changed(self, palette: dict):
        self._apply_theme()
        self._on_domain_toggled(self._domain_btn.isChecked())

    # ------------------------------------------------------------------
    # Keep-alive
    # ------------------------------------------------------------------

    def _keep_alive(self, worker: PipelineWorker):
        self._workers.append(worker)
        worker.finished.connect(
            lambda *_: self._workers.remove(worker) if worker in self._workers else None
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- toolbar ----
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 4, 4, 4)
        bar.setSpacing(4)

        self._zoom_minus = QPushButton("−")
        self._zoom_minus.setFixedSize(24, 24)
        self._zoom_minus.clicked.connect(self._zoom_out)

        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(40)
        self._zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._zoom_plus = QPushButton("+")
        self._zoom_plus.setFixedSize(24, 24)
        self._zoom_plus.clicked.connect(self._zoom_in)

        self._fit_btn = QPushButton("Fit")
        self._fit_btn.setFixedHeight(24)
        self._fit_btn.clicked.connect(self._fit_zoom)

        bar.addWidget(self._zoom_minus)
        bar.addWidget(self._zoom_label)
        bar.addWidget(self._zoom_plus)
        bar.addWidget(self._fit_btn)
        bar.addWidget(self._make_sep())

        self._interp_btn = QPushButton("NN")
        self._interp_btn.setCheckable(True)
        self._interp_btn.setChecked(False)
        self._interp_btn.setFixedWidth(36)
        self._interp_btn.clicked.connect(self._on_interp_toggled)
        bar.addWidget(self._interp_btn)
        bar.addWidget(self._make_sep())

        self._before_btn = QPushButton("BEFORE")
        self._before_btn.setFixedHeight(26)
        self._before_btn.clicked.connect(self._on_before_clicked)

        self._after_btn = QPushButton("AFTER")
        self._after_btn.setFixedHeight(26)
        self._after_btn.clicked.connect(self._on_after_clicked)

        bar.addWidget(self._before_btn)
        bar.addWidget(self._after_btn)
        bar.addWidget(self._make_sep())

        self._domain_btn = QPushButton("SPATIAL")
        self._domain_btn.setCheckable(True)
        self._domain_btn.setChecked(False)
        self._domain_btn.setFixedSize(64, 26)
        self._domain_btn.clicked.connect(self._on_domain_toggled)
        bar.addWidget(self._domain_btn)

        bar.addStretch()
        layout.addLayout(bar)

        # ---- progress strip (2px, animated) ----
        self.progress_strip = ProgressStrip(self)
        layout.addWidget(self.progress_strip)

        # ---- image scroll area ----
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
        layout.addWidget(self._scroll, stretch=1)

        # marching-ants ROI overlay (child of label, transparent to mouse)
        self._roi_overlay = RoiOverlay(self._label)
        self._roi_overlay.resize(self._label.size())

        # crossfade opacity effect on the image label
        self._opacity_eff = QGraphicsOpacityEffect(self._label)
        self._opacity_eff.setOpacity(1.0)
        self._label.setGraphicsEffect(self._opacity_eff)

        # ---- bottom info strip (24px) ----
        info_strip = QWidget()
        info_strip.setFixedHeight(24)
        info_lyt = QHBoxLayout(info_strip)
        info_lyt.setContentsMargins(8, 0, 8, 0)
        info_lyt.setSpacing(0)

        self._coord_lbl = QLabel("")
        self._coord_lbl.setFixedWidth(180)
        info_lyt.addWidget(self._coord_lbl)

        info_lyt.addStretch()

        self._before_badge_lbl = QLabel("BEFORE")
        self._before_badge_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._before_badge_lbl.setFixedWidth(60)
        self._before_badge_lbl.hide()
        info_lyt.addWidget(self._before_badge_lbl)

        info_lyt.addStretch()

        self._zoom_dim_lbl = QLabel("")
        self._zoom_dim_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        info_lyt.addWidget(self._zoom_dim_lbl)

        self._info_strip = info_strip
        layout.addWidget(info_strip)

        self._label.installEventFilter(self)
        self._scroll.viewport().installEventFilter(self)

        sc = QShortcut(QKeySequence("B"), self)
        sc.activated.connect(self._toggle_before)

        self._apply_theme()

    def _make_sep(self) -> QLabel:
        p = _get_theme()
        lbl = QLabel("|")
        lbl.setStyleSheet(
            f"QLabel {{ color: {p['BORDER2']}; font-size: 10px; padding: 0 2px; }}"
        )
        return lbl

    def _apply_theme(self):
        p = _get_theme()
        self._scroll.setStyleSheet(
            f"QScrollArea {{ background: {p['BG']}; border: none; }}"
        )
        self._label.setStyleSheet(f"QLabel {{ background: {p['BG']}; }}")
        self._zoom_label.setStyleSheet(
            f"QLabel {{ color: {p['MUTED']}; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; font-size: 9px; }}"
        )
        icon_ss = (
            f"QPushButton {{ background: {p['INPUT']}; color: {p['TEXT']};"
            f" border: 1px solid {p['BORDER2']}; border-radius: 2px;"
            f" font-family: 'JetBrains Mono', Consolas, monospace; font-size: 11px;"
            f" font-weight: bold; }}"
            f"QPushButton:hover {{ background: {p['PANEL2']}; }}"
        )
        for btn in (self._zoom_minus, self._zoom_plus, self._fit_btn):
            btn.setStyleSheet(icon_ss)
        self._coord_lbl.setStyleSheet(
            f"QLabel {{ color: {p['MUTED']}; background: transparent; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; font-size: 9px; }}"
        )
        self._before_badge_lbl.setStyleSheet(
            f"QLabel {{ color: {p['DARK']}; background: {p['AMBER']}; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; font-size: 8px; "
            f"font-weight: bold; border-radius: 2px; padding: 1px 4px; }}"
        )
        self._zoom_dim_lbl.setStyleSheet(
            f"QLabel {{ color: {p['MUTED']}; background: transparent; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; font-size: 9px; }}"
        )
        self._info_strip.setStyleSheet(f"background: {p['PANEL']};")
        self._update_interp_btn()
        self._update_view_buttons()
        self._on_domain_toggled(self._domain_btn.isChecked())

    def _update_interp_btn(self):
        p = _get_theme()
        checked = self._interp == 'bilinear'
        self._interp_btn.setChecked(checked)
        self._interp_btn.setText("BL" if checked else "NN")
        if checked:
            self._interp_btn.setStyleSheet(
                f"QPushButton {{ background: {p['ACCENT']}; color: {p['DARK']};"
                f" border: 1px solid {p['ACCENT']}; border-radius: 2px;"
                f" font-family: 'JetBrains Mono', Consolas, monospace; font-size: 9px;"
                f" font-weight: bold; }}"
            )
        else:
            self._interp_btn.setStyleSheet(
                f"QPushButton {{ background: transparent; color: {p['ACCENT']};"
                f" border: 1px solid {p['ACCENT']}; border-radius: 2px;"
                f" font-family: 'JetBrains Mono', Consolas, monospace; font-size: 9px;"
                f" font-weight: bold; }}"
                f"QPushButton:hover {{ background: {p['PANEL2']}; }}"
            )

    def _update_view_buttons(self):
        p = _get_theme()
        active_ss = (
            f"QPushButton {{ background: {p['ACCENT']}; color: {p['DARK']};"
            f" border: 1px solid {p['ACCENT']}; border-radius: 2px;"
            f" font-family: 'JetBrains Mono', Consolas, monospace; font-size: 9px;"
            f" font-weight: bold; padding: 2px 8px; }}"
        )
        inactive_ss = (
            f"QPushButton {{ background: transparent; color: {p['ACCENT']};"
            f" border: 1px solid {p['ACCENT']}; border-radius: 2px;"
            f" font-family: 'JetBrains Mono', Consolas, monospace; font-size: 9px;"
            f" font-weight: bold; padding: 2px 8px; }}"
            f"QPushButton:hover {{ background: {p['PANEL2']}; }}"
        )
        if self._view_mode == 'before':
            self._before_btn.setStyleSheet(active_ss)
            self._after_btn.setStyleSheet(inactive_ss)
            self._before_badge_lbl.show()
        else:
            self._before_btn.setStyleSheet(inactive_ss)
            self._after_btn.setStyleSheet(active_ss)
            self._before_badge_lbl.hide()

    def _update_zoom_label(self):
        self._zoom_label.setText(f"{int(self._zoom * 100)}%")

    def _update_info_strip(self):
        if self._current_image is None:
            self._zoom_dim_lbl.setText("")
            return
        h, w = self._current_image.shape[:2]
        pct = int(self._zoom * 100)
        self._zoom_dim_lbl.setText(f"{pct}%  {w}×{h}")

    # ------------------------------------------------------------------
    # Domain display
    # ------------------------------------------------------------------

    def _refresh_domain_display(self, crossfade: bool = False):
        if self._current_image is None:
            return
        source = (
            self._before_image
            if self._view_mode == 'before' and self._before_image is not None
            else self._current_image
        )
        if self._freq_mode:
            def _spec(image):
                from processing.frequency.spectrum import compute_spectrum, spectrum_to_display
                from utils.image_utils import to_grayscale
                gray = to_grayscale(image) if image.ndim == 3 else image
                shifted_fft, _, _ = compute_spectrum(gray)
                return spectrum_to_display(shifted_fft)

            state = _ArrayState(source)
            w = PipelineWorker(fn=_spec, op_name='_domain_preview', state=state)
            w.finished.connect(
                lambda _n, arr: self._render_array(arr, crossfade=crossfade)
            )
            w.error.connect(lambda _e: None)
            w.start()
            self._keep_alive(w)
        else:
            self._render_array(source, crossfade=crossfade)

    def _render_array(self, array: np.ndarray, crossfade: bool = False):
        pix = to_qpixmap(array)
        if self._zoom != 1.0:
            w = max(1, int(pix.width() * self._zoom))
            h = max(1, int(pix.height() * self._zoom))
            mode = (
                Qt.TransformationMode.SmoothTransformation
                if self._interp == 'bilinear'
                else Qt.TransformationMode.FastTransformation
            )
            pix = pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, mode)
        if crossfade:
            self._crossfade_to(pix)
        else:
            self._label.setPixmap(pix)
            self._label.resize(pix.size())
            self._roi_overlay.resize(pix.size())

    def _crossfade_to(self, pix: QPixmap):
        fade_out = QPropertyAnimation(self._opacity_eff, b"opacity", self)
        fade_out.setDuration(150)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)

        def _swap():
            self._label.setPixmap(pix)
            self._label.resize(pix.size())
            self._roi_overlay.resize(pix.size())
            fade_in = QPropertyAnimation(self._opacity_eff, b"opacity", self)
            fade_in.setDuration(150)
            fade_in.setStartValue(0.0)
            fade_in.setEndValue(1.0)
            fade_in.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

        fade_out.finished.connect(_swap)
        fade_out.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def _redraw(self):
        if self._current_image is None:
            return
        source = (
            self._before_image
            if self._view_mode == 'before' and self._before_image is not None
            else self._current_image
        )
        self._render_array(source)

    # ------------------------------------------------------------------
    # Toggle handlers
    # ------------------------------------------------------------------

    def _on_before_clicked(self):
        self._view_mode = 'before'
        self._update_view_buttons()
        self._refresh_domain_display(crossfade=True)

    def _on_after_clicked(self):
        self._view_mode = 'after'
        self._update_view_buttons()
        self._refresh_domain_display(crossfade=True)

    def _toggle_before(self):
        if self._view_mode == 'before':
            self._on_after_clicked()
        else:
            self._on_before_clicked()

    def _on_interp_toggled(self, checked: bool):
        self._interp = 'bilinear' if checked else 'nearest'
        self._update_interp_btn()
        self.interp_changed.emit(self._interp)
        self._refresh_domain_display()

    def _on_domain_toggled(self, checked: bool):
        self._freq_mode = checked
        p = _get_theme()
        if checked:
            self._domain_btn.setText('FREQ')
            self._domain_btn.setStyleSheet(
                f"QPushButton {{ background: {p['ACCENT']}; color: {p['DARK']};"
                f" border: 1px solid {p['ACCENT']}; border-radius: 3px;"
                f" font-family: 'JetBrains Mono', Consolas, monospace; font-size: 10px;"
                f" font-weight: bold; padding: 2px 4px; }}"
            )
        else:
            self._domain_btn.setText('SPATIAL')
            self._domain_btn.setStyleSheet(
                f"QPushButton {{ background: transparent; color: {p['ACCENT']};"
                f" border: 1px solid {p['ACCENT']}; border-radius: 3px;"
                f" font-family: 'JetBrains Mono', Consolas, monospace; font-size: 10px;"
                f" font-weight: bold; padding: 2px 4px; }}"
                f"QPushButton:hover {{ background: {p['PANEL2']}; }}"
            )
        self._refresh_domain_display()

    # ------------------------------------------------------------------
    # Zoom controls
    # ------------------------------------------------------------------

    def _zoom_in(self):
        self._zoom = min(8.0, self._zoom * 1.2)
        self._update_zoom_label()
        self._update_info_strip()
        self._redraw()

    def _zoom_out(self):
        self._zoom = max(0.05, self._zoom / 1.2)
        self._update_zoom_label()
        self._update_info_strip()
        self._redraw()

    def _fit_zoom(self):
        if self._current_image is None:
            return
        h, w = self._current_image.shape[:2]
        vp = self._scroll.viewport()
        vw, vh = max(1, vp.width()), max(1, vp.height())
        self._zoom = max(0.05, min(8.0, min(vw / w, vh / h)))
        self._update_zoom_label()
        self._update_info_strip()
        self._redraw()

    # ------------------------------------------------------------------
    # Mouse coord tracking
    # ------------------------------------------------------------------

    def _update_coord_display(self, label_pos: QPoint):
        if self._current_image is None:
            self._coord_lbl.setText("")
            return
        pix = self._label.pixmap()
        if pix is None:
            self._coord_lbl.setText("")
            return
        lw, lh = self._label.width(), self._label.height()
        pw, ph = pix.width(), pix.height()
        ox = (lw - pw) // 2
        oy = (lh - ph) // 2
        img_x = int((label_pos.x() - ox) / self._zoom)
        img_y = int((label_pos.y() - oy) / self._zoom)
        h, w = self._current_image.shape[:2]
        if 0 <= img_x < w and 0 <= img_y < h:
            img = self._current_image
            if img.ndim == 3:
                intensity = int(
                    0.299 * img[img_y, img_x, 0]
                    + 0.587 * img[img_y, img_x, 1]
                    + 0.114 * img[img_y, img_x, 2]
                )
            else:
                intensity = int(img[img_y, img_x])
            self._coord_lbl.setText(f"X: {img_x}  Y: {img_y}  I: {intensity}")
        else:
            self._coord_lbl.setText("")

    # ------------------------------------------------------------------
    # Event filter: wheel zoom + rubber-band ROI + coord tracking
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is self._label:
            t = event.type()
            if t == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._drag_origin = event.pos()
                self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self._label)
                self._rubber_band.setGeometry(QRect(self._drag_origin, QSize()))
                self._rubber_band.show()
                self._roi_overlay.set_rect(None)
                return True
            if t == QEvent.Type.MouseMove:
                self._update_coord_display(event.pos())
                if self._rubber_band is not None:
                    self._rubber_band.setGeometry(
                        QRect(self._drag_origin, event.pos()).normalized()
                    )
                return True
            if t == QEvent.Type.MouseButtonRelease and self._rubber_band is not None:
                rect = QRect(self._drag_origin, event.pos()).normalized()
                self._rubber_band.hide()
                self._rubber_band = None
                if rect.width() >= 4 and rect.height() >= 4:
                    self._roi_overlay.set_rect(rect)
                self._emit_roi(rect)
                return True
            if t == QEvent.Type.Leave:
                self._coord_lbl.setText("")
                return False
        if obj is self._scroll.viewport():
            if event.type() == QEvent.Type.Wheel:
                self._on_wheel_zoom(event.angleDelta().y())
                return True
        return super().eventFilter(obj, event)

    def _emit_roi(self, widget_rect: QRect):
        if self._current_image is None or widget_rect.width() < 2 or widget_rect.height() < 2:
            return
        pix = self._label.pixmap()
        if pix is None:
            return
        height, width = self._current_image.shape[:2]
        lw, lh = self._label.width(), self._label.height()
        pw, ph = pix.width(), pix.height()
        ox = (lw - pw) // 2
        oy = (lh - ph) // 2
        x0 = int((widget_rect.x() - ox) / self._zoom)
        y0 = int((widget_rect.y() - oy) / self._zoom)
        x1 = int((widget_rect.x() + widget_rect.width() - ox) / self._zoom)
        y1 = int((widget_rect.y() + widget_rect.height() - oy) / self._zoom)
        x0 = max(0, min(x0, width))
        y0 = max(0, min(y0, height))
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x = min(x0, x1)
        y = min(y0, y1)
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        if w <= 0 or h <= 0:
            return
        self.roi_selected.emit(x, y, w, h)

    # ------------------------------------------------------------------
    # Wheel zoom
    # ------------------------------------------------------------------

    def _on_wheel_zoom(self, delta: int):
        if self._current_image is None:
            return
        factor = 1.1 if delta > 0 else 0.9
        self._zoom = max(0.1, min(8.0, self._zoom * factor))
        self._update_zoom_label()
        self._update_info_strip()
        if not self._freq_mode:
            self._redraw()
            if self._worker and self._worker.isRunning():
                return
            source = (
                self._before_image
                if self._view_mode == 'before' and self._before_image is not None
                else self._current_image
            )
            state = _ArrayState(source)
            self._worker = PipelineWorker(
                apply_zoom, "zoom", state,
                zoom_factor=self._zoom, mode=self._interp,
            )
            self._worker.finished.connect(self._on_zoom_done)
            self._worker.error.connect(lambda _e: None)
            self._worker.start()

    def _on_zoom_done(self, _op_name: str, result: np.ndarray):
        pix = to_qpixmap(result)
        self._label.setPixmap(pix)
        self._label.resize(pix.size())
        self._roi_overlay.resize(pix.size())
