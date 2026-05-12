import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QComboBox, QButtonGroup,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from gui.theme import get as _get_theme
from gui.styles import btn_style, operation_btn_style, COMBO_SS, APPLY_BTN_SS, HEADER_SS, FIELD_SS
from gui.workers import PipelineWorker

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _MPL = True
except ImportError:
    _MPL = False


class HistogramPanel(QWidget):
    operation_applied = pyqtSignal(str, np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._worker: PipelineWorker | None = None
        self._roi: tuple[int, int, int, int] | None = None
        self._build_ui()

    def set_state(self, state):
        self._state = state
        self.refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        p = _get_theme()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(self._header("HISTOGRAM"))

        if _MPL:
            self._fig = Figure(figsize=(3, 1.6), dpi=80,
                               facecolor=p['BG'], tight_layout=True)
            self._ax = self._fig.add_subplot(111, facecolor=p['PANEL'])
            self._canvas = FigureCanvas(self._fig)
            self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self._canvas.setFixedHeight(140)
            layout.addWidget(self._canvas)
            self._ax.tick_params(colors=p['MUTED'], labelsize=7)
            for spine in self._ax.spines.values():
                spine.set_color(p['BORDER'])
        else:
            no_mpl = QLabel("matplotlib not available")
            no_mpl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_mpl.setStyleSheet(f"color: {p['MUTED']}; font-size: 9px;")
            layout.addWidget(no_mpl)

        bins_row = QHBoxLayout()
        bins_row.addWidget(self._lbl("BINS"))
        self._bins_combo = QComboBox()
        self._bins_combo.setStyleSheet(COMBO_SS)
        for b in ["4", "8", "16", "32", "64", "128", "256"]:
            self._bins_combo.addItem(b)
        self._bins_combo.setCurrentText("256")
        self._bins_combo.currentIndexChanged.connect(self.refresh)
        bins_row.addWidget(self._bins_combo)
        layout.addLayout(bins_row)

        layout.addWidget(self._header("GLOBAL EQUALIZATION"))
        self._eq_btn = QPushButton("APPLY EQUALIZATION")
        self._eq_btn.setStyleSheet(operation_btn_style())
        self._eq_btn.clicked.connect(self._apply_equalization)
        layout.addWidget(self._eq_btn)

        layout.addWidget(self._header("LOCAL EQUALIZATION"))

        blk_row = QHBoxLayout()
        blk_row.addWidget(self._lbl("BLOCK SIZE"))
        self._block_group = QButtonGroup(self)
        self._block_group.setExclusive(True)
        for bs in [16, 32, 64, 128]:
            btn = QPushButton(str(bs))
            btn.setCheckable(True)
            btn.setStyleSheet(btn_style('default'))
            if bs == 32:
                btn.setChecked(True)
            self._block_group.addButton(btn, bs)
            blk_row.addWidget(btn)
        layout.addLayout(blk_row)

        self._local_btn = QPushButton("APPLY LOCAL EQ")
        self._local_btn.setStyleSheet(operation_btn_style())
        self._local_btn.clicked.connect(self._apply_local_eq)
        layout.addWidget(self._local_btn)
        layout.addStretch()

    def _lbl(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(FIELD_SS)
        return lbl

    def _header(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(HEADER_SS)
        return lbl

    # ------------------------------------------------------------------
    # Histogram display
    # ------------------------------------------------------------------

    def update_histogram(self, image: np.ndarray):
        if not _MPL or image is None:
            return
        self._draw_histogram(image)

    def set_roi(self, x: int, y: int, w: int, h: int):
        self._roi = (x, y, w, h)
        self.refresh()

    def clear_roi(self):
        self._roi = None
        self.refresh()

    def refresh(self):
        if not _MPL:
            return
        if self._state is None:
            self._draw_histogram(None)
            return
        self._draw_histogram(self._state.current())

    def _draw_histogram(self, img):
        p = _get_theme()
        self._ax.cla()
        self._ax.set_facecolor(p['PANEL'])
        self._ax.tick_params(colors=p['MUTED'], labelsize=7)
        for spine in self._ax.spines.values():
            spine.set_color(p['BORDER'])

        if img is None:
            self._canvas.draw()
            return

        bins = int(self._bins_combo.currentText()) if hasattr(self, '_bins_combo') else 256

        from processing.histogram.histogram_utils import compute_histogram
        from utils.image_utils import normalize_to_uint8, to_grayscale, validate_grayscale

        gray = normalize_to_uint8(to_grayscale(img))
        validate_grayscale(gray)

        hist = compute_histogram(gray, bins=bins)
        bin_width = 256.0 / bins
        xs = np.arange(bins) * bin_width
        self._ax.bar(
            xs, hist, width=bin_width * 0.9,
            color=p['ACCENT'], alpha=0.72, linewidth=0, align='edge',
        )

        roi = self._valid_roi(gray)
        if roi is not None:
            x, y, w, h = roi
            roi_gray = gray[y:y + h, x:x + w]
            roi_hist = compute_histogram(roi_gray, bins=bins)
            self._ax.step(
                xs + bin_width * 0.5, roi_hist,
                where='mid', color='#ff5c7a', linewidth=1.4, alpha=0.95,
            )

        self._ax.set_xlim(0, 255)
        self._ax.set_yticks([])
        self._canvas.draw()

    def _valid_roi(self, gray: np.ndarray):
        if self._roi is None:
            return None
        x, y, w, h = self._roi
        height, width = gray.shape
        x = max(0, min(int(x), width))
        y = max(0, min(int(y), height))
        w = max(0, min(int(w), width - x))
        h = max(0, min(int(h), height - y))
        if w <= 0 or h <= 0:
            self._roi = None
            return None
        self._roi = (x, y, w, h)
        return self._roi

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _start_worker(self, fn, op_name: str, btn=None, btn_label=None, **kwargs):
        if self._state is None:
            return
        if self._worker and self._worker.isRunning():
            return
        if btn is not None:
            btn.setEnabled(False)
            btn.setText("⟳ Processing...")
        self._worker = PipelineWorker(fn, op_name, self._state, **kwargs)
        self._worker.finished.connect(self.operation_applied)
        self._worker.error.connect(self.error_occurred)
        if btn is not None:
            orig = btn_label or op_name
            self._worker.finished.connect(lambda *_: (btn.setEnabled(True), btn.setText(orig)))
            self._worker.error.connect(lambda *_: (btn.setEnabled(True), btn.setText(orig)))
        self._worker.start()

    def _apply_equalization(self):
        def _equalize(image):
            from processing.histogram.histogram_utils import compute_histogram, compute_cdf
            if image.ndim == 3:
                gray = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1]
                        + 0.114 * image[:, :, 2]).astype(np.uint8)
            else:
                gray = image
            hist = compute_histogram(gray)
            cdf = compute_cdf(hist)
            lut = (cdf * 255).astype(np.uint8)
            return lut[gray]

        self._start_worker(_equalize, "Hist Equalize", btn=self._eq_btn, btn_label="APPLY EQUALIZATION")

    def _apply_local_eq(self):
        from processing.histogram.local_equalization import local_histogram_equalization
        block_size = self._block_group.checkedId()
        if block_size <= 0:
            block_size = 32
        self._start_worker(
            local_histogram_equalization, "Local Hist EQ",
            btn=self._local_btn, btn_label="APPLY LOCAL EQ",
            block_size=block_size,
            clip_limit=2.0,
        )
