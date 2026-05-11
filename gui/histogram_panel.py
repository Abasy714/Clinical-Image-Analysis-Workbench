import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QSizePolicy, QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from gui.theme import get as _get_theme
from gui.styles import COMBO_SS, SPINBOX_SS, APPLY_BTN_SS, HEADER_SS, FIELD_SS
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
        for b in ["16", "32", "64", "128", "256"]:
            self._bins_combo.addItem(b)
        self._bins_combo.setCurrentText("256")
        self._bins_combo.currentIndexChanged.connect(self.refresh)
        bins_row.addWidget(self._bins_combo)
        layout.addLayout(bins_row)

        layout.addWidget(self._header("GLOBAL EQUALIZATION"))
        eq_btn = QPushButton("APPLY EQUALIZATION")
        eq_btn.setStyleSheet(APPLY_BTN_SS)
        eq_btn.clicked.connect(self._apply_equalization)
        layout.addWidget(eq_btn)

        layout.addWidget(self._header("LOCAL EQUALIZATION"))

        blk_row = QHBoxLayout()
        blk_row.addWidget(self._lbl("BLOCK SIZE"))
        self._block_spin = QSpinBox()
        self._block_spin.setStyleSheet(SPINBOX_SS)
        self._block_spin.setRange(8, 128)
        self._block_spin.setSingleStep(8)
        self._block_spin.setValue(32)
        blk_row.addWidget(self._block_spin)
        layout.addLayout(blk_row)

        clip_row = QHBoxLayout()
        clip_row.addWidget(self._lbl("CLIP LIMIT"))
        self._clip_spin = QDoubleSpinBox()
        self._clip_spin.setStyleSheet(SPINBOX_SS)
        self._clip_spin.setRange(1.0, 10.0)
        self._clip_spin.setSingleStep(0.5)
        self._clip_spin.setValue(3.0)
        clip_row.addWidget(self._clip_spin)
        layout.addLayout(clip_row)

        local_btn = QPushButton("APPLY LOCAL EQ")
        local_btn.setStyleSheet(APPLY_BTN_SS)
        local_btn.clicked.connect(self._apply_local_eq)
        layout.addWidget(local_btn)
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

        if img.ndim == 3:
            gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1]
                    + 0.114 * img[:, :, 2]).astype(np.uint8)
        else:
            gray = img

        if bins == 256:
            from processing.histogram.histogram_utils import compute_histogram
            hist = compute_histogram(gray)
            xs = range(256)
            self._ax.bar(xs, hist, width=1, color=p['ACCENT'], alpha=0.75, linewidth=0)
        else:
            hist, bin_edges = np.histogram(gray.ravel(), bins=bins, range=(0, 255))
            width = (bin_edges[1] - bin_edges[0]) * 0.9
            self._ax.bar(bin_edges[:-1], hist, width=width, color=p['ACCENT'],
                         alpha=0.75, linewidth=0)

        self._ax.set_xlim(0, 255)
        self._ax.set_yticks([])
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _start_worker(self, fn, op_name: str, **kwargs):
        if self._state is None:
            return
        if self._worker and self._worker.isRunning():
            return
        self._worker = PipelineWorker(fn, op_name, self._state, **kwargs)
        self._worker.finished.connect(self.operation_applied)
        self._worker.error.connect(self.error_occurred)
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

        self._start_worker(_equalize, "Hist Equalize")

    def _apply_local_eq(self):
        from processing.histogram.local_equalization import local_histogram_equalization
        self._start_worker(
            local_histogram_equalization, "Local Hist EQ",
            block_size=self._block_spin.value(),
            clip_limit=self._clip_spin.value(),
        )
