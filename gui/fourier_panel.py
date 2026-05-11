import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QTabWidget, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from gui.theme import get as _get_theme
from gui.styles import btn_style, SPINBOX_SS, APPLY_BTN_SS, HEADER_SS, FIELD_SS
from gui.workers import PipelineWorker


# ------------------------------------------------------------------
# Minimal state adapter for canvas workers with a fixed image
# ------------------------------------------------------------------

class _StaticState:
    def __init__(self, img: np.ndarray):
        self._img = img

    def get_base_image(self) -> np.ndarray:
        return self._img


# ------------------------------------------------------------------
# Module-level worker functions (never called on main thread)
# ------------------------------------------------------------------

def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1]
                + 0.114 * image[:, :, 2]).astype(np.uint8)
    return image


def _canvas_spatial(image):
    return _to_gray(image)


def _canvas_magnitude(image):
    from processing.frequency.spectrum import compute_spectrum, spectrum_to_display
    gray = _to_gray(image)
    shifted_fft, _, _ = compute_spectrum(gray)
    return spectrum_to_display(shifted_fft)


def _canvas_phase(image):
    from processing.frequency.spectrum import compute_spectrum, phase_to_display
    gray = _to_gray(image)
    shifted_fft, _, _ = compute_spectrum(gray)
    return phase_to_display(shifted_fft)


def _notch_filter_fn(image, notch_centers, radius):
    from processing.frequency.spectrum import compute_spectrum, inverse_spectrum
    from processing.frequency.notch_filter import create_notch_filter, apply_notch_filter
    gray = _to_gray(image)
    shape = gray.shape[:2]
    shifted_fft, _, _ = compute_spectrum(gray)
    combined = np.ones(shape, dtype=np.float64)
    for (row, col) in notch_centers:
        combined *= create_notch_filter(shape, u=col, v=row, radius=radius)
    filtered = apply_notch_filter(shifted_fft, combined)
    return inverse_spectrum(filtered)


# ------------------------------------------------------------------
# FourierPanel
# ------------------------------------------------------------------

class FourierPanel(QWidget):
    fourier_applied = pyqtSignal(str, np.ndarray)
    error_occurred  = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._worker: PipelineWorker | None = None
        self._canvas_worker: PipelineWorker | None = None
        self._current_image: np.ndarray | None = None
        self._notch_centers: list[tuple[int, int]] = []
        self._notch_spectrum_data: np.ndarray | None = None
        self._notch_H: int = 0
        self._notch_W: int = 0
        self._build_ui()

    def set_state(self, state):
        self._state = state

    # ------------------------------------------------------------------
    # Public API (called by main_window)
    # ------------------------------------------------------------------

    def update_display(self, image: np.ndarray):
        self._current_image = image
        # Compute spectrum for notch canvas (display only)
        try:
            from processing.frequency.spectrum import compute_spectrum, spectrum_to_display
            gray = _to_gray(image)
            shifted_fft, _, _ = compute_spectrum(gray)
            self._notch_spectrum_data = spectrum_to_display(shifted_fft)
            self._notch_H, self._notch_W = gray.shape[:2]
        except Exception:
            pass
        self._trigger_canvas_update()
        self._redraw_notch_canvas()

    def theme_changed(self, palette: dict):
        self._update_canvas_theme(palette)
        self._trigger_canvas_update()
        self._redraw_notch_canvas()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_spectrum_tab(), "SPECTRUM")
        tabs.addTab(self._build_notch_tab(), "NOTCH")
        layout.addWidget(tabs)

    def _lbl(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(FIELD_SS)
        return lbl

    def _header(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(HEADER_SS)
        return lbl

    # ---- SPECTRUM tab ----

    def _build_spectrum_tab(self) -> QWidget:
        p = _get_theme()
        w = QWidget()
        lyt = QVBoxLayout(w)
        lyt.setContentsMargins(8, 8, 8, 8)
        lyt.setSpacing(6)

        tog_row = QHBoxLayout()
        tog_row.setSpacing(4)

        self._freq_btn = QPushButton("Spatial Domain")
        self._freq_btn.setCheckable(True)
        self._freq_btn.setChecked(False)
        self._freq_btn.setStyleSheet(btn_style('default'))
        self._freq_btn.clicked.connect(self._on_domain_toggled)
        tog_row.addWidget(self._freq_btn)

        self._phase_btn = QPushButton("Magnitude")
        self._phase_btn.setCheckable(True)
        self._phase_btn.setChecked(False)
        self._phase_btn.setEnabled(False)
        self._phase_btn.setStyleSheet(btn_style('default'))
        self._phase_btn.clicked.connect(self._on_phase_toggled)
        tog_row.addWidget(self._phase_btn)

        lyt.addLayout(tog_row)

        self._fig = Figure(figsize=(3, 2), dpi=80, facecolor=p['BG'],
                           tight_layout=True)
        self._ax = self._fig.add_subplot(111, facecolor=p['PANEL'])
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        self._update_canvas_theme(p)
        lyt.addWidget(self._canvas)
        return w

    # ---- NOTCH tab ----

    def _build_notch_tab(self) -> QWidget:
        p = _get_theme()
        w = QWidget()
        lyt = QVBoxLayout(w)
        lyt.setContentsMargins(8, 8, 8, 8)
        lyt.setSpacing(6)

        lyt.addWidget(self._header("INTERACTIVE NOTCH"))

        instr = QLabel("Click spectrum to add notch points (auto-symmetric)")
        instr.setStyleSheet(FIELD_SS)
        instr.setWordWrap(True)
        lyt.addWidget(instr)

        # Notch matplotlib canvas
        self._notch_fig = Figure(figsize=(3, 3), dpi=80, facecolor=p['BG'],
                                 tight_layout=True)
        self._notch_ax = self._notch_fig.add_subplot(111, facecolor=p['PANEL'])
        self._notch_ax.axis('off')
        self._notch_canvas = FigureCanvas(self._notch_fig)
        self._notch_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Expanding)
        self._notch_canvas.mpl_connect('button_press_event', self._on_notch_click)
        lyt.addWidget(self._notch_canvas)

        self._points_lbl = QLabel("Points: 0")
        self._points_lbl.setStyleSheet(FIELD_SS)
        lyt.addWidget(self._points_lbl)

        radius_row = QHBoxLayout()
        radius_row.addWidget(self._lbl("RADIUS"))
        self._radius_spin = QSpinBox()
        self._radius_spin.setStyleSheet(SPINBOX_SS)
        self._radius_spin.setRange(1, 50)
        self._radius_spin.setValue(10)
        radius_row.addWidget(self._radius_spin)
        lyt.addLayout(radius_row)

        clear_btn = QPushButton("CLEAR POINTS")
        clear_btn.setStyleSheet(btn_style('default'))
        clear_btn.clicked.connect(self._clear_notch_points)
        lyt.addWidget(clear_btn)

        apply_btn = QPushButton("APPLY NOTCH FILTER")
        apply_btn.setStyleSheet(APPLY_BTN_SS)
        apply_btn.clicked.connect(self._apply_notch)
        lyt.addWidget(apply_btn)

        return w

    # ------------------------------------------------------------------
    # Canvas helpers
    # ------------------------------------------------------------------

    def _update_canvas_theme(self, p: dict):
        self._fig.set_facecolor(p['BG'])
        self._ax.set_facecolor(p['PANEL'])
        self._ax.tick_params(colors=p['MUTED'], labelsize=7)
        for spine in self._ax.spines.values():
            spine.set_color(p['BORDER'])
        if hasattr(self, '_notch_fig'):
            self._notch_fig.set_facecolor(p['BG'])
            self._notch_ax.set_facecolor(p['PANEL'])
            self._notch_ax.tick_params(colors=p['MUTED'], labelsize=7)
            for spine in self._notch_ax.spines.values():
                spine.set_color(p['BORDER'])

    def _trigger_canvas_update(self):
        if self._current_image is None:
            return
        if self._canvas_worker and self._canvas_worker.isRunning():
            return

        if self._freq_btn.isChecked():
            fn = _canvas_phase if self._phase_btn.isChecked() else _canvas_magnitude
        else:
            fn = _canvas_spatial

        state = _StaticState(self._current_image)
        self._canvas_worker = PipelineWorker(fn, "_canvas", state)
        self._canvas_worker.finished.connect(self._on_canvas_done)
        self._canvas_worker.error.connect(self.error_occurred)
        self._canvas_worker.start()

    def _on_canvas_done(self, _op_name: str, result: np.ndarray):
        p = _get_theme()
        self._ax.cla()
        self._ax.imshow(result, cmap='gray', aspect='auto',
                        interpolation='nearest')
        self._ax.axis('off')
        self._ax.set_facecolor(p['PANEL'])
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Notch interactive canvas
    # ------------------------------------------------------------------

    def _redraw_notch_canvas(self):
        if self._notch_spectrum_data is None:
            return
        p = _get_theme()
        self._notch_ax.cla()
        self._notch_fig.set_facecolor(p['BG'])
        self._notch_ax.set_facecolor(p['PANEL'])
        self._notch_ax.imshow(self._notch_spectrum_data, cmap='gray', aspect='auto')
        self._notch_ax.axis('off')
        for (r, c) in self._notch_centers:
            self._notch_ax.plot(c, r, 'rx', markersize=8, markeredgewidth=2)
        self._notch_canvas.draw()

    def _on_notch_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        if self._notch_spectrum_data is None:
            return
        H, W = self._notch_spectrum_data.shape[:2]
        col = int(round(float(event.xdata)))
        row = int(round(float(event.ydata)))
        col = max(0, min(col, W - 1))
        row = max(0, min(row, H - 1))
        sym_row = H - 1 - row
        sym_col = W - 1 - col
        self._notch_centers.append((row, col))
        if (sym_row, sym_col) != (row, col):
            self._notch_centers.append((sym_row, sym_col))
        self._points_lbl.setText(f"Points: {len(self._notch_centers)}")
        self._redraw_notch_canvas()

    def _clear_notch_points(self):
        self._notch_centers.clear()
        self._points_lbl.setText("Points: 0")
        self._redraw_notch_canvas()

    # ------------------------------------------------------------------
    # Toggle handlers
    # ------------------------------------------------------------------

    def _on_domain_toggled(self, checked: bool):
        self._freq_btn.setText("Frequency Domain" if checked else "Spatial Domain")
        self._phase_btn.setEnabled(checked)
        if not checked:
            self._phase_btn.setChecked(False)
            self._phase_btn.setText("Magnitude")
        self._trigger_canvas_update()

    def _on_phase_toggled(self, checked: bool):
        self._phase_btn.setText("Phase" if checked else "Magnitude")
        self._trigger_canvas_update()

    # ------------------------------------------------------------------
    # Apply operations
    # ------------------------------------------------------------------

    def _start_worker(self, fn, op_name: str, **kwargs):
        if self._state is None:
            return
        if self._worker and self._worker.isRunning():
            return
        self._worker = PipelineWorker(fn, op_name, self._state, **kwargs)
        self._worker.finished.connect(self.fourier_applied)
        self._worker.error.connect(self.error_occurred)
        self._worker.start()

    def _apply_notch(self):
        if not self._notch_centers:
            return
        self._start_worker(
            _notch_filter_fn, "Notch Filter",
            notch_centers=list(self._notch_centers),
            radius=self._radius_spin.value(),
        )
