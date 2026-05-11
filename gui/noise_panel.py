import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QTabWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal

from gui.theme import get as _get_theme
from gui.styles import COMBO_SS, SPINBOX_SS, APPLY_BTN_SS, HEADER_SS, FIELD_SS
from gui.workers import PipelineWorker


class NoisePanel(QWidget):
    noise_applied  = pyqtSignal(str, np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._worker: PipelineWorker | None = None
        self._roi: tuple | None = None  # (x, y, w, h)
        self._build_ui()

    def set_state(self, state):
        self._state = state

    def set_roi(self, x: int, y: int, w: int, h: int):
        self._roi = (x, y, w, h)
        self._roi_lbl.setText(f"ROI: ({x},{y}) {w}×{h}")
        self._compute_stats()

    def clear_roi(self):
        self._roi = None
        self._roi_lbl.setText("ROI: none selected")
        for lbl in (self._stat_mean, self._stat_std, self._stat_var,
                    self._stat_snr, self._stat_ent, self._stat_mse,
                    self._stat_psnr, self._stat_cnr):
            lbl.setText("---")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_inject_tab(),  "INJECT")
        tabs.addTab(self._build_roi_tab(),     "ROI STATS")
        tabs.addTab(self._build_adaptive_tab(), "ADAPTIVE MEDIAN")
        layout.addWidget(tabs)

    def _lbl(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(FIELD_SS)
        return lbl

    def _header(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(HEADER_SS)
        return lbl

    def _stat_row(self, label: str):
        p = _get_theme()
        row = QWidget()
        row_lyt = QHBoxLayout(row)
        row_lyt.setContentsMargins(0, 2, 0, 2)
        name_lbl = QLabel(label)
        name_lbl.setStyleSheet(FIELD_SS)
        val_lbl = QLabel("---")
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        val_lbl.setStyleSheet(
            f"color: {p['TEXT']}; font-family: 'JetBrains Mono', Consolas, monospace; "
            f"font-size: 10px;"
        )
        row_lyt.addWidget(name_lbl)
        row_lyt.addWidget(val_lbl)
        return row, val_lbl

    # ---- INJECT tab ----

    def _build_inject_tab(self) -> QWidget:
        w = QWidget()
        lyt = QVBoxLayout(w)
        lyt.setContentsMargins(8, 8, 8, 8)
        lyt.setSpacing(6)

        lyt.addWidget(self._header("NOISE TYPE"))
        self._noise_combo = QComboBox()
        self._noise_combo.setStyleSheet(COMBO_SS)
        for name in ["Gaussian", "Uniform", "Rayleigh", "Exponential", "Salt & Pepper"]:
            self._noise_combo.addItem(name)
        self._noise_combo.currentIndexChanged.connect(self._on_noise_changed)
        lyt.addWidget(self._noise_combo)

        # Gaussian
        self._gauss_w = QWidget()
        gl = QVBoxLayout(self._gauss_w)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setSpacing(4)
        mean_row = QHBoxLayout()
        mean_row.addWidget(self._lbl("MEAN"))
        self._gauss_mean = QDoubleSpinBox()
        self._gauss_mean.setStyleSheet(SPINBOX_SS)
        self._gauss_mean.setRange(-50.0, 50.0)
        self._gauss_mean.setValue(0.0)
        mean_row.addWidget(self._gauss_mean)
        gl.addLayout(mean_row)
        std_row = QHBoxLayout()
        std_row.addWidget(self._lbl("STD"))
        self._gauss_std = QDoubleSpinBox()
        self._gauss_std.setStyleSheet(SPINBOX_SS)
        self._gauss_std.setRange(1.0, 100.0)
        self._gauss_std.setValue(25.0)
        std_row.addWidget(self._gauss_std)
        gl.addLayout(std_row)
        lyt.addWidget(self._gauss_w)

        # Uniform
        self._uniform_w = QWidget()
        ul = QVBoxLayout(self._uniform_w)
        ul.setContentsMargins(0, 0, 0, 0)
        ul.setSpacing(4)
        low_row = QHBoxLayout()
        low_row.addWidget(self._lbl("LOW"))
        self._uniform_low = QDoubleSpinBox()
        self._uniform_low.setStyleSheet(SPINBOX_SS)
        self._uniform_low.setRange(-100.0, 0.0)
        self._uniform_low.setValue(-30.0)
        low_row.addWidget(self._uniform_low)
        ul.addLayout(low_row)
        high_row = QHBoxLayout()
        high_row.addWidget(self._lbl("HIGH"))
        self._uniform_high = QDoubleSpinBox()
        self._uniform_high.setStyleSheet(SPINBOX_SS)
        self._uniform_high.setRange(0.0, 100.0)
        self._uniform_high.setValue(30.0)
        high_row.addWidget(self._uniform_high)
        ul.addLayout(high_row)
        lyt.addWidget(self._uniform_w)

        # Rayleigh
        self._rayleigh_w = QWidget()
        rl = QVBoxLayout(self._rayleigh_w)
        rl.setContentsMargins(0, 0, 0, 0)
        scale_row = QHBoxLayout()
        scale_row.addWidget(self._lbl("SCALE"))
        self._rayleigh_scale = QDoubleSpinBox()
        self._rayleigh_scale.setStyleSheet(SPINBOX_SS)
        self._rayleigh_scale.setRange(1.0, 100.0)
        self._rayleigh_scale.setValue(20.0)
        scale_row.addWidget(self._rayleigh_scale)
        rl.addLayout(scale_row)
        lyt.addWidget(self._rayleigh_w)

        # Exponential
        self._exp_w = QWidget()
        el = QVBoxLayout(self._exp_w)
        el.setContentsMargins(0, 0, 0, 0)
        exp_row = QHBoxLayout()
        exp_row.addWidget(self._lbl("SCALE"))
        self._exp_scale = QDoubleSpinBox()
        self._exp_scale.setStyleSheet(SPINBOX_SS)
        self._exp_scale.setRange(1.0, 100.0)
        self._exp_scale.setValue(20.0)
        exp_row.addWidget(self._exp_scale)
        el.addLayout(exp_row)
        lyt.addWidget(self._exp_w)

        # Salt & Pepper
        self._sp_w = QWidget()
        sl = QVBoxLayout(self._sp_w)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(4)
        salt_row = QHBoxLayout()
        salt_row.addWidget(self._lbl("SALT PROB"))
        self._salt_prob = QDoubleSpinBox()
        self._salt_prob.setStyleSheet(SPINBOX_SS)
        self._salt_prob.setRange(0.001, 0.5)
        self._salt_prob.setSingleStep(0.01)
        self._salt_prob.setDecimals(3)
        self._salt_prob.setValue(0.02)
        salt_row.addWidget(self._salt_prob)
        sl.addLayout(salt_row)
        pep_row = QHBoxLayout()
        pep_row.addWidget(self._lbl("PEPPER PROB"))
        self._pepper_prob = QDoubleSpinBox()
        self._pepper_prob.setStyleSheet(SPINBOX_SS)
        self._pepper_prob.setRange(0.001, 0.5)
        self._pepper_prob.setSingleStep(0.01)
        self._pepper_prob.setDecimals(3)
        self._pepper_prob.setValue(0.02)
        pep_row.addWidget(self._pepper_prob)
        sl.addLayout(pep_row)
        lyt.addWidget(self._sp_w)

        apply_btn = QPushButton("INJECT NOISE")
        apply_btn.setStyleSheet(APPLY_BTN_SS)
        apply_btn.clicked.connect(self._apply_noise)
        lyt.addWidget(apply_btn)
        lyt.addStretch()

        self._on_noise_changed(0)
        return w

    # ---- ROI STATS tab ----

    def _build_roi_tab(self) -> QWidget:
        w = QWidget()
        lyt = QVBoxLayout(w)
        lyt.setContentsMargins(8, 8, 8, 8)
        lyt.setSpacing(6)

        lyt.addWidget(self._header("ROI STATISTICS"))
        self._roi_lbl = QLabel("ROI: none selected")
        self._roi_lbl.setStyleSheet(FIELD_SS)
        lyt.addWidget(self._roi_lbl)

        row, self._stat_mean = self._stat_row("Mean")
        lyt.addWidget(row)
        row, self._stat_std = self._stat_row("Std Dev")
        lyt.addWidget(row)
        row, self._stat_var = self._stat_row("Variance")
        lyt.addWidget(row)
        row, self._stat_snr = self._stat_row("SNR")
        lyt.addWidget(row)
        row, self._stat_ent = self._stat_row("Entropy")
        lyt.addWidget(row)

        lyt.addWidget(self._header("MSE / PSNR"))
        row, self._stat_mse  = self._stat_row("MSE")
        lyt.addWidget(row)
        row, self._stat_psnr = self._stat_row("PSNR")
        lyt.addWidget(row)

        lyt.addWidget(self._header("CNR"))
        row, self._stat_cnr = self._stat_row("CNR")
        lyt.addWidget(row)

        lyt.addStretch()
        return w

    # ---- ADAPTIVE MEDIAN tab ----

    def _build_adaptive_tab(self) -> QWidget:
        w = QWidget()
        lyt = QVBoxLayout(w)
        lyt.setContentsMargins(8, 8, 8, 8)
        lyt.setSpacing(6)

        lyt.addWidget(self._header("ADAPTIVE MEDIAN FILTER"))

        win_row = QHBoxLayout()
        win_row.addWidget(self._lbl("MAX WINDOW"))
        self._max_window_spin = QSpinBox()
        self._max_window_spin.setStyleSheet(SPINBOX_SS)
        self._max_window_spin.setRange(3, 21)
        self._max_window_spin.setSingleStep(2)
        self._max_window_spin.setValue(7)
        win_row.addWidget(self._max_window_spin)
        lyt.addLayout(win_row)

        apply_btn = QPushButton("APPLY ADAPTIVE MEDIAN")
        apply_btn.setStyleSheet(APPLY_BTN_SS)
        apply_btn.clicked.connect(self._apply_adaptive_median)
        lyt.addWidget(apply_btn)
        lyt.addStretch()
        return w

    # ------------------------------------------------------------------
    # Visibility helpers
    # ------------------------------------------------------------------

    def _on_noise_changed(self, _idx: int):
        name = self._noise_combo.currentText()
        self._gauss_w.setVisible(name == "Gaussian")
        self._uniform_w.setVisible(name == "Uniform")
        self._rayleigh_w.setVisible(name == "Rayleigh")
        self._exp_w.setVisible(name == "Exponential")
        self._sp_w.setVisible(name == "Salt & Pepper")

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _start_worker(self, fn, op_name: str, **kwargs):
        if self._state is None:
            return
        if self._worker and self._worker.isRunning():
            return
        self._worker = PipelineWorker(fn, op_name, self._state, **kwargs)
        self._worker.finished.connect(self.noise_applied)
        self._worker.error.connect(self.error_occurred)
        self._worker.start()

    def _apply_noise(self):
        if self._state is None:
            return
        name = self._noise_combo.currentText()
        if name == "Gaussian":
            from processing.noise.noise_injection import add_gaussian_noise
            self._start_worker(add_gaussian_noise, "Gaussian Noise",
                               mean=self._gauss_mean.value(),
                               sigma=self._gauss_std.value())
        elif name == "Uniform":
            from processing.noise.noise_injection import add_uniform_noise
            self._start_worker(add_uniform_noise, "Uniform Noise",
                               low=self._uniform_low.value(),
                               high=self._uniform_high.value())
        elif name == "Rayleigh":
            from processing.noise.noise_injection import add_rayleigh_noise
            self._start_worker(add_rayleigh_noise, "Rayleigh Noise",
                               scale=self._rayleigh_scale.value())
        elif name == "Exponential":
            from processing.noise.noise_injection import add_exponential_noise
            self._start_worker(add_exponential_noise, "Exponential Noise",
                               scale=self._exp_scale.value())
        elif name == "Salt & Pepper":
            from processing.noise.noise_injection import add_salt_and_pepper_noise
            self._start_worker(add_salt_and_pepper_noise, "Salt&Pepper Noise",
                               salt_prob=self._salt_prob.value(),
                               pepper_prob=self._pepper_prob.value())

    def _compute_stats(self):
        if self._state is None or self._roi is None:
            return
        img = self._state.current()
        if img is None:
            return
        if img.ndim == 3:
            img = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1]
                   + 0.114 * img[:, :, 2]).astype(np.uint8)

        from processing.noise.roi_stats import compute_roi_stats
        x, y, w, h = self._roi
        stats = compute_roi_stats(img, x, y, w, h)

        mean = stats['mean']
        var  = stats['variance']
        std  = float(np.sqrt(var))
        snr  = mean / std if std > 1e-9 else float('inf')

        hist = stats['histogram'].astype(np.float64)
        total = hist.sum()
        if total > 0:
            p = hist / total
            p = p[p > 0]
            entropy = float(-np.sum(p * np.log2(p)))
        else:
            entropy = 0.0

        self._stat_mean.setText(f"{mean:.2f}")
        self._stat_std.setText(f"{std:.2f}")
        self._stat_var.setText(f"{var:.2f}")
        self._stat_snr.setText(f"{snr:.2f}" if np.isfinite(snr) else "∞")
        self._stat_ent.setText(f"{entropy:.3f} bits")
        self._stat_mse.setText("---")
        self._stat_psnr.setText("---")
        self._stat_cnr.setText("---")

    def _apply_adaptive_median(self):
        from processing.spatial.adaptive_median import adaptive_median_filter
        self._start_worker(adaptive_median_filter, "Adaptive Median",
                           max_window=self._max_window_spin.value())
