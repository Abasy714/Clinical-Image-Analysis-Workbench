import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QRadioButton, QButtonGroup, QDoubleSpinBox, QSpinBox,
    QTableWidget, QTableWidgetItem, QDialog, QGridLayout, QLineEdit,
    QTabWidget, QScrollArea, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from gui.theme import get as _get_theme
from gui.styles import btn_style, COMBO_SS, SPINBOX_SS, APPLY_BTN_SS, HEADER_SS, FIELD_SS
from gui.workers import PipelineWorker


def _sobel_fn(image, direction='magnitude'):
    from processing.spatial.edge_detection import sobel
    gx, gy, mag = sobel(image)
    return {'x': gx, 'y': gy, 'magnitude': mag}[direction]


def _prewitt_fn(image, direction='magnitude'):
    from processing.spatial.edge_detection import prewitt
    gx, gy, mag = prewitt(image)
    return {'x': gx, 'y': gy, 'magnitude': mag}[direction]


def _custom_fn(image, kernel):
    from processing.spatial.convolution import convolve2d
    raw = convolve2d(image, kernel).astype(np.float64)
    mn, mx = raw.min(), raw.max()
    if mx == mn:
        return np.zeros_like(image, dtype=np.uint8)
    return ((raw - mn) / (mx - mn) * 255).astype(np.uint8)


def _gaussian_preview_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _freq_filter_fn(image, filter_type, cutoff=30.0, order=2,
                    low_cutoff=10.0, high_cutoff=50.0):
    from processing.frequency.spectrum import compute_spectrum, inverse_spectrum
    from processing.frequency.filters import (
        create_low_pass_filter, create_high_pass_filter,
        create_band_pass_filter, create_band_reject_filter,
        apply_frequency_filter,
    )
    if image.ndim == 3:
        gray = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1]
                + 0.114 * image[:, :, 2]).astype(np.uint8)
    else:
        gray = image
    shape = gray.shape[:2]
    shifted_fft, _, _ = compute_spectrum(gray)
    center = (low_cutoff + high_cutoff) / 2.0
    bandwidth = max(high_cutoff - low_cutoff, 1.0)
    _masks = {
        'Ideal Lowpass':        lambda: create_low_pass_filter(shape, cutoff, kind='ideal'),
        'Ideal Highpass':       lambda: create_high_pass_filter(shape, cutoff, kind='ideal'),
        'Butterworth Lowpass':  lambda: create_low_pass_filter(shape, cutoff, kind='butterworth', order=order),
        'Butterworth Highpass': lambda: create_high_pass_filter(shape, cutoff, kind='butterworth', order=order),
        'Gaussian Lowpass':     lambda: create_low_pass_filter(shape, cutoff, kind='gaussian'),
        'Gaussian Highpass':    lambda: create_high_pass_filter(shape, cutoff, kind='gaussian'),
        'Bandpass':             lambda: create_band_pass_filter(shape, center, bandwidth),
        'Bandreject':           lambda: create_band_reject_filter(shape, center, bandwidth),
    }
    mask_fn = _masks.get(filter_type)
    if mask_fn is None:
        raise ValueError(f"Unknown filter_type: {filter_type!r}")
    filtered = apply_frequency_filter(shifted_fft, mask_fn())
    return inverse_spectrum(filtered)


_KERNEL_PRESETS = {
    'Sharpen':      np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float),
    'Emboss':       np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], float),
    'Edge Enhance': np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], float),
    'Box Blur':     np.ones((3, 3)) / 9,
    'Laplacian':    np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float),
}


class KernelEditorDialog(QDialog):
    def __init__(self, size=3, kernel=None, parent=None):
        super().__init__(parent)
        p = _get_theme()
        self.setWindowTitle("Custom Kernel")
        self.setModal(True)
        self._size = size
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # AUTO-FILL row
        fill_row = QHBoxLayout()
        fill_lbl = QLabel("AUTO-FILL")
        fill_lbl.setStyleSheet(FIELD_SS)
        fill_row.addWidget(fill_lbl)
        self._preset_combo = QComboBox()
        self._preset_combo.setStyleSheet(COMBO_SS)
        for name in _KERNEL_PRESETS.keys():
            self._preset_combo.addItem(name)
        fill_row.addWidget(self._preset_combo)
        fill_btn = QPushButton("FILL")
        fill_btn.setStyleSheet(btn_style('default'))
        fill_btn.clicked.connect(self._fill_preset)
        fill_row.addWidget(fill_btn)
        layout.addLayout(fill_row)

        self._grid = QGridLayout()
        self._grid.setSpacing(2)
        self._cells: list[list[QLineEdit]] = []

        cell_ss = (
            f"QLineEdit {{ background: {p['INPUT']}; color: {p['TEXT']}; "
            f"border: 1px solid {p['BORDER2']}; border-radius: 2px; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; font-size: 10px; "
            f"padding: 2px; }}"
            f"QLineEdit:focus {{ border-color: {p['ACCENT']}; }}"
        )
        for r in range(size):
            row_cells = []
            for c in range(size):
                le = QLineEdit("0")
                le.setFixedSize(48, 28)
                le.setAlignment(Qt.AlignmentFlag.AlignCenter)
                le.setStyleSheet(cell_ss)
                if kernel is not None:
                    le.setText(f"{kernel[r, c]:.3g}")
                self._grid.addWidget(le, r, c)
                row_cells.append(le)
            self._cells.append(row_cells)

        layout.addLayout(self._grid)

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("CLEAR")
        clear_btn.setStyleSheet(btn_style('default'))
        clear_btn.clicked.connect(self._clear_cells)
        cancel_btn = QPushButton("CANCEL")
        cancel_btn.setStyleSheet(btn_style('default'))
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("OK")
        ok_btn.setStyleSheet(btn_style('primary'))
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

    def _clear_cells(self):
        for r in range(self._size):
            for c in range(self._size):
                self._cells[r][c].setText("0")

    def _fill_preset(self):
        name = self._preset_combo.currentText()
        preset = _KERNEL_PRESETS.get(name)
        if preset is None:
            return
        self._clear_cells()
        ph, pw = preset.shape
        if self._size >= ph:
            r0 = (self._size - ph) // 2
            c0 = (self._size - pw) // 2
            for r in range(ph):
                for c in range(pw):
                    self._cells[r0 + r][c0 + c].setText(f"{preset[r, c]:.3g}")
        else:
            for r in range(self._size):
                for c in range(self._size):
                    self._cells[r][c].setText(f"{preset[r, c]:.3g}")

    def get_kernel(self) -> np.ndarray | None:
        try:
            data = [[float(self._cells[r][c].text()) for c in range(self._size)]
                    for r in range(self._size)]
            return np.array(data, dtype=np.float64)
        except ValueError:
            return None


class FilterPanel(QWidget):
    filter_applied = pyqtSignal(str, np.ndarray)
    error_occurred = pyqtSignal(str)
    interp_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._worker: PipelineWorker | None = None
        self._custom_kernel: np.ndarray | None = None
        self._build_ui()

    def set_state(self, state):
        self._state = state

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_spatial_tab(), "SPATIAL")
        tabs.addTab(self._build_geometric_tab(), "GEOMETRIC")
        tabs.addTab(self._build_interp_tab(), "INTERP")
        layout.addWidget(tabs)

    def _lbl(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(FIELD_SS)
        return lbl

    def _header(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(HEADER_SS)
        return lbl

    # ---- SPATIAL tab ----

    def _build_spatial_tab(self) -> QScrollArea:
        p = _get_theme()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        w = QWidget()
        lyt = QVBoxLayout(w)
        lyt.setContentsMargins(6, 6, 6, 6)
        lyt.setSpacing(6)

        # ===== Section 1: SPATIAL FILTERS =====
        lyt.addWidget(self._header("SPATIAL FILTERS"))

        self._filter_combo = QComboBox()
        self._filter_combo.setStyleSheet(COMBO_SS)
        for name in [
            "Average", "Gaussian", "Median", "Sobel", "Prewitt",
            "Harmonic Mean", "Contra-Harmonic", "Midpoint", "Min", "Max",
            "Adaptive Median", "Custom Kernel",
        ]:
            self._filter_combo.addItem(name)
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        lyt.addWidget(self._filter_combo)

        # Kernel size radios
        self._ksize_row_w = QWidget()
        ksize_lyt = QHBoxLayout(self._ksize_row_w)
        ksize_lyt.setContentsMargins(0, 0, 0, 0)
        ksize_lyt.addWidget(self._lbl("KERNEL SIZE"))
        self._ksize_group = QButtonGroup(self)
        for sz, label in [(3, "3x3"), (5, "5x5"), (7, "7x7"), (9, "9x9")]:
            rb = QRadioButton(label)
            if sz == 3:
                rb.setChecked(True)
            self._ksize_group.addButton(rb, sz)
            ksize_lyt.addWidget(rb)
        self._ksize_group.idClicked.connect(self._update_kernel_preview)
        lyt.addWidget(self._ksize_row_w)

        # Direction container (Sobel/Prewitt)
        self._dir_container = QWidget()
        dir_lyt = QVBoxLayout(self._dir_container)
        dir_lyt.setContentsMargins(0, 0, 0, 0)
        dir_lyt.setSpacing(4)
        dir_lyt.addWidget(self._lbl("DIRECTION"))
        dir_btn_row = QHBoxLayout()
        dir_btn_row.setSpacing(4)
        self._dir_group = QButtonGroup(self)
        self._dir_group.setExclusive(True)
        for did, lbl in [(0, "H"), (1, "V"), (2, "Mag")]:
            b = QPushButton(lbl)
            b.setCheckable(True)
            b.setStyleSheet(btn_style('default'))
            if did == 2:
                b.setChecked(True)
            self._dir_group.addButton(b, did)
            dir_btn_row.addWidget(b)
        self._dir_group.idClicked.connect(self._update_kernel_preview)
        dir_lyt.addLayout(dir_btn_row)
        lyt.addWidget(self._dir_container)

        # Sigma container (Gaussian)
        self._sigma_container = QWidget()
        sig_lyt = QHBoxLayout(self._sigma_container)
        sig_lyt.setContentsMargins(0, 0, 0, 0)
        sig_lyt.addWidget(self._lbl("SIGMA"))
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setStyleSheet(SPINBOX_SS)
        self._sigma_spin.setRange(0.1, 20.0)
        self._sigma_spin.setSingleStep(0.1)
        self._sigma_spin.setValue(1.0)
        self._sigma_spin.valueChanged.connect(self._update_kernel_preview)
        sig_lyt.addWidget(self._sigma_spin)
        lyt.addWidget(self._sigma_container)

        # Q container (Contra-Harmonic)
        self._q_container = QWidget()
        q_lyt = QHBoxLayout(self._q_container)
        q_lyt.setContentsMargins(0, 0, 0, 0)
        q_lyt.addWidget(self._lbl("Q"))
        self._q_spin = QDoubleSpinBox()
        self._q_spin.setStyleSheet(SPINBOX_SS)
        self._q_spin.setRange(-5.0, 5.0)
        self._q_spin.setSingleStep(0.5)
        self._q_spin.setValue(1.5)
        q_lyt.addWidget(self._q_spin)
        lyt.addWidget(self._q_container)

        # Max window container (Adaptive Median)
        self._maxwin_container = QWidget()
        mw_lyt = QHBoxLayout(self._maxwin_container)
        mw_lyt.setContentsMargins(0, 0, 0, 0)
        mw_lyt.addWidget(self._lbl("MAX WINDOW"))
        self._maxwin_spin = QSpinBox()
        self._maxwin_spin.setStyleSheet(SPINBOX_SS)
        self._maxwin_spin.setRange(3, 21)
        self._maxwin_spin.setSingleStep(2)
        self._maxwin_spin.setValue(7)
        mw_lyt.addWidget(self._maxwin_spin)
        lyt.addWidget(self._maxwin_container)

        # Kernel preview
        self._kernel_preview_lbl = self._lbl("KERNEL PREVIEW")
        lyt.addWidget(self._kernel_preview_lbl)
        self._kernel_table = QTableWidget(3, 3)
        self._kernel_table.setFixedHeight(90)
        self._kernel_table.horizontalHeader().setVisible(False)
        self._kernel_table.verticalHeader().setVisible(False)
        self._kernel_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._kernel_table.setStyleSheet(
            f"QTableWidget {{ background: {p['BG']}; border: 1px solid {p['BORDER']}; "
            f"color: {p['TEXT']}; font-family: 'JetBrains Mono', Consolas, monospace; "
            f"font-size: 9px; gridline-color: {p['BORDER']}; }}"
        )
        lyt.addWidget(self._kernel_table)

        self._custom_btn = QPushButton("EDIT CUSTOM KERNEL")
        self._custom_btn.setStyleSheet(btn_style('ghost'))
        self._custom_btn.clicked.connect(self._open_kernel_editor)
        lyt.addWidget(self._custom_btn)

        apply_spatial_btn = QPushButton("APPLY SPATIAL FILTER")
        apply_spatial_btn.setStyleSheet(APPLY_BTN_SS)
        apply_spatial_btn.clicked.connect(self._apply_spatial)
        lyt.addWidget(apply_spatial_btn)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"color: {p['BORDER']}; background: {p['BORDER']};")
        divider.setFixedHeight(1)
        lyt.addWidget(divider)

        # ===== Section 2: FREQUENCY DOMAIN FILTERS =====
        lyt.addWidget(self._header("FREQUENCY DOMAIN FILTERS"))

        self._freq_combo = QComboBox()
        self._freq_combo.setStyleSheet(COMBO_SS)
        for name in [
            "Ideal Lowpass", "Ideal Highpass",
            "Butterworth Lowpass", "Butterworth Highpass",
            "Gaussian Lowpass", "Gaussian Highpass",
            "Bandpass", "Bandreject",
        ]:
            self._freq_combo.addItem(name)
        self._freq_combo.currentIndexChanged.connect(self._on_freq_filter_changed)
        lyt.addWidget(self._freq_combo)

        self._cutoff_w = QWidget()
        cl = QHBoxLayout(self._cutoff_w)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(self._lbl("CUTOFF"))
        self._cutoff_spin = QDoubleSpinBox()
        self._cutoff_spin.setStyleSheet(SPINBOX_SS)
        self._cutoff_spin.setRange(1.0, 200.0)
        self._cutoff_spin.setValue(30.0)
        cl.addWidget(self._cutoff_spin)
        lyt.addWidget(self._cutoff_w)

        self._order_w = QWidget()
        ol = QHBoxLayout(self._order_w)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.addWidget(self._lbl("ORDER"))
        self._order_spin = QSpinBox()
        self._order_spin.setStyleSheet(SPINBOX_SS)
        self._order_spin.setRange(1, 10)
        self._order_spin.setValue(2)
        ol.addWidget(self._order_spin)
        lyt.addWidget(self._order_w)

        self._low_w = QWidget()
        ll = QHBoxLayout(self._low_w)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self._lbl("LOW CUTOFF"))
        self._low_spin = QDoubleSpinBox()
        self._low_spin.setStyleSheet(SPINBOX_SS)
        self._low_spin.setRange(1.0, 200.0)
        self._low_spin.setValue(10.0)
        ll.addWidget(self._low_spin)
        lyt.addWidget(self._low_w)

        self._high_w = QWidget()
        hl = QHBoxLayout(self._high_w)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self._lbl("HIGH CUTOFF"))
        self._high_spin = QDoubleSpinBox()
        self._high_spin.setStyleSheet(SPINBOX_SS)
        self._high_spin.setRange(1.0, 200.0)
        self._high_spin.setValue(50.0)
        hl.addWidget(self._high_spin)
        lyt.addWidget(self._high_w)

        apply_freq_btn = QPushButton("APPLY FREQUENCY FILTER")
        apply_freq_btn.setStyleSheet(APPLY_BTN_SS)
        apply_freq_btn.clicked.connect(self._apply_freq)
        lyt.addWidget(apply_freq_btn)

        lyt.addStretch()

        scroll.setWidget(w)

        self._on_filter_changed(0)
        self._on_freq_filter_changed(0)
        return scroll

    # ---- GEOMETRIC tab ----

    def _build_geometric_tab(self) -> QWidget:
        w = QWidget()
        lyt = QVBoxLayout(w)
        lyt.setContentsMargins(6, 6, 6, 6)
        lyt.setSpacing(6)

        lyt.addWidget(self._header("ROTATION"))
        angle_row = QHBoxLayout()
        angle_row.addWidget(self._lbl("ANGLE (°)"))
        self._angle_spin = QDoubleSpinBox()
        self._angle_spin.setStyleSheet(SPINBOX_SS)
        self._angle_spin.setRange(-360.0, 360.0)
        self._angle_spin.setSingleStep(1.0)
        self._angle_spin.setValue(0.0)
        angle_row.addWidget(self._angle_spin)
        lyt.addLayout(angle_row)

        rotate_btn = QPushButton("APPLY ROTATION")
        rotate_btn.setStyleSheet(APPLY_BTN_SS)
        rotate_btn.clicked.connect(self._apply_rotation)
        lyt.addWidget(rotate_btn)

        lyt.addWidget(self._header("SHEAR"))
        sx_row = QHBoxLayout()
        sx_row.addWidget(self._lbl("SHEAR X"))
        self._shear_x_spin = QDoubleSpinBox()
        self._shear_x_spin.setStyleSheet(SPINBOX_SS)
        self._shear_x_spin.setRange(-2.0, 2.0)
        self._shear_x_spin.setSingleStep(0.05)
        self._shear_x_spin.setValue(0.0)
        sx_row.addWidget(self._shear_x_spin)
        lyt.addLayout(sx_row)

        sy_row = QHBoxLayout()
        sy_row.addWidget(self._lbl("SHEAR Y"))
        self._shear_y_spin = QDoubleSpinBox()
        self._shear_y_spin.setStyleSheet(SPINBOX_SS)
        self._shear_y_spin.setRange(-2.0, 2.0)
        self._shear_y_spin.setSingleStep(0.05)
        self._shear_y_spin.setValue(0.0)
        sy_row.addWidget(self._shear_y_spin)
        lyt.addLayout(sy_row)

        shear_btn = QPushButton("APPLY SHEAR")
        shear_btn.setStyleSheet(APPLY_BTN_SS)
        shear_btn.clicked.connect(self._apply_shear)
        lyt.addWidget(shear_btn)
        lyt.addStretch()
        return w

    # ---- INTERP tab ----

    def _build_interp_tab(self) -> QWidget:
        w = QWidget()
        lyt = QVBoxLayout(w)
        lyt.setContentsMargins(6, 6, 6, 6)
        lyt.setSpacing(6)

        lyt.addWidget(self._header("INTERPOLATION"))

        self._interp_group = QButtonGroup(self)
        self._nn_radio = QRadioButton("Nearest Neighbor")
        self._bl_radio = QRadioButton("Bilinear")
        self._nn_radio.setChecked(True)
        self._interp_group.addButton(self._nn_radio, 0)
        self._interp_group.addButton(self._bl_radio, 1)
        lyt.addWidget(self._nn_radio)
        lyt.addWidget(self._bl_radio)

        lyt.addWidget(self._header("ZOOM"))
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(self._lbl("FACTOR"))
        self._zoom_spin = QDoubleSpinBox()
        self._zoom_spin.setStyleSheet(SPINBOX_SS)
        self._zoom_spin.setRange(0.1, 8.0)
        self._zoom_spin.setSingleStep(0.1)
        self._zoom_spin.setValue(1.0)
        zoom_row.addWidget(self._zoom_spin)
        lyt.addLayout(zoom_row)

        zoom_btn = QPushButton("APPLY ZOOM")
        zoom_btn.setStyleSheet(APPLY_BTN_SS)
        zoom_btn.clicked.connect(self._apply_zoom)
        lyt.addWidget(zoom_btn)
        lyt.addStretch()
        return w

    # ------------------------------------------------------------------
    # Visibility / preview
    # ------------------------------------------------------------------

    def _on_filter_changed(self, idx: int):
        name = self._filter_combo.currentText()
        self._dir_container.setVisible(name in ("Sobel", "Prewitt"))
        self._sigma_container.setVisible(name == "Gaussian")
        self._q_container.setVisible(name == "Contra-Harmonic")
        self._maxwin_container.setVisible(name == "Adaptive Median")
        self._custom_btn.setVisible(name == "Custom Kernel")
        has_kernel = name in ("Average", "Gaussian", "Median", "Custom Kernel")
        self._kernel_table.setVisible(has_kernel)
        self._kernel_preview_lbl.setVisible(has_kernel)
        self._update_kernel_preview()

    def _on_freq_filter_changed(self, idx: int):
        name = self._freq_combo.currentText()
        is_band = name in ("Bandpass", "Bandreject")
        is_butter = "Butterworth" in name
        self._cutoff_w.setVisible(not is_band)
        self._order_w.setVisible(is_butter)
        self._low_w.setVisible(is_band)
        self._high_w.setVisible(is_band)

    def _update_kernel_preview(self):
        p = _get_theme()
        name = self._filter_combo.currentText()
        size = self._ksize_group.checkedId()
        if size <= 0:
            size = 3

        if name == "Average":
            k = np.ones((size, size)) / (size * size)
        elif name == "Gaussian":
            k = _gaussian_preview_kernel(size, self._sigma_spin.value())
        elif name == "Median":
            k = np.ones((size, size)) / (size * size)
        elif name == "Custom Kernel":
            k = self._custom_kernel if self._custom_kernel is not None else np.zeros((size, size))
        else:
            return

        rows, cols = k.shape
        self._kernel_table.setRowCount(rows)
        self._kernel_table.setColumnCount(cols)
        for r in range(rows):
            for c in range(cols):
                v = k[r, c]
                item = QTableWidgetItem(f"{v:.3g}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if v > 1e-9:
                    item.setForeground(QColor(p['ACCENT_DIM']))
                elif v < -1e-9:
                    item.setForeground(QColor(p['RED']))
                else:
                    item.setForeground(QColor(p['BORDER2']))
                self._kernel_table.setItem(r, c, item)
        cw = max(1, self._kernel_table.width() // cols)
        rh = max(1, 72 // rows)
        for c in range(cols):
            self._kernel_table.setColumnWidth(c, cw)
        for r in range(rows):
            self._kernel_table.setRowHeight(r, rh)

    def _open_kernel_editor(self):
        size = self._ksize_group.checkedId()
        if size <= 0:
            size = 3
        dlg = KernelEditorDialog(size, self._custom_kernel, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            k = dlg.get_kernel()
            if k is not None:
                self._custom_kernel = k
                self._update_kernel_preview()

    # ------------------------------------------------------------------
    # Apply operations
    # ------------------------------------------------------------------

    def _start_worker(self, fn, op_name: str, **kwargs):
        if self._state is None:
            return
        if self._worker and self._worker.isRunning():
            return
        self._worker = PipelineWorker(fn, op_name, self._state, **kwargs)
        self._worker.finished.connect(self.filter_applied)
        self._worker.error.connect(self.error_occurred)
        self._worker.start()

    def _apply_spatial(self):
        if self._state is None:
            return
        name = self._filter_combo.currentText()
        size = self._ksize_group.checkedId()
        if size <= 0:
            size = 3
        sigma = self._sigma_spin.value()
        q = self._q_spin.value()
        max_window = self._maxwin_spin.value()
        dir_id = self._dir_group.checkedId()
        direction = {0: 'x', 1: 'y', 2: 'magnitude'}.get(dir_id, 'magnitude')

        if name == "Average":
            from processing.spatial.smoothing import average_filter
            self._start_worker(average_filter, "Average Filter", kernel_size=size)
        elif name == "Gaussian":
            from processing.spatial.smoothing import gaussian_filter
            self._start_worker(gaussian_filter, "Gaussian Filter", kernel_size=size, sigma=sigma)
        elif name == "Median":
            from processing.spatial.median_filter import median_filter
            self._start_worker(median_filter, "Median Filter", kernel_size=size)
        elif name == "Sobel":
            self._start_worker(_sobel_fn, f"Sobel {direction.title()}", direction=direction)
        elif name == "Prewitt":
            self._start_worker(_prewitt_fn, f"Prewitt {direction.title()}", direction=direction)
        elif name == "Harmonic Mean":
            from processing.spatial.mean_filters import harmonic_mean_filter
            self._start_worker(harmonic_mean_filter, "Harmonic Mean", kernel_size=size)
        elif name == "Contra-Harmonic":
            from processing.spatial.mean_filters import contraharmonic_mean_filter
            self._start_worker(contraharmonic_mean_filter, "Contra-Harmonic", kernel_size=size, Q=q)
        elif name == "Midpoint":
            from processing.spatial.order_statistic_filters import midpoint_filter
            self._start_worker(midpoint_filter, "Midpoint Filter", kernel_size=size)
        elif name == "Min":
            from processing.spatial.order_statistic_filters import min_filter
            self._start_worker(min_filter, "Min Filter", kernel_size=size)
        elif name == "Max":
            from processing.spatial.order_statistic_filters import max_filter
            self._start_worker(max_filter, "Max Filter", kernel_size=size)
        elif name == "Adaptive Median":
            from processing.spatial.adaptive_median import adaptive_median_filter
            self._start_worker(adaptive_median_filter, "Adaptive Median", max_window=max_window)
        elif name == "Custom Kernel":
            if self._custom_kernel is None:
                self.error_occurred.emit("No custom kernel defined.")
                return
            self._start_worker(_custom_fn, "Custom Filter", kernel=self._custom_kernel)

    def _apply_freq(self):
        name = self._freq_combo.currentText()
        self._start_worker(
            _freq_filter_fn, f"Freq: {name}",
            filter_type=name,
            cutoff=self._cutoff_spin.value(),
            order=self._order_spin.value(),
            low_cutoff=self._low_spin.value(),
            high_cutoff=self._high_spin.value(),
        )

    def _apply_rotation(self):
        from processing.geometric.rotation import rotate_image
        self._start_worker(rotate_image, "Rotate", angle_deg=self._angle_spin.value())

    def _apply_shear(self):
        from processing.geometric.shearing import shear_image
        self._start_worker(shear_image, "Shear",
                           shear_x=self._shear_x_spin.value(),
                           shear_y=self._shear_y_spin.value())

    def _apply_zoom(self):
        from processing.interpolation.zoom import apply_zoom
        mode = 'bilinear' if self._bl_radio.isChecked() else 'nearest'
        self.interp_changed.emit(mode)
        self._start_worker(apply_zoom, "Zoom",
                           zoom_factor=self._zoom_spin.value(),
                           mode=mode)
