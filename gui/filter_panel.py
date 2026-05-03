# STATUS: IMPLEMENTED
"""
Control panel for spatial filtering operations.
Allows the user to select kernel size and apply average, Gaussian, Sobel/Prewitt, or median filters.
"""

# PyQt6.QtWidgets — QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox, QLabel, QRadioButton
# PyQt6.QtCore — pyqtSignal
# processing.spatial.smoothing — average_filter, gaussian_filter
# processing.spatial.edge_detection — sobel, prewitt, combined_magnitude
# processing.spatial.median_filter — median_filter
# utils.image_utils — validate_grayscale: ensure image is single-channel before filtering

# FUNCTIONS / CLASSES
# class FilterPanel(QWidget):
#   def __init__: build UI — filter type selector, kernel size spinner, sigma input (Gaussian only)
#   def on_apply_clicked: read UI state → dispatch to correct processing function → emit result
#   def _toggle_sigma_input: show/hide sigma spinner based on selected filter type
# signal: filter_applied(np.ndarray) — emitted with the filtered image result

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale, normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call at the top of on_apply_clicked before dispatching
# normalize_to_uint8(result)          # call on the filter output before emitting the signal
# @wrap_errors                        # decorate on_apply_clicked — kernel ops can raise on bad input

import math
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
                              QLabel, QPushButton, QDoubleSpinBox, QDialog,
                              QGridLayout, QLineEdit, QScrollArea, QButtonGroup,
                              QRadioButton, QFrame, QSizePolicy)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor

from gui.styles import (BG, PANEL, PANEL2, INPUT, BORDER, BORDER2,
                        ACCENT, ACCENT2, ACCENT_DIM, TEXT, MUTED, MUTED2, btn_style)
from utils import (validate_grayscale, normalize_to_uint8, wrap_errors, show_error_dialog,)


class FilterPanel(QWidget):
    filter_applied = pyqtSignal(str, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{PANEL};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # section label
        hdr = QLabel("SPATIAL FILTER")
        hdr.setStyleSheet(f"color:{ACCENT};font-size:9px;font-weight:bold;letter-spacing:.15em;")
        layout.addWidget(hdr)

        # filter type
        type_lbl = QLabel("Filter type")
        type_lbl.setStyleSheet(f"color:{MUTED};font-size:9px;")
        layout.addWidget(type_lbl)
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["Average", "Gaussian", "Sobel", "Prewitt", "Median"])
        layout.addWidget(self._filter_combo)

        # sigma (Gaussian only)
        self._sigma_row = QWidget()
        sr = QHBoxLayout(self._sigma_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sr.addWidget(QLabel("σ", styleSheet=f"color:{MUTED};font-size:9px;"))
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(0.1, 20.0)
        self._sigma_spin.setValue(1.5)
        self._sigma_spin.setSingleStep(0.1)
        sr.addWidget(self._sigma_spin)
        self._sigma_row.hide()
        layout.addWidget(self._sigma_row)

        # edge output row (Sobel/Prewitt)
        self._edge_row = QWidget()
        er = QHBoxLayout(self._edge_row)
        er.setContentsMargins(0, 0, 0, 0)
        er.addWidget(QLabel("Output", styleSheet=f"color:{MUTED};font-size:9px;"))
        self._edge_group = QButtonGroup(self)
        for label in ("H", "V", "Mag"):
            rb = QRadioButton(label)
            rb.setStyleSheet(f"color:{TEXT};font-size:9px;")
            self._edge_group.addButton(rb)
            er.addWidget(rb)
        self._edge_group.buttons()[2].setChecked(True)
        self._edge_row.hide()
        layout.addWidget(self._edge_row)

        # kernel size
        ksz_lbl = QLabel("Kernel size")
        ksz_lbl.setStyleSheet(f"color:{MUTED};font-size:9px;")
        layout.addWidget(ksz_lbl)

        ksz_row = QWidget()
        krl = QHBoxLayout(ksz_row)
        krl.setContentsMargins(0, 0, 0, 0)
        krl.setSpacing(4)
        self._ksz_group = QButtonGroup(self)
        self._custom_ksz: int | None = None
        for sz in (3, 5, 7, 9):
            rb = QRadioButton(f"{sz}×{sz}")
            rb.setProperty("ksz", sz)
            rb.setStyleSheet(f"color:{TEXT};font-size:9px;")
            self._ksz_group.addButton(rb)
            krl.addWidget(rb)
            if sz == 3:
                rb.setChecked(True)
        self.kernel_btn = QPushButton("⊞")
        self.kernel_btn.setFixedSize(24, 24)
        self.kernel_btn.setStyleSheet(btn_style('ghost'))
        krl.addWidget(self.kernel_btn)
        layout.addWidget(ksz_row)

        # kernel preview
        prev_lbl = QLabel("Kernel preview")
        prev_lbl.setStyleSheet(f"color:{MUTED};font-size:9px;")
        layout.addWidget(prev_lbl)

        self._preview_container = QWidget()
        self._preview_container.setStyleSheet(f"background:{BG};border:1px solid {BORDER};")
        self._preview_container.setFixedHeight(80)
        self._preview_grid = QGridLayout(self._preview_container)
        self._preview_grid.setContentsMargins(4, 4, 4, 4)
        self._preview_grid.setSpacing(2)
        layout.addWidget(self._preview_container)

        self._preview_cells: list = []

        # apply button
        self.apply_btn = QPushButton("Apply Filter")
        self.apply_btn.setStyleSheet(btn_style('primary'))
        layout.addWidget(self.apply_btn)

        layout.addStretch()

        # connect
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)
        self._sigma_spin.valueChanged.connect(self._update_kernel_preview)
        self._ksz_group.buttonClicked.connect(self._update_kernel_preview)
        self._on_filter_changed("Average")

    # ------------------------------------------------------------------ slots

    def _on_filter_changed(self, name: str):
        self._sigma_row.setVisible(name == "Gaussian")
        self._edge_row.setVisible(name in ("Sobel", "Prewitt"))
        self._update_kernel_preview()

    def _update_kernel_preview(self):
        sz = self._get_current_kernel_size()
        sigma = self._sigma_spin.value()
        ft = self._filter_combo.currentText()
        vals = self._build_kernel_values(sz, sigma, ft)

        # rebuild grid
        for cell in self._preview_cells:
            cell.setParent(None)
        self._preview_cells.clear()

        max_v = max((abs(v) for v in vals), default=1.0) or 1.0
        cell_h = max(8, min(22, 70 // sz))

        for row in range(sz):
            for col in range(sz):
                v = vals[row * sz + col]
                intensity = abs(v) / max_v
                if v >= 0:
                    r, g, b = (
                        int(0x11 + (0xc8 - 0x11) * intensity),
                        int(0x12 + (0xf1 - 0x12) * intensity),
                        int(0x10 + (0x35 - 0x10) * intensity),
                    )
                else:
                    r, g, b = (
                        int(0x11 + (0xff - 0x11) * intensity),
                        int(0x12 + (0x4d - 0x12) * intensity),
                        int(0x10 + (0x3a - 0x10) * intensity),
                    )
                cell = QLabel(f"{v:.2f}" if sz <= 5 else "")
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setFixedHeight(cell_h)
                cell.setStyleSheet(
                    f"background:rgb({r},{g},{b});color:{TEXT};font-size:7px;border:none;"
                )
                self._preview_grid.addWidget(cell, row, col)
                self._preview_cells.append(cell)

    def _build_kernel_values(self, sz: int, sigma: float, filter_type: str) -> list:
        n = sz * sz
        if filter_type == "Average":
            v = 1.0 / n
            return [v] * n
        if filter_type == "Gaussian":
            cx = cy = sz // 2
            vals = []
            total = 0.0
            for r in range(sz):
                for c in range(sz):
                    x, y = c - cx, r - cy
                    v = math.exp(-(x * x + y * y) / (2 * sigma * sigma))
                    vals.append(v)
                    total += v
            return [v / total for v in vals]
        if filter_type == "Sobel":
            base = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
            if sz == 3:
                return base
            out = [0.0] * n
            offset_r = (sz - 3) // 2
            offset_c = (sz - 3) // 2
            for i in range(3):
                for j in range(3):
                    out[(i + offset_r) * sz + j + offset_c] = float(base[i * 3 + j])
            return out
        if filter_type == "Prewitt":
            base = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
            if sz == 3:
                return base
            out = [0.0] * n
            offset_r = (sz - 3) // 2
            offset_c = (sz - 3) // 2
            for i in range(3):
                for j in range(3):
                    out[(i + offset_r) * sz + j + offset_c] = float(base[i * 3 + j])
            return out
        return [0.0] * n  # Median — non-linear, no kernel

    def _get_current_kernel_size(self) -> int:
        if self._custom_ksz is not None:
            return self._custom_ksz
        for btn in self._ksz_group.buttons():
            if btn.isChecked():
                return btn.property("ksz")
        return 3

    @wrap_errors
    def on_apply_clicked(self, image: np.ndarray):
        validate_grayscale(image)
        ft = self._filter_combo.currentText()
        sz = self._get_current_kernel_size()
        sigma = self._sigma_spin.value()

        if ft == "Average":
            from processing.spatial.smoothing import average_filter
            result = average_filter(image, sz)
            op_name = f"Average {sz}×{sz}"
        elif ft == "Gaussian":
            from processing.spatial.smoothing import gaussian_filter
            result = gaussian_filter(image, sz, sigma)
            op_name = f"Gaussian {sz}×{sz} σ={sigma:.1f}"
        elif ft == "Sobel":
            from processing.spatial.edge_detection import sobel
            gx, gy, mag = sobel(image)
            checked = self._edge_group.checkedButton()
            label = checked.text() if checked else "Mag"
            result = {"H": gx, "V": gy, "Mag": mag}.get(label, mag)
            op_name = f"Sobel-{label}"
        elif ft == "Prewitt":
            from processing.spatial.edge_detection import prewitt
            gx, gy, mag = prewitt(image)
            checked = self._edge_group.checkedButton()
            label = checked.text() if checked else "Mag"
            result = {"H": gx, "V": gy, "Mag": mag}.get(label, mag)
            op_name = f"Prewitt-{label}"
        elif ft == "Median":
            from processing.spatial.median_filter import median_filter
            result = median_filter(image, sz)
            op_name = f"Median {sz}×{sz}"
        else:
            return

        result = normalize_to_uint8(result)
        self.filter_applied.emit(op_name, result)

    @wrap_errors
    def open_kernel_modal(self, image: np.ndarray):
        validate_grayscale(image)
        dlg = KernelEditorDialog(sigma=self._sigma_spin.value(), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            kernel = dlg.get_kernel()
            from processing.spatial.convolution import convolve2d
            raw = convolve2d(image.astype(np.float64), kernel)
            result = normalize_to_uint8(raw)
            self.filter_applied.emit(f"Custom {kernel.shape[0]}×{kernel.shape[1]}", result)


# ---------------------------------------------------------------------------

class KernelEditorDialog(QDialog):
    """Full-screen modal for custom kernel editing."""

    _SIZES = [3, 5, 7, 9, 11, 15]

    def __init__(self, sigma: float = 1.5, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Kernel Editor")
        self.setMinimumSize(480, 520)
        self.setStyleSheet(
            f"background:{BG};color:{TEXT};"
            f"font-family:'JetBrains Mono','Fira Code',Consolas,monospace;font-size:11px;"
        )
        self._sigma = sigma
        self._current_sz = 5
        self._cells: list = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # size selector
        sz_row = QWidget()
        srl = QHBoxLayout(sz_row)
        srl.setContentsMargins(0, 0, 0, 0)
        srl.addWidget(QLabel("Size:", styleSheet=f"color:{MUTED};"))
        self._sz_group = QButtonGroup(self)
        for sz in self._SIZES:
            rb = QRadioButton(f"{sz}")
            rb.setProperty("sz", sz)
            rb.setStyleSheet(f"color:{TEXT};font-size:10px;")
            self._sz_group.addButton(rb)
            srl.addWidget(rb)
            if sz == 5:
                rb.setChecked(True)
        srl.addStretch()
        layout.addWidget(sz_row)

        # scroll area for grid
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(f"QScrollArea{{background:{BG};border:1px solid {BORDER};}}")
        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet(f"background:{BG};")
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(3)
        self._scroll.setWidget(self._grid_widget)
        layout.addWidget(self._scroll)

        # sum label
        self._sum_label = QLabel()
        self._sum_label.setStyleSheet(f"color:{MUTED};font-size:10px;")
        layout.addWidget(self._sum_label)

        # presets
        preset_row = QWidget()
        prl = QHBoxLayout(preset_row)
        prl.setContentsMargins(0, 0, 0, 0)
        prl.setSpacing(6)
        for label, fn in [("Fill Gaussian", self.fill_gaussian),
                          ("Fill Laplacian", self.fill_laplacian),
                          ("Clear", self.fill_zero)]:
            btn = QPushButton(label)
            btn.setStyleSheet(btn_style('ghost'))
            btn.clicked.connect(fn)
            prl.addWidget(btn)
        layout.addWidget(preset_row)

        # accept / cancel
        btn_row = QWidget()
        brl = QHBoxLayout(btn_row)
        brl.setContentsMargins(0, 0, 0, 0)
        brl.setSpacing(6)
        brl.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setStyleSheet(btn_style('ghost'))
        cancel.clicked.connect(self.reject)
        brl.addWidget(cancel)
        accept = QPushButton("Apply Kernel")
        accept.setStyleSheet(btn_style('primary'))
        accept.clicked.connect(self.accept)
        brl.addWidget(accept)
        layout.addWidget(btn_row)

        self._sz_group.buttonClicked.connect(lambda b: self._build_grid(b.property("sz")))
        self._build_grid(5)

    def _build_grid(self, sz: int):
        self._current_sz = sz
        for c in self._cells:
            c.setParent(None)
        self._cells.clear()
        vals = self._gaussian_vals(sz)
        for r in range(sz):
            for c in range(sz):
                cell = QLineEdit(f"{vals[r * sz + c]:.4f}")
                cell.setFixedSize(60, 26)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setStyleSheet(
                    f"background:{INPUT};border:1px solid {BORDER};color:{TEXT};"
                    f"font-size:9px;border-radius:1px;"
                )
                cell.textChanged.connect(self._update_sum)
                self._grid_layout.addWidget(cell, r, c)
                self._cells.append(cell)
        self._update_sum()

    def _gaussian_vals(self, sz: int) -> list:
        sigma = self._sigma
        cx = cy = sz // 2
        vals, total = [], 0.0
        for r in range(sz):
            for c in range(sz):
                x, y = c - cx, r - cy
                v = math.exp(-(x * x + y * y) / (2 * sigma * sigma))
                vals.append(v)
                total += v
        return [v / total for v in vals]

    def _update_sum(self):
        total = 0.0
        for cell in self._cells:
            try:
                total += float(cell.text())
            except ValueError:
                pass
        if abs(total) < 1e-6:
            self._sum_label.setText(f"sum = {total:.3f}  ·  zero-sum (edge detection)")
        elif abs(total - 1.0) < 1e-3:
            self._sum_label.setText(f"sum = {total:.3f}  ·  normalized (smoothing)")
        else:
            self._sum_label.setText(f"sum = {total:.3f}  ·  not normalized")

    def fill_gaussian(self):
        vals = self._gaussian_vals(self._current_sz)
        for cell, v in zip(self._cells, vals):
            cell.setText(f"{v:.4f}")

    def fill_laplacian(self):
        sz = self._current_sz
        n = sz * sz
        for i, cell in enumerate(self._cells):
            cell.setText(f"{float(sz * sz - 1) if i == n // 2 else -1.0:.4f}")

    def fill_zero(self):
        for cell in self._cells:
            cell.setText("0.0000")

    def get_kernel(self) -> np.ndarray:
        sz = self._current_sz
        vals = []
        for cell in self._cells:
            try:
                vals.append(float(cell.text()))
            except ValueError:
                vals.append(0.0)
        return np.array(vals, dtype=np.float64).reshape(sz, sz)
