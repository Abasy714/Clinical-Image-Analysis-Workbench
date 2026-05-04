# STATUS: SPATIAL DOMAIN WIRED
"""
Control panel for spatial filtering operations.
Allows the user to select kernel size and apply average, Gaussian, Sobel/Prewitt, or median filters.
"""

import math
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
                              QLabel, QPushButton, QDoubleSpinBox, QDialog,
                              QGridLayout, QLineEdit, QScrollArea, QButtonGroup,
                              QRadioButton, QFrame, QSizePolicy)
from PyQt6.QtCore import pyqtSignal, Qt, QThread
from PyQt6.QtGui import QColor

from gui.styles import (BG, PANEL, PANEL2, INPUT, BORDER, BORDER2,
                        ACCENT, ACCENT2, ACCENT_DIM, TEXT, MUTED, MUTED2, btn_style,
                        HEADER_SS, FIELD_SS, COMBO_SS, SPINBOX_SS, APPLY_BTN_SS)
from utils import (validate_grayscale, normalize_to_uint8, wrap_errors, show_error_dialog,)


class FilterWorker(QThread):
    """Runs a filter function in a background thread."""
    finished = pyqtSignal(str, object)
    error = pyqtSignal(str)

    def __init__(self, fn, op_name: str, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._op_name = op_name

    def run(self):
        try:
            result = self._fn()
            self.finished.emit(self._op_name, result)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class GeometricWorker(QThread):
    """Runs a geometric transform in a background thread."""
    finished = pyqtSignal(str, object)
    error = pyqtSignal(str)

    def __init__(self, fn, op_name: str, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._op_name = op_name

    def run(self):
        try:
            result = self._fn()
            self.finished.emit(self._op_name, result)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class FilterPanel(QWidget):
    filter_applied = pyqtSignal(str, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_image: np.ndarray | None = None
        self._custom_kernel: np.ndarray | None = None
        self._using_custom_kernel: bool = False
        self.setStyleSheet(f"background:{PANEL};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # section label
        hdr = QLabel("SPATIAL FILTER")
        hdr.setStyleSheet(HEADER_SS)
        layout.addWidget(hdr)

        # filter type
        type_lbl = QLabel("Filter type")
        type_lbl.setStyleSheet(FIELD_SS)
        layout.addWidget(type_lbl)
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["Average", "Gaussian", "Sobel", "Prewitt", "Median"])
        self._filter_combo.setStyleSheet(COMBO_SS)
        layout.addWidget(self._filter_combo)

        # sigma (Gaussian only)
        self._sigma_row = QWidget()
        sr = QHBoxLayout(self._sigma_row)
        sr.setContentsMargins(0, 0, 0, 0)
        sigma_lbl = QLabel("σ")
        sigma_lbl.setStyleSheet(FIELD_SS)
        sr.addWidget(sigma_lbl)
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(0.1, 20.0)
        self._sigma_spin.setValue(1.5)
        self._sigma_spin.setSingleStep(0.1)
        self._sigma_spin.setStyleSheet(SPINBOX_SS)
        sr.addWidget(self._sigma_spin)
        self._sigma_row.hide()
        layout.addWidget(self._sigma_row)

        # edge output row (Sobel/Prewitt) — styled toggle buttons
        self._edge_row = QWidget()
        er = QVBoxLayout(self._edge_row)
        er.setContentsMargins(0, 4, 0, 0)
        er.setSpacing(4)
        output_lbl = QLabel("OUTPUT")
        output_lbl.setStyleSheet(FIELD_SS)
        er.addWidget(output_lbl)
        edge_btn_row = QWidget()
        edge_btn_layout = QHBoxLayout(edge_btn_row)
        edge_btn_layout.setSpacing(3)
        edge_btn_layout.setContentsMargins(0, 0, 0, 0)
        self._edge_btns: dict = {}
        for key in ("H", "V", "Mag"):
            btn = QPushButton(key)
            btn.setCheckable(True)
            btn.setFixedHeight(26)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background: #252623;
                    color: #6b6f65;
                    border: 1px solid #353730;
                    border-radius: 2px;
                    font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
                    font-size: 10px;
                    font-weight: bold;
                    padding: 0 10px;
                }
                QPushButton:checked {
                    background: #1a2208;
                    color: #c8f135;
                    border-color: #6a8a10;
                }
                QPushButton:hover:!checked {
                    color: #eceee8;
                    border-color: #484b44;
                    background: #2c2e2a;
                }
            """)
            btn.clicked.connect(lambda checked, k=key: self._on_edge_btn_clicked(k))
            self._edge_btns[key] = btn
            edge_btn_layout.addWidget(btn)
        self._edge_btns["Mag"].setChecked(True)
        er.addWidget(edge_btn_row)
        self._edge_row.hide()
        layout.addWidget(self._edge_row)

        # kernel size
        self._ksz_lbl = QLabel("Kernel size")
        self._ksz_lbl.setStyleSheet(FIELD_SS)
        layout.addWidget(self._ksz_lbl)

        self._custom_ksz: int | None = None
        self._kernel_size_widget = QWidget()
        krl = self._build_kernel_size_row()
        self._kernel_size_widget.setLayout(krl)
        layout.addWidget(self._kernel_size_widget)

        self._kernel_fixed_label = QLabel("3×3  (fixed for edge operators)")
        self._kernel_fixed_label.setStyleSheet("""
            QLabel {
                color: #6b6f65;
                font-family: 'JetBrains Mono', Consolas, monospace;
                font-size: 10px;
                font-style: italic;
                padding: 4px 0;
            }
        """)
        self._kernel_fixed_label.setVisible(False)
        layout.addWidget(self._kernel_fixed_label)

        # kernel preview
        self._kernel_preview_label = QLabel("KERNEL PREVIEW")
        self._kernel_preview_label.setStyleSheet(FIELD_SS)
        layout.addWidget(self._kernel_preview_label)

        self._kernel_preview_container = QWidget()
        self._kernel_preview_container.setStyleSheet("background:#111210;")
        self._kernel_grid_layout = QGridLayout(self._kernel_preview_container)
        self._kernel_grid_layout.setContentsMargins(4, 4, 4, 4)
        self._kernel_grid_layout.setSpacing(2)

        self._kernel_scroll = QScrollArea()
        self._kernel_scroll.setWidget(self._kernel_preview_container)
        self._kernel_scroll.setWidgetResizable(True)
        self._kernel_scroll.setFixedHeight(180)
        self._kernel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._kernel_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._kernel_scroll.setStyleSheet("""
            QScrollArea {
                background: #111210;
                border: 1px solid #2c2e2a;
                border-radius: 2px;
            }
            QScrollBar:horizontal { background: #111210; height: 5px; }
            QScrollBar::handle:horizontal { background: #353730; border-radius: 2px; min-width: 16px; }
            QScrollBar:vertical { background: #111210; width: 5px; }
            QScrollBar::handle:vertical { background: #353730; border-radius: 2px; min-height: 16px; }
            QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
        """)
        layout.addWidget(self._kernel_scroll)

        # apply button
        self.apply_btn = QPushButton("Apply Filter")
        self.apply_btn.setStyleSheet(APPLY_BTN_SS)
        layout.addWidget(self.apply_btn)

        # separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color:{BORDER};")
        layout.addWidget(sep)

        # geometric header
        geo_hdr = QLabel("GEOMETRIC TRANSFORM")
        geo_hdr.setStyleSheet(HEADER_SS)
        layout.addWidget(geo_hdr)

        # rotation row
        rot_row = QWidget()
        rrl = QHBoxLayout(rot_row)
        rrl.setContentsMargins(0, 0, 0, 0)
        rrl.setSpacing(6)
        angle_lbl = QLabel("Angle °")
        angle_lbl.setStyleSheet(FIELD_SS)
        rrl.addWidget(angle_lbl)
        self._angle_spin = QDoubleSpinBox()
        self._angle_spin.setRange(-360.0, 360.0)
        self._angle_spin.setValue(45.0)
        self._angle_spin.setSingleStep(1.0)
        self._angle_spin.setStyleSheet(SPINBOX_SS)
        self._angle_spin.setMinimumWidth(80)
        self._angle_spin.setMaximumWidth(120)
        rrl.addWidget(self._angle_spin, 1)
        self._rotate_btn = QPushButton("Rotate")
        self._rotate_btn.setStyleSheet(APPLY_BTN_SS)
        self._rotate_btn.setFixedWidth(80)
        self._rotate_btn.setFixedHeight(28)
        rrl.addWidget(self._rotate_btn)
        layout.addWidget(rot_row)

        # shear row
        shear_row = QWidget()
        shl = QHBoxLayout(shear_row)
        shl.setContentsMargins(0, 0, 0, 0)
        shl.setSpacing(4)
        sx_lbl = QLabel("Sx")
        sx_lbl.setFixedWidth(16)
        sx_lbl.setStyleSheet(FIELD_SS)
        shl.addWidget(sx_lbl)
        self._shear_x_spin = QDoubleSpinBox()
        self._shear_x_spin.setRange(-2.0, 2.0)
        self._shear_x_spin.setValue(0.3)
        self._shear_x_spin.setSingleStep(0.05)
        self._shear_x_spin.setStyleSheet(SPINBOX_SS)
        self._shear_x_spin.setMinimumWidth(60)
        self._shear_x_spin.setMaximumWidth(90)
        shl.addWidget(self._shear_x_spin, 1)
        sy_lbl = QLabel("Sy")
        sy_lbl.setFixedWidth(16)
        sy_lbl.setStyleSheet(FIELD_SS)
        shl.addWidget(sy_lbl)
        self._shear_y_spin = QDoubleSpinBox()
        self._shear_y_spin.setRange(-2.0, 2.0)
        self._shear_y_spin.setValue(0.0)
        self._shear_y_spin.setSingleStep(0.05)
        self._shear_y_spin.setStyleSheet(SPINBOX_SS)
        self._shear_y_spin.setMinimumWidth(60)
        self._shear_y_spin.setMaximumWidth(90)
        shl.addWidget(self._shear_y_spin, 1)
        self._shear_btn = QPushButton("Shear")
        self._shear_btn.setStyleSheet(APPLY_BTN_SS)
        self._shear_btn.setFixedWidth(80)
        self._shear_btn.setFixedHeight(28)
        shl.addWidget(self._shear_btn)
        layout.addWidget(shear_row)

        layout.addStretch()

        # connect
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)
        self._sigma_spin.valueChanged.connect(self._update_kernel_preview)
        self._rotate_btn.clicked.connect(self._apply_rotation)
        self._shear_btn.clicked.connect(self._apply_shearing)
        self._on_filter_changed("Average")

    # ------------------------------------------------------------------ slots

    def _on_filter_changed(self, name: str):
        self._sigma_row.setVisible(name == "Gaussian")
        is_edge = name in ("Sobel", "Prewitt")
        self._edge_row.setVisible(is_edge)
        if is_edge:
            self._ksz_lbl.setVisible(False)
            self._kernel_size_widget.setVisible(False)
            self._kernel_fixed_label.setVisible(True)
            self._current_kernel_size = 3
        else:
            self._ksz_lbl.setVisible(True)
            self._kernel_size_widget.setVisible(True)
            self._kernel_fixed_label.setVisible(False)
        self._update_kernel_preview()

    def _build_kernel_size_row(self) -> QHBoxLayout:
        self._kernel_size_btns = {}
        row = QHBoxLayout()
        row.setSpacing(3)
        row.setContentsMargins(0, 0, 0, 0)

        for sz in [3, 5, 7, 9]:
            btn = QPushButton(f"{sz}×{sz}")
            btn.setCheckable(True)
            btn.setFixedHeight(26)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background: #252623;
                    color: #6b6f65;
                    border: 1px solid #353730;
                    border-radius: 2px;
                    font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
                    font-size: 10px;
                    font-weight: bold;
                    padding: 0 6px;
                }
                QPushButton:checked {
                    background: #1a2208;
                    color: #c8f135;
                    border-color: #6a8a10;
                }
                QPushButton:hover:!checked {
                    color: #eceee8;
                    border-color: #484b44;
                    background: #2c2e2a;
                }
            """)
            btn.clicked.connect(lambda checked, s=sz: self._on_kernel_size_changed(s))
            self._kernel_size_btns[sz] = btn
            row.addWidget(btn)

        # Default selection: 3×3
        self._kernel_size_btns[3].setChecked(True)
        self._current_kernel_size = 3

        # More / custom kernel button
        more_btn = QPushButton("⊞")
        more_btn.setFixedSize(30, 26)
        more_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        more_btn.setToolTip("Open custom kernel editor")
        more_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #6a8a10;
                border: 1px dashed #6a8a10;
                border-radius: 2px;
                font-size: 14px;
                padding: 0;
            }
            QPushButton:hover {
                color: #c8f135;
                border-color: #c8f135;
            }
        """)
        more_btn.clicked.connect(self._open_kernel_modal)
        self.kernel_btn = more_btn
        row.addWidget(more_btn)

        return row

    def _on_kernel_size_changed(self, size: int):
        self._current_kernel_size = size
        for sz, btn in self._kernel_size_btns.items():
            btn.setChecked(sz == size)
        self._update_kernel_preview()

    def _open_kernel_modal(self):
        pass  # main_window.py handles modal opening via kernel_btn.clicked signal

    def set_current_image(self, image: np.ndarray):
        self._current_image = image

    def _on_edge_btn_clicked(self, key: str):
        for k, btn in self._edge_btns.items():
            btn.setChecked(k == key)
        self._update_kernel_preview()

    def _get_edge_output(self) -> str:
        for key, btn in self._edge_btns.items():
            if btn.isChecked():
                return key
        return "Mag"

    def _get_cell_size(self, kernel_size: int) -> tuple:
        """
        Compute cell dimensions that fit within the scroll area.
        Scroll area usable width ≈ 240px (panel 280 - margins 24 - scroll 8).
        """
        usable = 240
        raw_w = usable // kernel_size
        cell_w = max(16, min(52, raw_w))
        cell_h = max(16, min(32, cell_w - 4))
        return (cell_w, cell_h)

    def _build_kernel_cell(self, value: float, max_val: float,
                            all_equal: bool = False,
                            cell_w: int = 36, cell_h: int = 24) -> QLabel:
        # Choose display format based on cell width
        if cell_w < 22:
            text = f"{value:.1f}"
        elif cell_w < 30:
            text = f"{value:.2f}"
        else:
            text = f"{value:.3f}"

        cell = QLabel(text)
        cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cell.setFixedSize(cell_w, cell_h)

        # Safe font size — never below 7
        font_size = max(7, min(9, cell_w // 4))

        # Intensity calculation
        if max_val == 0 or value == 0:
            intensity = 0.0
        elif all_equal:
            intensity = 0.4
        else:
            intensity = abs(value) / max_val

        if value > 0:
            r = int(17 + intensity * (200 - 17))
            g = int(18 + intensity * (241 - 18))
            b = int(16 + intensity * (53  - 16))
            bg = f"rgb({r},{g},{b})"
            text_color = "#0d1002" if intensity > 0.4 else "#6b6f65"
        elif value < 0:
            r = int(17 + intensity * (255 - 17))
            g = int(18 + intensity * (77  - 18))
            b = int(16 + intensity * (58  - 16))
            bg = f"rgb({r},{g},{b})"
            text_color = "#eceee8" if intensity > 0.4 else "#ff4d3a"
        else:
            bg = "#1e1f1d"
            text_color = "#4a4d46"

        cell.setStyleSheet(f"""
            QLabel {{
                background-color: {bg};
                color: {text_color};
                border: 1px solid #2c2e2a;
                border-radius: 1px;
                font-family: 'JetBrains Mono', Consolas, monospace;
                font-size: {font_size}px;
                font-weight: bold;
            }}
        """)
        return cell

    def _update_kernel_preview(self):
        """Rebuild kernel preview grid with current filter settings."""
        filter_type = self._filter_combo.currentText()

        # Determine values + label suffix
        if self._using_custom_kernel and self._custom_kernel is not None:
            sz = self._custom_kernel.shape[0]
            values = self._custom_kernel.flatten().tolist()
            label_text = f"KERNEL PREVIEW  {sz}×{sz} · Custom"
        else:
            sz = self._get_current_kernel_size()
            sigma = self._sigma_spin.value()
            values = self._build_kernel_values(sz, sigma, filter_type)
            label_text = f"KERNEL PREVIEW  {sz}×{sz} · {filter_type}"
            if filter_type == "Gaussian":
                label_text += f" σ={sigma:.1f}"
            elif filter_type in ("Sobel", "Prewitt"):
                label_text += f" · {self._get_edge_output()}"

        max_val = max(abs(v) for v in values) if values else 1.0
        if max_val == 0:
            max_val = 1.0
        unique = set(round(v, 6) for v in values)
        all_equal = len(unique) == 1 and max_val > 0

        cell_w, cell_h = self._get_cell_size(sz)
        self._kernel_preview_label.setText(label_text)

        # Clear existing grid completely
        while self._kernel_grid_layout.count():
            item = self._kernel_grid_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        # Set grid column minimum widths so cells don't collapse
        for col in range(sz):
            self._kernel_grid_layout.setColumnMinimumWidth(col, cell_w)

        # Build new cells
        for idx, v in enumerate(values):
            row = idx // sz
            col = idx % sz
            cell = self._build_kernel_cell(v, max_val, all_equal, cell_w, cell_h)
            self._kernel_grid_layout.addWidget(cell, row, col)

        # Force container resize to fit grid
        self._kernel_preview_container.adjustSize()
        self._kernel_scroll.updateGeometry()

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
        if filter_type in ("Sobel", "Prewitt"):
            edge = self._get_edge_output()  # 'H', 'V', or 'Mag'
            if filter_type == "Sobel":
                Kx = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
                Ky = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
            else:  # Prewitt
                Kx = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
                Ky = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            # H = horizontal edges = Ky kernel; V = vertical edges = Kx; Mag shows Kx as representative
            base = Ky if edge == "H" else Kx
            return [float(v) for v in base]
        return [0.0] * n  # Median — non-linear, no kernel

    def _get_current_kernel_size(self) -> int:
        if self._custom_ksz is not None:
            return self._custom_ksz
        return self._current_kernel_size

    def on_apply_clicked(self, image: np.ndarray):
        try:
            validate_grayscale(image)
            ft = self._filter_combo.currentText()
            sz = self._get_current_kernel_size()
            sigma = self._sigma_spin.value()
        except Exception as e:
            show_error_dialog("Filter Error", str(e))
            return

        _image = image.copy()
        _sz = sz

        # Custom kernel path — overrides dropdown selection
        if self._using_custom_kernel and self._custom_kernel is not None:
            _kernel = self._custom_kernel.copy()
            _ksz = _kernel.shape[0]
            self._using_custom_kernel = False  # one-shot
            def _fn():
                from processing.spatial import convolve2d
                from utils import normalize_to_uint8 as _n
                raw = convolve2d(_image.astype(np.float64), _kernel)
                return _n(raw)
            _op = f"Custom {_ksz}×{_ksz}"
            self._worker = FilterWorker(_fn, _op, parent=self)
            self._worker.finished.connect(self._on_worker_finished)
            self._worker.error.connect(self._on_worker_error)
            self._worker.start()
            return

        if ft == "Average":
            def _fn():
                from processing.spatial import average_filter
                from utils import normalize_to_uint8 as _n
                return _n(average_filter(_image, _sz))
            _op = f"Average {_sz}×{_sz}"

        elif ft == "Gaussian":
            _sigma = float(sigma)
            def _fn():
                from processing.spatial import gaussian_filter
                from utils import normalize_to_uint8 as _n
                return _n(gaussian_filter(_image, _sz, _sigma))
            _op = f"Gaussian {_sz}×{_sz} σ={_sigma:.1f}"

        elif ft == "Sobel":
            _edge = self._get_edge_output()
            def _fn():
                from processing.spatial import sobel
                from utils import normalize_to_uint8 as _n
                gx, gy, mag = sobel(_image)
                return _n({"H": gx, "V": gy}.get(_edge, mag))
            _op = f"Sobel-{_edge}"

        elif ft == "Prewitt":
            _edge = self._get_edge_output()
            def _fn():
                from processing.spatial import prewitt
                from utils import normalize_to_uint8 as _n
                gx, gy, mag = prewitt(_image)
                return _n({"H": gx, "V": gy}.get(_edge, mag))
            _op = f"Prewitt-{_edge}"

        elif ft == "Median":
            def _fn():
                from processing.spatial import median_filter
                from utils import normalize_to_uint8 as _n
                return _n(median_filter(_image, _sz))
            _op = f"Median {_sz}×{_sz}"

        else:
            return

        self._worker = FilterWorker(_fn, _op, parent=self)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_worker_finished(self, op_name: str, result):
        if result is not None and isinstance(result, np.ndarray):
            self.filter_applied.emit(op_name, result)

    def _on_worker_error(self, error_msg: str):
        import logging
        logging.getLogger('ciaw').error(f"Filter worker error: {error_msg}")
        show_error_dialog("Filter Error", error_msg)

    def _apply_rotation(self):
        if self._current_image is None:
            return
        _image = self._current_image.copy()
        _angle = self._angle_spin.value()
        def _fn():
            from processing.geometric import rotate_image
            return rotate_image(_image, _angle)
        self._geo_worker = GeometricWorker(_fn, f"Rotate {_angle:.1f}°", parent=self)
        self._geo_worker.finished.connect(self._on_geo_finished)
        self._geo_worker.error.connect(self._on_geo_error)
        self._geo_worker.start()

    def _apply_shearing(self):
        if self._current_image is None:
            return
        _image = self._current_image.copy()
        _sx = self._shear_x_spin.value()
        _sy = self._shear_y_spin.value()
        def _fn():
            from processing.geometric import shear_image
            return shear_image(_image, _sx, _sy)
        self._geo_worker = GeometricWorker(_fn, f"Shear Sx={_sx:.2f} Sy={_sy:.2f}", parent=self)
        self._geo_worker.finished.connect(self._on_geo_finished)
        self._geo_worker.error.connect(self._on_geo_error)
        self._geo_worker.start()

    def _on_geo_finished(self, op_name: str, result):
        if result is not None and isinstance(result, np.ndarray):
            self.filter_applied.emit(op_name, result)

    def _on_geo_error(self, error_msg: str):
        import logging
        logging.getLogger('ciaw').error(f"Geometric worker error: {error_msg}")
        show_error_dialog("Geometric Transform Error", error_msg)

    @wrap_errors
    def open_kernel_modal(self, image: np.ndarray):
        validate_grayscale(image)
        dlg = KernelEditorDialog(sigma=self._sigma_spin.value(), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            kernel = dlg.get_kernel()
            if kernel is None or kernel.size == 0:
                return
            self._custom_kernel = kernel
            self._using_custom_kernel = True
            # Refresh preview to show custom kernel; user must click Apply Filter to apply
            self._update_kernel_preview()


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

        # size selector — styled toggle buttons
        sz_row = QWidget()
        srl = QHBoxLayout(sz_row)
        srl.setContentsMargins(0, 0, 0, 0)
        srl.setSpacing(4)
        size_lbl = QLabel("Size:")
        size_lbl.setStyleSheet(
            f"color:{MUTED};font-family:'JetBrains Mono',Consolas,monospace;"
            f"font-size:10px;font-weight:bold;"
        )
        srl.addWidget(size_lbl)
        self._size_btns = {}
        for sz in self._SIZES:
            btn = QPushButton(str(sz))
            btn.setCheckable(True)
            btn.setFixedSize(32, 26)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background: #252623;
                    color: #6b6f65;
                    border: 1px solid #353730;
                    border-radius: 2px;
                    font-family: 'JetBrains Mono', Consolas, monospace;
                    font-size: 10px;
                    font-weight: bold;
                }
                QPushButton:checked {
                    background: #1a2208;
                    color: #c8f135;
                    border-color: #6a8a10;
                }
                QPushButton:hover:!checked {
                    color: #eceee8;
                    background: #2c2e2a;
                }
            """)
            btn.clicked.connect(lambda checked, s=sz: self._on_size_clicked(s))
            self._size_btns[sz] = btn
            srl.addWidget(btn)
        self._size_btns[5].setChecked(True)
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

        self._build_grid(5)

    def _on_size_clicked(self, sz: int):
        for s, btn in self._size_btns.items():
            btn.setChecked(s == sz)
        self._build_grid(sz)

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
