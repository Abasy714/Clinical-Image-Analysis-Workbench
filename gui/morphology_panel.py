# STATUS: IMPLEMENTED
"""
Panel for binary morphological operations and segmentation on medical images.
Provides binarization, SE selection, basic/advanced morphological ops, and Otsu-based segmentation.
"""

import logging
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QSlider, QLabel, QGridLayout, QButtonGroup,
                              QRadioButton, QFrame, QSizePolicy, QSpinBox,
                              QDoubleSpinBox, QCheckBox)
from PyQt6.QtCore import pyqtSignal, Qt, QThread

from gui.styles import (BG, PANEL, PANEL2, INPUT, BORDER, BORDER2,
                        ACCENT, TEXT, MUTED, MUTED2, btn_style,
                        HEADER_SS, FIELD_SS, APPLY_BTN_SS, SPINBOX_SS)
from utils import (validate_grayscale, binarize, normalize_to_uint8, to_grayscale,
                   to_qpixmap, wrap_errors, show_error_dialog,)

_log = logging.getLogger('ciaw')


class PreviewLabel(QLabel):
    """Small fixed preview used for before/after segmentation comparisons."""

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(86)
        self.setMinimumWidth(112)
        self.setStyleSheet(
            f"background:{BG};border:1px solid {BORDER};border-radius:2px;"
            f"color:{MUTED2};font-size:9px;"
        )
        self.setScaledContents(False)

    def set_array(self, image: np.ndarray | None):
        if image is None:
            self.clear()
            self.setText("No image")
            return
        pixmap = to_qpixmap(normalize_to_uint8(image))
        scaled = pixmap.scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setText("")
        self.setPixmap(scaled)


class MorphologyWorker(QThread):
    result_ready = pyqtSignal(str, np.ndarray)
    error        = pyqtSignal(str)

    def __init__(self, state, op_name: str, se: np.ndarray, threshold: int,
                 is_advanced: bool = False, parent=None):
        super().__init__(parent)
        self._state       = state
        self._op_name     = op_name
        self._se          = se
        self._threshold   = threshold
        self._is_advanced = is_advanced

    def run(self):
        _log.debug("[DEBUG] MorphologyWorker.run() start: op=%s", self._op_name)
        try:
            image = self._state.get_base_image()
            if image is None:
                self.error.emit("No image loaded.")
                return
            image = normalize_to_uint8(to_grayscale(image))
            validate_grayscale(image)
            binary = binarize(image, self._threshold)

            if self._is_advanced:
                from processing.morphology.advanced import apply_advanced_op
                result_bool = apply_advanced_op(self._op_name, binary, self._se)
                result = normalize_to_uint8(result_bool.astype(np.uint8) * 255)
                label = self._op_name.replace("_", " ").title()
            else:
                op = self._op_name
                if op == "Erode":
                    from processing.morphology.erosion_dilation import erode
                    result = normalize_to_uint8(erode(binary, self._se).astype(np.uint8) * 255)
                elif op == "Dilate":
                    from processing.morphology.erosion_dilation import dilate
                    result = normalize_to_uint8(dilate(binary, self._se).astype(np.uint8) * 255)
                elif op == "Open":
                    from processing.morphology.opening_closing import opening
                    result = normalize_to_uint8(opening(binary, self._se).astype(np.uint8) * 255)
                elif op == "Close":
                    from processing.morphology.opening_closing import closing
                    result = normalize_to_uint8(closing(binary, self._se).astype(np.uint8) * 255)
                elif op == "Boundary":
                    from processing.morphology.boundary_extraction import extract_boundary
                    result = normalize_to_uint8(extract_boundary(binary, self._se).astype(np.uint8) * 255)
                else:
                    _log.error("[ERROR] MorphologyWorker: unknown op '%s'", op)
                    return
                label = self._op_name

            _log.debug("[DEBUG] MorphologyWorker.run() emitting: op=%s", label)
            self.result_ready.emit(label, result)
        except Exception as e:
            _log.error("[ERROR] MorphologyWorker: %s", e)
            self.error.emit(str(e))


class SegmentationWorker(QThread):
    result_ready    = pyqtSignal(str, np.ndarray)
    threshold_found = pyqtSignal(int)
    error           = pyqtSignal(str)

    def __init__(self, state, method: str, params: dict, parent=None):
        super().__init__(parent)
        self._state  = state
        self._method = method
        self._params = params

    def run(self):
        _log.debug("[DEBUG] SegmentationWorker.run() start: method=%s", self._method)
        try:
            image = self._state.get_base_image()
            if image is None:
                self.error.emit("No image loaded.")
                return
            image = normalize_to_uint8(to_grayscale(image))

            from processing.segmentation.otsu_threshold import (
                otsu_binarize, adaptive_threshold,
                multi_level_otsu, apply_colormap_overlay,
            )

            if self._method == "otsu":
                binary, t = otsu_binarize(image)
                result = normalize_to_uint8(binary)
                self.threshold_found.emit(int(t))
                _log.debug("[DEBUG] SegmentationWorker emitting: Otsu t=%d", t)
                self.result_ready.emit(f"Otsu t={t}", result)

            elif self._method == "adaptive":
                block_size = self._params.get("block_size", 51)
                C = self._params.get("C", 5.0)
                result = normalize_to_uint8(adaptive_threshold(image, block_size, C))
                _log.debug("[DEBUG] SegmentationWorker emitting: Adaptive b=%d", block_size)
                self.result_ready.emit(
                    f"Adaptive Threshold b={block_size} C={C:.1f}", result
                )

            elif self._method == "multi":
                n_classes     = self._params.get("n_classes", 2)
                color_overlay = self._params.get("color_overlay", False)
                thresholds, label_map = multi_level_otsu(image, n_classes)
                if color_overlay:
                    result = apply_colormap_overlay(image, label_map)
                else:
                    scale = 255 // max(n_classes - 1, 1)
                    result = normalize_to_uint8(
                        (label_map.astype(np.float64) * scale).clip(0, 255)
                    )
                _log.debug("[DEBUG] SegmentationWorker emitting: Multi-Otsu %d-class", n_classes)
                self.result_ready.emit(
                    f"Multi-Otsu {n_classes}-class t={thresholds}", result
                )

        except Exception as e:
            _log.error("[ERROR] SegmentationWorker: %s", e)
            self.error.emit(str(e))

_log = logging.getLogger('ciaw')


class MorphologyWorker(QThread):
    result_ready = pyqtSignal(str, np.ndarray)
    error        = pyqtSignal(str)

    def __init__(self, state, op_name: str, se: np.ndarray, threshold: int,
                 is_advanced: bool = False, parent=None):
        super().__init__(parent)
        self._state       = state
        self._op_name     = op_name
        self._se          = se
        self._threshold   = threshold
        self._is_advanced = is_advanced

    def run(self):
        _log.debug("[DEBUG] MorphologyWorker.run() start: op=%s", self._op_name)
        try:
            image = self._state.get_base_image()
            if image is None:
                self.error.emit("No image loaded.")
                return
            validate_grayscale(image)
            binary = binarize(image, self._threshold)

            if self._is_advanced:
                from processing.morphology.advanced import apply_advanced_op
                result_bool = apply_advanced_op(self._op_name, binary, self._se)
                result = normalize_to_uint8(result_bool.astype(np.uint8) * 255)
                label = self._op_name.replace("_", " ").title()
            else:
                op = self._op_name
                if op == "Erode":
                    from processing.morphology.erosion_dilation import erode
                    result = normalize_to_uint8(erode(binary, self._se).astype(np.uint8) * 255)
                elif op == "Dilate":
                    from processing.morphology.erosion_dilation import dilate
                    result = normalize_to_uint8(dilate(binary, self._se).astype(np.uint8) * 255)
                elif op == "Open":
                    from processing.morphology.opening_closing import opening
                    result = normalize_to_uint8(opening(binary, self._se).astype(np.uint8) * 255)
                elif op == "Close":
                    from processing.morphology.opening_closing import closing
                    result = normalize_to_uint8(closing(binary, self._se).astype(np.uint8) * 255)
                elif op == "Boundary":
                    from processing.morphology.boundary_extraction import extract_boundary
                    result = normalize_to_uint8(extract_boundary(binary, self._se).astype(np.uint8) * 255)
                else:
                    _log.error("[ERROR] MorphologyWorker: unknown op '%s'", op)
                    return
                label = self._op_name

            _log.debug("[DEBUG] MorphologyWorker.run() emitting: op=%s", label)
            self.result_ready.emit(label, result)
        except Exception as e:
            _log.error("[ERROR] MorphologyWorker: %s", e)
            self.error.emit(str(e))


class SegmentationWorker(QThread):
    result_ready    = pyqtSignal(str, np.ndarray)
    threshold_found = pyqtSignal(int)
    error           = pyqtSignal(str)

    def __init__(self, state, method: str, params: dict, parent=None):
        super().__init__(parent)
        self._state  = state
        self._method = method
        self._params = params

    def run(self):
        _log.debug("[DEBUG] SegmentationWorker.run() start: method=%s", self._method)
        try:
            image = self._state.get_base_image()
            if image is None:
                self.error.emit("No image loaded.")
                return

            from processing.segmentation.otsu_threshold import (
                otsu_binarize, adaptive_threshold,
                multi_level_otsu, apply_colormap_overlay,
            )

            if self._method == "otsu":
                binary, t = otsu_binarize(image)
                result = normalize_to_uint8(binary)
                self.threshold_found.emit(int(t))
                _log.debug("[DEBUG] SegmentationWorker emitting: Otsu t=%d", t)
                self.result_ready.emit(f"Otsu t={t}", result)

            elif self._method == "adaptive":
                block_size = self._params.get("block_size", 51)
                C = self._params.get("C", 5.0)
                result = normalize_to_uint8(adaptive_threshold(image, block_size, C))
                _log.debug("[DEBUG] SegmentationWorker emitting: Adaptive b=%d", block_size)
                self.result_ready.emit(
                    f"Adaptive Threshold b={block_size} C={C:.1f}", result
                )

            elif self._method == "multi":
                n_classes     = self._params.get("n_classes", 2)
                color_overlay = self._params.get("color_overlay", False)
                thresholds, label_map = multi_level_otsu(image, n_classes)
                if color_overlay:
                    result = apply_colormap_overlay(image, label_map)
                else:
                    scale = 255 // max(n_classes - 1, 1)
                    result = normalize_to_uint8(
                        (label_map.astype(np.float64) * scale).clip(0, 255)
                    )
                _log.debug("[DEBUG] SegmentationWorker emitting: Multi-Otsu %d-class", n_classes)
                self.result_ready.emit(
                    f"Multi-Otsu {n_classes}-class t={thresholds}", result
                )

        except Exception as e:
            _log.error("[ERROR] SegmentationWorker: %s", e)
            self.error.emit(str(e))

_log = logging.getLogger('ciaw')


class MorphologyWorker(QThread):
    result_ready = pyqtSignal(str, np.ndarray)
    error        = pyqtSignal(str)

    def __init__(self, state, op_name: str, se: np.ndarray, threshold: int,
                 is_advanced: bool = False, parent=None):
        super().__init__(parent)
        self._state       = state
        self._op_name     = op_name
        self._se          = se
        self._threshold   = threshold
        self._is_advanced = is_advanced

    def run(self):
        _log.debug("[DEBUG] MorphologyWorker.run() start: op=%s", self._op_name)
        try:
            image = self._state.get_base_image()
            if image is None:
                self.error.emit("No image loaded.")
                return
            validate_grayscale(image)
            binary = binarize(image, self._threshold)

            if self._is_advanced:
                from processing.morphology.advanced import apply_advanced_op
                result_bool = apply_advanced_op(self._op_name, binary, self._se)
                result = normalize_to_uint8(result_bool.astype(np.uint8) * 255)
                label = self._op_name.replace("_", " ").title()
            else:
                op = self._op_name
                if op == "Erode":
                    from processing.morphology.erosion_dilation import erode
                    result = normalize_to_uint8(erode(binary, self._se).astype(np.uint8) * 255)
                elif op == "Dilate":
                    from processing.morphology.erosion_dilation import dilate
                    result = normalize_to_uint8(dilate(binary, self._se).astype(np.uint8) * 255)
                elif op == "Open":
                    from processing.morphology.opening_closing import opening
                    result = normalize_to_uint8(opening(binary, self._se).astype(np.uint8) * 255)
                elif op == "Close":
                    from processing.morphology.opening_closing import closing
                    result = normalize_to_uint8(closing(binary, self._se).astype(np.uint8) * 255)
                elif op == "Boundary":
                    from processing.morphology.boundary_extraction import extract_boundary
                    result = normalize_to_uint8(extract_boundary(binary, self._se).astype(np.uint8) * 255)
                else:
                    _log.error("[ERROR] MorphologyWorker: unknown op '%s'", op)
                    return
                label = self._op_name

            _log.debug("[DEBUG] MorphologyWorker.run() emitting: op=%s", label)
            self.result_ready.emit(label, result)
        except Exception as e:
            _log.error("[ERROR] MorphologyWorker: %s", e)
            self.error.emit(str(e))


class SegmentationWorker(QThread):
    result_ready    = pyqtSignal(str, np.ndarray)
    threshold_found = pyqtSignal(int)
    error           = pyqtSignal(str)

    def __init__(self, state, method: str, params: dict, parent=None):
        super().__init__(parent)
        self._state  = state
        self._method = method
        self._params = params

    def run(self):
        _log.debug("[DEBUG] SegmentationWorker.run() start: method=%s", self._method)
        try:
            image = self._state.get_base_image()
            if image is None:
                self.error.emit("No image loaded.")
                return

            from processing.segmentation.otsu_threshold import (
                otsu_binarize, adaptive_threshold,
                multi_level_otsu, apply_colormap_overlay,
            )

            if self._method == "otsu":
                binary, t = otsu_binarize(image)
                result = normalize_to_uint8(binary)
                self.threshold_found.emit(int(t))
                _log.debug("[DEBUG] SegmentationWorker emitting: Otsu t=%d", t)
                self.result_ready.emit(f"Otsu t={t}", result)

            elif self._method == "adaptive":
                block_size = self._params.get("block_size", 51)
                C = self._params.get("C", 5.0)
                result = normalize_to_uint8(adaptive_threshold(image, block_size, C))
                _log.debug("[DEBUG] SegmentationWorker emitting: Adaptive b=%d", block_size)
                self.result_ready.emit(
                    f"Adaptive Threshold b={block_size} C={C:.1f}", result
                )

            elif self._method == "multi":
                n_classes     = self._params.get("n_classes", 2)
                color_overlay = self._params.get("color_overlay", False)
                thresholds, label_map = multi_level_otsu(image, n_classes)
                if color_overlay:
                    result = apply_colormap_overlay(image, label_map)
                else:
                    scale = 255 // max(n_classes - 1, 1)
                    result = normalize_to_uint8(
                        (label_map.astype(np.float64) * scale).clip(0, 255)
                    )
                _log.debug("[DEBUG] SegmentationWorker emitting: Multi-Otsu %d-class", n_classes)
                self.result_ready.emit(
                    f"Multi-Otsu {n_classes}-class t={thresholds}", result
                )

        except Exception as e:
            _log.error("[ERROR] SegmentationWorker: %s", e)
            self.error.emit(str(e))


class MorphologyPanel(QWidget):
    morphology_applied   = pyqtSignal(str, np.ndarray)
    segmentation_applied = pyqtSignal(str, np.ndarray)
    display_override     = pyqtSignal(np.ndarray)

    def __init__(self, state=None, parent=None):
        super().__init__(parent)
        self._state              = state
        self._source_image: np.ndarray | None = None
        self._seg_before:   np.ndarray | None = None
        self._seg_result:   np.ndarray | None = None
        self._pending_seg_before: np.ndarray | None = None
        self._seg_showing_result = True
        self._morph_worker: MorphologyWorker | None = None
        self._seg_worker:   SegmentationWorker | None = None

        self.setStyleSheet(f"background:{PANEL};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        hdr = QLabel("MORPHOLOGY")
        hdr.setStyleSheet(HEADER_SS)
        layout.addWidget(hdr)

        # ---- binarization ----
        bin_lbl = QLabel("BINARIZATION")
        bin_lbl.setStyleSheet(HEADER_SS)
        layout.addWidget(bin_lbl)

        slider_row = QWidget()
        srl = QHBoxLayout(slider_row)
        srl.setContentsMargins(0, 0, 0, 0)
        srl.setSpacing(8)

        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(0, 255)
        self._threshold_slider.setValue(128)
        self._threshold_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 3px; background: #353730; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 13px; height: 13px; margin: -5px 0;
                background: #c8f135; border-radius: 7px; border: 2px solid #0d1002;
            }
            QSlider::sub-page:horizontal { background: #c8f135; border-radius: 2px; }
            QSlider::add-page:horizontal  { background: #353730; border-radius: 2px; }
        """)
        srl.addWidget(self._threshold_slider)

        self._thresh_val_lbl = QLabel("128")
        self._thresh_val_lbl.setFixedWidth(30)
        self._thresh_val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._thresh_val_lbl.setStyleSheet(
            f"color:{ACCENT};font-family:'JetBrains Mono',Consolas,monospace;"
            f"font-size:13px;font-weight:bold;"
        )
        srl.addWidget(self._thresh_val_lbl)
        layout.addWidget(slider_row)

        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)

        self._manual_threshold_btn = QPushButton("Apply Manual Threshold")
        self._manual_threshold_btn.setStyleSheet(APPLY_BTN_SS)
        self._manual_threshold_btn.clicked.connect(self._apply_manual_threshold)
        layout.addWidget(self._manual_threshold_btn)

        bin_preview_lbl = QLabel("BINARIZATION PREVIEW")
        bin_preview_lbl.setStyleSheet(FIELD_SS)
        layout.addWidget(bin_preview_lbl)
        bin_preview_grid = QGridLayout()
        bin_preview_grid.setContentsMargins(0, 0, 0, 0)
        bin_preview_grid.setSpacing(4)
        before_bin_lbl = QLabel("Before")
        before_bin_lbl.setStyleSheet(FIELD_SS)
        after_bin_lbl = QLabel("After")
        after_bin_lbl.setStyleSheet(FIELD_SS)
        self._before_bin_preview = PreviewLabel("No image")
        self._after_bin_preview = PreviewLabel("No image")
        bin_preview_grid.addWidget(before_bin_lbl, 0, 0)
        bin_preview_grid.addWidget(after_bin_lbl, 0, 1)
        bin_preview_grid.addWidget(self._before_bin_preview, 1, 0)
        bin_preview_grid.addWidget(self._after_bin_preview, 1, 1)
        layout.addLayout(bin_preview_grid)

        layout.addWidget(_hdiv())

        # ---- structuring element ----
        se_lbl = QLabel("STRUCTURING ELEMENT")
        se_lbl.setStyleSheet(HEADER_SS)
        layout.addWidget(se_lbl)

        shape_row = QWidget()
        shrl = QHBoxLayout(shape_row)
        shrl.setContentsMargins(0, 0, 0, 0)
        shrl.setSpacing(4)
        shape_field_lbl = QLabel("Shape")
        shape_field_lbl.setStyleSheet(FIELD_SS)
        shrl.addWidget(shape_field_lbl)
        self._shape_group = QButtonGroup(self)
        for label in ("Square", "Cross", "Disk"):
            rb = QRadioButton(label)
            rb.setStyleSheet(f"color:{TEXT};font-size:9px;")
            self._shape_group.addButton(rb)
            shrl.addWidget(rb)
            if label == "Square":
                rb.setChecked(True)
        shrl.addStretch()
        layout.addWidget(shape_row)

        size_row = QWidget()
        szrl = QHBoxLayout(size_row)
        szrl.setContentsMargins(0, 0, 0, 0)
        szrl.setSpacing(4)
        size_field_lbl = QLabel("Size")
        size_field_lbl.setStyleSheet(FIELD_SS)
        szrl.addWidget(size_field_lbl)
        self._size_group = QButtonGroup(self)
        for sz in (3, 5, 7):
            rb = QRadioButton(f"{sz}×{sz}")
            rb.setProperty("sz", sz)
            rb.setStyleSheet(f"color:{TEXT};font-size:9px;")
            self._size_group.addButton(rb)
            szrl.addWidget(rb)
            if sz == 3:
                rb.setChecked(True)
        szrl.addStretch()
        layout.addWidget(size_row)

        layout.addWidget(_hdiv())

        # ---- basic operations ----
        ops_lbl = QLabel("OPERATIONS")
        ops_lbl.setStyleSheet(HEADER_SS)
        layout.addWidget(ops_lbl)

        ops_grid = QWidget()
        og = QGridLayout(ops_grid)
        og.setContentsMargins(0, 0, 0, 0)
        og.setSpacing(6)
        for i, (label, op) in enumerate([("Erode", "Erode"), ("Dilate", "Dilate"),
                                          ("Open", "Open"), ("Close", "Close")]):
            btn = QPushButton(label)
            btn.setStyleSheet(btn_style())
            btn.clicked.connect(lambda checked, o=op: self._apply_op(o))
            og.addWidget(btn, i // 2, i % 2)
        layout.addWidget(ops_grid)

        boundary_btn = QPushButton("Extract Boundary")
        boundary_btn.setStyleSheet(APPLY_BTN_SS)
        boundary_btn.clicked.connect(lambda: self._apply_op("Boundary"))
        layout.addWidget(boundary_btn)

        hint = QLabel("Boundary = A − (A ⊖ B)")
        hint.setStyleSheet(f"color:{MUTED2};font-size:9px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        layout.addWidget(_hdiv())

        # ---- advanced operations ----
        adv_lbl = QLabel("ADVANCED OPS")
        adv_lbl.setStyleSheet(HEADER_SS)
        layout.addWidget(adv_lbl)

        adv_grid = QWidget()
        ag = QGridLayout(adv_grid)
        ag.setContentsMargins(0, 0, 0, 0)
        ag.setSpacing(6)
        for i, (label, op) in enumerate([
            ("Gradient", "gradient"),
            ("Top Hat (W)", "white_top_hat"),
            ("Top Hat (B)", "black_top_hat"),
        ]):
            btn = QPushButton(label)
            btn.setStyleSheet(btn_style())
            btn.clicked.connect(lambda checked, o=op: self._apply_advanced(o))
            ag.addWidget(btn, i // 2, i % 2)
        layout.addWidget(adv_grid)

        layout.addWidget(_hdiv())

        # ---- segmentation ----
        seg_hdr = QLabel("SEGMENTATION")
        seg_hdr.setStyleSheet(HEADER_SS)
        layout.addWidget(seg_hdr)

        otsu_row = QWidget()
        orl = QHBoxLayout(otsu_row)
        orl.setContentsMargins(0, 0, 0, 0)
        orl.setSpacing(6)
        otsu_btn = QPushButton("Auto Otsu")
        otsu_btn.setStyleSheet(btn_style())
        otsu_btn.clicked.connect(self._run_otsu)
        orl.addWidget(otsu_btn)
        self._otsu_thresh_lbl = QLabel("t = —")
        self._otsu_thresh_lbl.setStyleSheet(f"color:{ACCENT};font-size:9px;font-weight:bold;")
        orl.addWidget(self._otsu_thresh_lbl)
        orl.addStretch()
        layout.addWidget(otsu_row)

        adapt_row = QWidget()
        arl = QHBoxLayout(adapt_row)
        arl.setContentsMargins(0, 0, 0, 0)
        arl.setSpacing(4)
        adapt_btn = QPushButton("Adaptive")
        adapt_btn.setStyleSheet(btn_style())
        adapt_btn.clicked.connect(self._run_adaptive)
        arl.addWidget(adapt_btn)
        bs_lbl = QLabel("Block")
        bs_lbl.setStyleSheet(FIELD_SS)
        arl.addWidget(bs_lbl)
        self._block_size_spin = QSpinBox()
        self._block_size_spin.setRange(3, 201)
        self._block_size_spin.setValue(51)
        self._block_size_spin.setSingleStep(2)
        self._block_size_spin.setFixedWidth(50)
        self._block_size_spin.setStyleSheet(SPINBOX_SS)
        arl.addWidget(self._block_size_spin)
        c_lbl = QLabel("C")
        c_lbl.setStyleSheet(FIELD_SS)
        arl.addWidget(c_lbl)
        self._c_spin = QDoubleSpinBox()
        self._c_spin.setRange(0.0, 50.0)
        self._c_spin.setValue(5.0)
        self._c_spin.setSingleStep(0.5)
        self._c_spin.setFixedWidth(50)
        self._c_spin.setStyleSheet(SPINBOX_SS)
        arl.addWidget(self._c_spin)
        layout.addWidget(adapt_row)

        multi_row = QWidget()
        mrl = QHBoxLayout(multi_row)
        mrl.setContentsMargins(0, 0, 0, 0)
        mrl.setSpacing(4)
        multi_btn = QPushButton("Multi-class")
        multi_btn.setStyleSheet(btn_style())
        multi_btn.clicked.connect(self._run_multi)
        mrl.addWidget(multi_btn)
        nc_lbl = QLabel("Classes")
        nc_lbl.setStyleSheet(FIELD_SS)
        mrl.addWidget(nc_lbl)
        self._nclasses_spin = QSpinBox()
        self._nclasses_spin.setRange(2, 4)
        self._nclasses_spin.setValue(2)
        self._nclasses_spin.setFixedWidth(40)
        self._nclasses_spin.setStyleSheet(SPINBOX_SS)
        mrl.addWidget(self._nclasses_spin)
        mrl.addStretch()
        layout.addWidget(multi_row)

        self._color_overlay_chk = QCheckBox("Color Overlay")
        self._color_overlay_chk.setStyleSheet(f"color:{TEXT};font-size:9px;")
        layout.addWidget(self._color_overlay_chk)

        seg_preview_lbl = QLabel("SEGMENTATION PREVIEW")
        seg_preview_lbl.setStyleSheet(FIELD_SS)
        layout.addWidget(seg_preview_lbl)
        seg_preview_grid = QGridLayout()
        seg_preview_grid.setContentsMargins(0, 0, 0, 0)
        seg_preview_grid.setSpacing(4)
        before_seg_lbl = QLabel("Before")
        before_seg_lbl.setStyleSheet(FIELD_SS)
        after_seg_lbl = QLabel("After")
        after_seg_lbl.setStyleSheet(FIELD_SS)
        self._before_seg_preview = PreviewLabel("No image")
        self._after_seg_preview = PreviewLabel("No image")
        seg_preview_grid.addWidget(before_seg_lbl, 0, 0)
        seg_preview_grid.addWidget(after_seg_lbl, 0, 1)
        seg_preview_grid.addWidget(self._before_seg_preview, 1, 0)
        seg_preview_grid.addWidget(self._after_seg_preview, 1, 1)
        layout.addLayout(seg_preview_grid)

        self._seg_toggle_btn = QPushButton("Show Original")
        self._seg_toggle_btn.setStyleSheet(btn_style('ghost'))
        self._seg_toggle_btn.setEnabled(False)
        self._seg_toggle_btn.clicked.connect(self._on_seg_toggle_clicked)
        layout.addWidget(self._seg_toggle_btn)

        layout.addStretch()

    # ------------------------------------------------------------------ public

    def set_image(self, image: np.ndarray):
        try:
            if image is None:
                return
            image = normalize_to_uint8(to_grayscale(image))
            validate_grayscale(image)
        except ValueError:
            return
        self._source_image = image
        self._on_threshold_changed(self._threshold_slider.value())

    # ------------------------------------------------------------------ private

    def _on_threshold_changed(self, value: int):
        self._thresh_val_lbl.setText(str(value))
        self._update_binarization_preview()

    def _update_binarization_preview(self):
        if self._source_image is None:
            return
        try:
            source = normalize_to_uint8(to_grayscale(self._source_image))
            binary = binarize(source, self._threshold_slider.value()).astype(np.uint8) * 255
            self._before_bin_preview.set_array(source)
            self._after_bin_preview.set_array(binary)
        except Exception as e:
            _log.error("binarization preview failed: %s", e)

    def _apply_manual_threshold(self):
        if self._state is None:
            show_error_dialog("No state", "Panel not connected to pipeline state.")
            return
        image = self._state.get_base_image()
        if image is None:
            show_error_dialog("No Image", "Load an image before applying manual thresholding.")
            return
        try:
            before = normalize_to_uint8(to_grayscale(image))
            result = binarize(before, self._threshold_slider.value()).astype(np.uint8) * 255
            self._seg_before = before
            self._seg_result = result
            self._seg_showing_result = True
            self._seg_toggle_btn.setEnabled(True)
            self._seg_toggle_btn.setText("Show Before")
            self._before_bin_preview.set_array(before)
            self._after_bin_preview.set_array(result)
            self._before_seg_preview.set_array(before)
            self._after_seg_preview.set_array(result)
            self.segmentation_applied.emit(
                f"Manual Threshold t={self._threshold_slider.value()}",
                result.astype(np.uint8),
            )
        except Exception as e:
            _log.error("manual threshold failed: %s", e)
            show_error_dialog("Threshold Error", str(e))

    def _get_se(self) -> np.ndarray:
        size = 3
        for btn in self._size_group.buttons():
            if btn.isChecked():
                size = btn.property("sz")
                break
        shape_btn = self._shape_group.checkedButton()
        shape = shape_btn.text() if shape_btn else "Square"
        from processing.morphology.structuring_element import get_se
        return get_se(shape, size)

    def _apply_op(self, op_name: str):
        if self._state is None:
            show_error_dialog("No state", "Panel not connected to pipeline state.")
            return
        if self._morph_worker is not None and self._morph_worker.isRunning():
            return
        try:
            se = self._get_se()
        except Exception as e:
            show_error_dialog("SE Error", str(e))
            return
        self._morph_worker = MorphologyWorker(
            self._state, op_name, se, self._threshold_slider.value(),
            is_advanced=False, parent=self
        )
        self._morph_worker.result_ready.connect(self.morphology_applied)
        self._morph_worker.error.connect(
            lambda e: show_error_dialog("Morphology Error", e)
        )
        self._morph_worker.start()

    def _apply_advanced(self, op_name: str):
        if self._state is None:
            show_error_dialog("No state", "Panel not connected to pipeline state.")
            return
        if self._morph_worker is not None and self._morph_worker.isRunning():
            return
        try:
            se = self._get_se()
        except Exception as e:
            show_error_dialog("SE Error", str(e))
            return
        self._morph_worker = MorphologyWorker(
            self._state, op_name, se, self._threshold_slider.value(),
            is_advanced=True, parent=self
        )
        self._morph_worker.result_ready.connect(self.morphology_applied)
        self._morph_worker.error.connect(
            lambda e: show_error_dialog("Advanced Morph Error", e)
        )
        self._morph_worker.start()

    def _run_segmentation(self, method: str, params: dict):
        if self._state is None:
            show_error_dialog("No state", "Panel not connected to pipeline state.")
            return
        if self._seg_worker is not None and self._seg_worker.isRunning():
            return
        base = self._state.get_base_image()
        if base is None:
            show_error_dialog("No Image", "Load an image before running segmentation.")
            return
        self._pending_seg_before = normalize_to_uint8(to_grayscale(base))
        self._before_seg_preview.set_array(self._pending_seg_before)
        self._seg_worker = SegmentationWorker(self._state, method, params, parent=self)
        self._seg_worker.result_ready.connect(self._on_seg_done)
        self._seg_worker.threshold_found.connect(
            lambda t: self._otsu_thresh_lbl.setText(f"t = {t}")
        )
        self._seg_worker.error.connect(
            lambda e: show_error_dialog("Segmentation Error", e)
        )
        self._seg_worker.start()

    def _on_seg_done(self, op_name: str, result: np.ndarray):
        self._seg_before = self._pending_seg_before
        self._seg_result = normalize_to_uint8(result)
        self._seg_showing_result = True
        self._seg_toggle_btn.setEnabled(True)
        self._seg_toggle_btn.setText("Show Before")
        self._before_seg_preview.set_array(self._seg_before)
        self._after_seg_preview.set_array(self._seg_result)
        if op_name.lower().startswith("otsu") or "threshold" in op_name.lower():
            self._before_bin_preview.set_array(self._seg_before)
            self._after_bin_preview.set_array(self._seg_result)
        self.segmentation_applied.emit(op_name, self._seg_result)

    def _on_seg_toggle_clicked(self):
        if self._seg_before is None or self._seg_result is None:
            return
        self._seg_showing_result = not self._seg_showing_result
        if self._seg_showing_result:
            self._seg_toggle_btn.setText("Show Before")
            self.display_override.emit(self._seg_result)
        else:
            self._seg_toggle_btn.setText("Show After")
            self.display_override.emit(self._seg_before)

    def _run_otsu(self):
        self._run_segmentation("otsu", {})

    def _run_adaptive(self):
        block_size = self._block_size_spin.value()
        if block_size % 2 == 0:
            block_size += 1
        self._run_segmentation("adaptive", {
            "block_size": block_size,
            "C": self._c_spin.value(),
        })

    def _run_multi(self):
        self._run_segmentation("multi", {
            "n_classes": self._nclasses_spin.value(),
            "color_overlay": self._color_overlay_chk.isChecked(),
        })


def _hdiv() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"background:#2c2e2a;max-height:1px;")
    return f
