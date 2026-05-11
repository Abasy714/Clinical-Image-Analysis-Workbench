import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QTabWidget, QButtonGroup, QRadioButton,
    QCheckBox, QSlider,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from gui.theme import get as _get_theme
from gui.styles import btn_style, SPINBOX_SS, APPLY_BTN_SS, HEADER_SS, FIELD_SS
from gui.workers import PipelineWorker


# ------------------------------------------------------------------
# Module-level binarization helpers
# ------------------------------------------------------------------

def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1]
                + 0.114 * image[:, :, 2]).astype(np.uint8)
    return image


def _binarize(image: np.ndarray):
    from processing.segmentation.otsu_threshold import otsu_threshold
    gray = _to_gray(image)
    t = otsu_threshold(gray)
    return (gray > t).astype(bool), gray


def _bool_to_uint8(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.uint8) * 255


# ------------------------------------------------------------------
# Morphological operation wrappers (image → uint8 ndarray)
# ------------------------------------------------------------------

def _erode_fn(image, se_shape='square', se_size=3):
    from processing.morphology.structuring_element import get_se
    from processing.morphology.erosion_dilation import erode
    binary, _ = _binarize(image)
    return _bool_to_uint8(erode(binary, get_se(se_shape, se_size)))


def _dilate_fn(image, se_shape='square', se_size=3):
    from processing.morphology.structuring_element import get_se
    from processing.morphology.erosion_dilation import dilate
    binary, _ = _binarize(image)
    return _bool_to_uint8(dilate(binary, get_se(se_shape, se_size)))


def _open_fn(image, se_shape='square', se_size=3):
    from processing.morphology.structuring_element import get_se
    from processing.morphology.opening_closing import opening
    binary, _ = _binarize(image)
    return _bool_to_uint8(opening(binary, get_se(se_shape, se_size)))


def _close_fn(image, se_shape='square', se_size=3):
    from processing.morphology.structuring_element import get_se
    from processing.morphology.opening_closing import closing
    binary, _ = _binarize(image)
    return _bool_to_uint8(closing(binary, get_se(se_shape, se_size)))


def _boundary_fn(image, se_shape='square', se_size=3):
    from processing.morphology.structuring_element import get_se
    from processing.morphology.boundary_extraction import extract_boundary_vectorized
    binary, _ = _binarize(image)
    return _bool_to_uint8(extract_boundary_vectorized(binary, get_se(se_shape, se_size)))


def _gradient_fn(image, se_shape='square', se_size=3):
    from processing.morphology.structuring_element import get_se
    from processing.morphology.advanced import morphological_gradient
    binary, _ = _binarize(image)
    return _bool_to_uint8(morphological_gradient(binary, get_se(se_shape, se_size)))


def _white_tophat_fn(image, se_shape='square', se_size=3):
    from processing.morphology.structuring_element import get_se
    from processing.morphology.advanced import white_top_hat
    binary, _ = _binarize(image)
    return _bool_to_uint8(white_top_hat(binary, get_se(se_shape, se_size)))


def _black_tophat_fn(image, se_shape='square', se_size=3):
    from processing.morphology.structuring_element import get_se
    from processing.morphology.advanced import black_top_hat
    binary, _ = _binarize(image)
    return _bool_to_uint8(black_top_hat(binary, get_se(se_shape, se_size)))


class MorphologyPanel(QWidget):
    morphology_applied   = pyqtSignal(str, np.ndarray)
    segmentation_applied = pyqtSignal(str, np.ndarray)
    error_occurred       = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._worker: PipelineWorker | None = None
        self._seg_gray:      np.ndarray | None = None
        self._seg_binary:    np.ndarray | None = None
        self._seg_label_map: np.ndarray | None = None
        self._seg_showing_binary = True
        self._pending_ref: list = []
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(200)
        self._debounce_timer.timeout.connect(self._apply_manual_thresh)
        self._build_ui()

    def set_state(self, state):
        self._state = state

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_morph_tab(),   "MORPH")
        tabs.addTab(self._build_segment_tab(), "SEGMENT")
        layout.addWidget(tabs)

    def _lbl(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(FIELD_SS)
        return lbl

    def _header(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(HEADER_SS)
        return lbl

    # ---- MORPH tab ----

    def _build_morph_tab(self) -> QWidget:
        p = _get_theme()
        w = QWidget()
        lyt = QVBoxLayout(w)
        lyt.setContentsMargins(8, 8, 8, 8)
        lyt.setSpacing(6)

        radio_ss = (
            f"QRadioButton {{ color: {p['TEXT']}; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; font-size: 10px; }}"
        )

        lyt.addWidget(self._header("SE SHAPE"))
        self._shape_group = QButtonGroup(self)
        shape_row = QHBoxLayout()
        for i, name in enumerate(["Square", "Cross", "Disk"]):
            rb = QRadioButton(name)
            rb.setStyleSheet(radio_ss)
            if i == 0:
                rb.setChecked(True)
            self._shape_group.addButton(rb, i)
            shape_row.addWidget(rb)
        lyt.addLayout(shape_row)

        lyt.addWidget(self._header("SE SIZE"))
        self._size_group = QButtonGroup(self)
        size_row = QHBoxLayout()
        for i, sz in enumerate([3, 5, 7, 9]):
            rb = QRadioButton(str(sz))
            rb.setStyleSheet(radio_ss)
            if i == 0:
                rb.setChecked(True)
            self._size_group.addButton(rb, sz)
            size_row.addWidget(rb)
        lyt.addLayout(size_row)

        lyt.addWidget(self._header("OPERATIONS"))

        ops = [
            ("Erode",      _erode_fn,        "Erode"),
            ("Dilate",     _dilate_fn,       "Dilate"),
            ("Open",       _open_fn,         "Open"),
            ("Close",      _close_fn,        "Close"),
            ("Boundary",   _boundary_fn,     "Boundary"),
            ("Gradient",   _gradient_fn,     "Gradient"),
            ("Top Hat W",  _white_tophat_fn, "White Top Hat"),
            ("Top Hat B",  _black_tophat_fn, "Black Top Hat"),
        ]

        col_left  = QVBoxLayout()
        col_right = QVBoxLayout()
        col_left.setSpacing(4)
        col_right.setSpacing(4)
        for idx, (label, fn, op_name) in enumerate(ops):
            btn = QPushButton(label)
            btn.setStyleSheet(btn_style('default'))
            btn.clicked.connect(lambda _, f=fn, n=op_name: self._apply_morph(f, n))
            if idx % 2 == 0:
                col_left.addWidget(btn)
            else:
                col_right.addWidget(btn)

        btn_grid = QHBoxLayout()
        btn_grid.setSpacing(4)
        btn_grid.addLayout(col_left)
        btn_grid.addLayout(col_right)
        lyt.addLayout(btn_grid)
        lyt.addStretch()
        return w

    # ---- SEGMENT tab ----

    def _build_segment_tab(self) -> QWidget:
        p = _get_theme()
        w = QWidget()
        lyt = QVBoxLayout(w)
        lyt.setContentsMargins(8, 8, 8, 8)
        lyt.setSpacing(6)

        lyt.addWidget(self._header("AUTO OTSU"))
        otsu_btn = QPushButton("APPLY OTSU")
        otsu_btn.setStyleSheet(APPLY_BTN_SS)
        otsu_btn.clicked.connect(self._apply_otsu)
        lyt.addWidget(otsu_btn)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(self._lbl("THRESHOLD"))
        self._threshold_lbl = QLabel("---")
        self._threshold_lbl.setStyleSheet(
            f"color: {p['ACCENT']}; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; font-size: 10px;"
        )
        thresh_row.addWidget(self._threshold_lbl)
        lyt.addLayout(thresh_row)

        lyt.addWidget(self._header("MANUAL THRESHOLD"))
        manual_row = QHBoxLayout()
        self._manual_slider = QSlider(Qt.Orientation.Horizontal)
        self._manual_slider.setRange(0, 255)
        self._manual_slider.setValue(128)
        self._manual_spin = QSpinBox()
        self._manual_spin.setStyleSheet(SPINBOX_SS)
        self._manual_spin.setRange(0, 255)
        self._manual_spin.setValue(128)
        self._manual_spin.setFixedWidth(52)
        self._manual_slider.valueChanged.connect(self._manual_spin.setValue)
        self._manual_slider.valueChanged.connect(self._on_manual_thresh_changed)
        self._manual_spin.valueChanged.connect(self._manual_slider.setValue)
        manual_row.addWidget(self._manual_slider)
        manual_row.addWidget(self._manual_spin)
        lyt.addLayout(manual_row)
        manual_apply_btn = QPushButton("APPLY MANUAL")
        manual_apply_btn.setStyleSheet(APPLY_BTN_SS)
        manual_apply_btn.clicked.connect(self._apply_manual_thresh)
        lyt.addWidget(manual_apply_btn)

        lyt.addWidget(self._header("ADAPTIVE THRESHOLD"))
        blk_row = QHBoxLayout()
        blk_row.addWidget(self._lbl("BLOCK SIZE"))
        self._blk_spin = QSpinBox()
        self._blk_spin.setStyleSheet(SPINBOX_SS)
        self._blk_spin.setRange(3, 201)
        self._blk_spin.setSingleStep(2)
        self._blk_spin.setValue(51)
        blk_row.addWidget(self._blk_spin)
        lyt.addLayout(blk_row)

        c_row = QHBoxLayout()
        c_row.addWidget(self._lbl("C"))
        self._c_spin = QDoubleSpinBox()
        self._c_spin.setStyleSheet(SPINBOX_SS)
        self._c_spin.setRange(-50.0, 50.0)
        self._c_spin.setSingleStep(0.5)
        self._c_spin.setValue(5.0)
        c_row.addWidget(self._c_spin)
        lyt.addLayout(c_row)

        adapt_btn = QPushButton("APPLY ADAPTIVE")
        adapt_btn.setStyleSheet(APPLY_BTN_SS)
        adapt_btn.clicked.connect(self._apply_adaptive)
        lyt.addWidget(adapt_btn)

        lyt.addWidget(self._header("MULTI-CLASS OTSU"))
        ncls_row = QHBoxLayout()
        ncls_row.addWidget(self._lbl("N CLASSES"))
        self._ncls_spin = QSpinBox()
        self._ncls_spin.setStyleSheet(SPINBOX_SS)
        self._ncls_spin.setRange(2, 4)
        self._ncls_spin.setValue(2)
        ncls_row.addWidget(self._ncls_spin)
        lyt.addLayout(ncls_row)

        multi_btn = QPushButton("APPLY MULTI-CLASS")
        multi_btn.setStyleSheet(APPLY_BTN_SS)
        multi_btn.clicked.connect(self._apply_multi_otsu)
        lyt.addWidget(multi_btn)

        self._color_overlay_cb = QCheckBox("Color Overlay")
        self._color_overlay_cb.setStyleSheet(
            f"QCheckBox {{ color: {p['TEXT']}; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; font-size: 10px; }}"
        )
        self._color_overlay_cb.toggled.connect(self._on_overlay_toggled)
        lyt.addWidget(self._color_overlay_cb)

        self._toggle_btn = QPushButton("SHOW ORIGINAL")
        self._toggle_btn.setStyleSheet(btn_style('ghost'))
        self._toggle_btn.clicked.connect(self._toggle_binary)
        lyt.addWidget(self._toggle_btn)

        lyt.addStretch()
        return w

    # ------------------------------------------------------------------
    # SE helpers
    # ------------------------------------------------------------------

    def _se_params(self):
        shape_map = {0: 'square', 1: 'cross', 2: 'disk'}
        shape = shape_map.get(self._shape_group.checkedId(), 'square')
        size  = self._size_group.checkedId()
        if size <= 0:
            size = 3
        return shape, size

    # ------------------------------------------------------------------
    # Morphology operations
    # ------------------------------------------------------------------

    def _apply_morph(self, fn, op_name: str):
        if self._state is None:
            return
        if self._worker and self._worker.isRunning():
            return
        shape, size = self._se_params()
        self._worker = PipelineWorker(fn, op_name, self._state,
                                      se_shape=shape, se_size=size)
        self._worker.finished.connect(self.morphology_applied)
        self._worker.error.connect(self.error_occurred)
        self._worker.start()

    # ------------------------------------------------------------------
    # Segmentation operations
    # ------------------------------------------------------------------

    def _apply_otsu(self):
        if self._state is None:
            return
        if self._worker and self._worker.isRunning():
            return
        img = self._state.current()
        self._seg_gray = _to_gray(img) if img is not None else None
        ref = [None]
        self._pending_ref = ref

        def _fn(image):
            from processing.segmentation.otsu_threshold import otsu_binarize
            binary, t = otsu_binarize(image)
            ref[0] = t
            return binary

        self._worker = PipelineWorker(_fn, "Otsu Binary", self._state)
        self._worker.finished.connect(self._on_otsu_done)
        self._worker.error.connect(self.error_occurred)
        self._worker.start()

    def _on_manual_thresh_changed(self, _val: int):
        self._debounce_timer.start()

    def _apply_manual_thresh(self):
        if self._state is None:
            return
        if self._worker and self._worker.isRunning():
            return
        t = self._manual_spin.value()
        img = self._state.current()
        self._seg_gray = _to_gray(img) if img is not None else None

        def _fn(image):
            gray = _to_gray(image)
            binary = (gray > t).astype(np.uint8) * 255
            return binary

        self._worker = PipelineWorker(_fn, f"Manual Thresh {t}", self._state)
        self._worker.finished.connect(self._on_seg_done)
        self._worker.error.connect(self.error_occurred)
        self._worker.start()

    def _on_otsu_done(self, op_name: str, result: np.ndarray):
        self._seg_binary    = result
        self._seg_label_map = (result > 127).astype(np.uint8)
        self._seg_showing_binary = True
        self._toggle_btn.setText("SHOW ORIGINAL")
        t = self._pending_ref[0] if self._pending_ref else '?'
        self._threshold_lbl.setText(str(t))
        self.segmentation_applied.emit(op_name, result)

    def _apply_adaptive(self):
        if self._state is None:
            return
        if self._worker and self._worker.isRunning():
            return
        img = self._state.current()
        self._seg_gray = _to_gray(img) if img is not None else None

        def _fn(image, block_size, C):
            from processing.segmentation.otsu_threshold import adaptive_threshold
            return adaptive_threshold(image, block_size=block_size, C=C)

        self._worker = PipelineWorker(_fn, "Adaptive Thresh", self._state,
                                      block_size=self._blk_spin.value(),
                                      C=self._c_spin.value())
        self._worker.finished.connect(self._on_seg_done)
        self._worker.error.connect(self.error_occurred)
        self._worker.start()

    def _apply_multi_otsu(self):
        if self._state is None:
            return
        if self._worker and self._worker.isRunning():
            return
        img = self._state.current()
        self._seg_gray = _to_gray(img) if img is not None else None
        ref = [None]
        self._pending_ref = ref
        n = self._ncls_spin.value()
        scale = 255 // max(n - 1, 1)

        def _fn(image, n_classes):
            from processing.segmentation.otsu_threshold import multi_level_otsu
            thresholds, label_map = multi_level_otsu(image, n_classes=n_classes)
            ref[0] = thresholds
            return (label_map * scale).astype(np.uint8)

        self._worker = PipelineWorker(_fn, "Multi Otsu", self._state, n_classes=n)
        self._worker.finished.connect(self._on_multi_done)
        self._worker.error.connect(self.error_occurred)
        self._worker.start()

    def _on_multi_done(self, op_name: str, result: np.ndarray):
        self._seg_binary    = result
        n = self._ncls_spin.value()
        divisor = max(255 // max(n - 1, 1), 1)
        self._seg_label_map = (result // divisor).astype(np.uint8)
        self._seg_showing_binary = True
        self._toggle_btn.setText("SHOW ORIGINAL")
        thresholds = self._pending_ref[0] if self._pending_ref else []
        self._threshold_lbl.setText(str(thresholds))
        self.segmentation_applied.emit(op_name, result)

    def _on_seg_done(self, op_name: str, result: np.ndarray):
        self._seg_binary    = result
        self._seg_label_map = (result > 127).astype(np.uint8)
        self._seg_showing_binary = True
        self._toggle_btn.setText("SHOW ORIGINAL")
        self.segmentation_applied.emit(op_name, result)

    def _toggle_binary(self):
        if self._seg_binary is None or self._seg_gray is None:
            return
        self._seg_showing_binary = not self._seg_showing_binary
        if self._seg_showing_binary:
            self._toggle_btn.setText("SHOW ORIGINAL")
            out = self._seg_binary
        else:
            self._toggle_btn.setText("SHOW BINARY")
            out = self._seg_gray
        self.segmentation_applied.emit("Toggle Binary", out)

    def _on_overlay_toggled(self, checked: bool):
        if not checked or self._seg_gray is None or self._seg_label_map is None:
            return
        from processing.segmentation.otsu_threshold import apply_colormap_overlay
        result = apply_colormap_overlay(self._seg_gray, self._seg_label_map)
        self.segmentation_applied.emit("Color Overlay", result)
