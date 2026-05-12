import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QButtonGroup, QRadioButton, QSlider,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from gui.theme import get as _get_theme
from gui.styles import btn_style, operation_btn_style, SPINBOX_SS, HEADER_SS, FIELD_SS
from gui.workers import PipelineWorker


class MorphologyPanel(QWidget):
    morphology_applied = pyqtSignal(str, np.ndarray)
    error_occurred     = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._worker: PipelineWorker | None = None
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
        layout.addWidget(self._build_morph_content())

    def _lbl(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(FIELD_SS)
        return lbl

    def _header(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(HEADER_SS)
        return lbl

    def _build_morph_content(self) -> QWidget:
        p = _get_theme()
        w = QWidget()
        lyt = QVBoxLayout(w)
        lyt.setContentsMargins(8, 8, 8, 8)
        lyt.setSpacing(6)

        # ---- Binarization threshold ----
        lyt.addWidget(self._header("BINARIZATION"))
        thresh_row = QHBoxLayout()
        self._morph_thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._morph_thresh_slider.setRange(0, 255)
        self._morph_thresh_slider.setValue(128)
        self._morph_thresh_spin = QSpinBox()
        self._morph_thresh_spin.setStyleSheet(SPINBOX_SS)
        self._morph_thresh_spin.setRange(0, 255)
        self._morph_thresh_spin.setValue(128)
        self._morph_thresh_spin.setFixedWidth(52)
        self._morph_thresh_val_lbl = QLabel("[128]")
        self._morph_thresh_val_lbl.setStyleSheet(
            f"color: {p['ACCENT']}; font-family: 'JetBrains Mono', Consolas, monospace;"
            f" font-size: 10px;"
        )
        self._morph_thresh_val_lbl.setFixedWidth(38)
        thresh_row.addWidget(self._morph_thresh_slider)
        thresh_row.addWidget(self._morph_thresh_spin)
        thresh_row.addWidget(self._morph_thresh_val_lbl)
        lyt.addLayout(thresh_row)

        pre_op_lbl = QLabel("Pre-op threshold")
        pre_op_lbl.setStyleSheet(
            f"color: {p['MUTED']}; font-size: 8px;"
            f" font-family: 'JetBrains Mono', Consolas, monospace;"
        )
        lyt.addWidget(pre_op_lbl)

        self._morph_thresh_preview = QLabel()
        self._morph_thresh_preview.setFixedHeight(60)
        self._morph_thresh_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._morph_thresh_preview.setStyleSheet(
            f"background: {p['BG']}; border: 1px solid {p['BORDER']};"
        )
        lyt.addWidget(self._morph_thresh_preview)

        self._morph_thresh_timer = QTimer(self)
        self._morph_thresh_timer.setSingleShot(True)
        self._morph_thresh_timer.setInterval(400)
        self._morph_thresh_timer.timeout.connect(self._preview_morph_thresh)

        def _on_thresh_changed(v: int):
            self._morph_thresh_spin.setValue(v)
            self._morph_thresh_val_lbl.setText(f"[{v}]")
            self._morph_thresh_timer.start()

        self._morph_thresh_slider.valueChanged.connect(_on_thresh_changed)
        self._morph_thresh_spin.valueChanged.connect(self._morph_thresh_slider.setValue)

        # ---- SE SHAPE ----
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
            ("Erode",      "erode",          "Erode"),
            ("Dilate",     "dilate",         "Dilate"),
            ("Open",       "open",           "Open"),
            ("Close",      "close",          "Close"),
            ("Boundary",   "boundary",       "Boundary"),
            ("Gradient",   "gradient",       "Gradient"),
            ("Top Hat W",  "white_top_hat",  "White Top Hat"),
            ("Top Hat B",  "black_top_hat",  "Black Top Hat"),
        ]

        col_left  = QVBoxLayout()
        col_right = QVBoxLayout()
        col_left.setSpacing(4)
        col_right.setSpacing(4)
        for idx, (label, operation, op_name) in enumerate(ops):
            btn = QPushButton(label)
            btn.setStyleSheet(operation_btn_style())
            btn.clicked.connect(
                lambda _, op=operation, n=op_name: self._apply_morph(op, n)
            )
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

    def _preview_morph_thresh(self):
        if self._state is None:
            return
        image = self._state.get_base_image()
        if image is None or not isinstance(image, np.ndarray):
            return
        from utils.image_utils import to_grayscale, normalize_to_uint8, to_qpixmap
        t = self._morph_thresh_slider.value()
        gray = normalize_to_uint8(to_grayscale(image))
        binary = (gray > t).astype(np.uint8) * 255
        w = self._morph_thresh_preview.width() or 200
        h = self._morph_thresh_preview.height()
        pix = to_qpixmap(binary).scaled(
            w, h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self._morph_thresh_preview.setPixmap(pix)

    def _apply_morph(self, operation: str, op_name: str):
        if self._state is None:
            return
        if self._worker and self._worker.isRunning():
            return
        shape, size = self._se_params()
        threshold = self._morph_thresh_slider.value()
        full_name = f"{op_name} (t={threshold})"

        def _fn(image, operation, se_shape, se_size, threshold):
            from processing.morphology.pipeline_ops import apply_morphology
            return apply_morphology(
                image,
                operation=operation,
                se_shape=se_shape,
                se_size=se_size,
                threshold=threshold,
            )

        self._worker = PipelineWorker(
            _fn, full_name, self._state,
            operation=operation, se_shape=shape, se_size=size, threshold=threshold,
        )
        self._worker.finished.connect(self._on_morph_done)
        self._worker.error.connect(self.error_occurred)
        self._worker.start()

    def _on_morph_done(self, op_name: str, result: np.ndarray):
        self.morphology_applied.emit(op_name, result)
