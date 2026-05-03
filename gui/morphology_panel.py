# STATUS: IMPLEMENTED
"""
Panel for binary morphological operations on thresholded medical images.
User can binarize the image, choose structuring element shape and size, and apply erosion, dilation, opening, closing, or boundary extraction.
"""

# PyQt6.QtWidgets — QWidget, QVBoxLayout, QHBoxLayout, QSlider, QSpinBox, QComboBox, QPushButton, QLabel, QGroupBox
# PyQt6.QtCore — pyqtSignal
# processing.morphology.structuring_element — get_square_se, get_cross_se
# processing.morphology.erosion_dilation — erode, dilate
# processing.morphology.opening_closing — opening, closing
# processing.morphology.boundary_extraction — extract_boundary
# utils.image_utils — binarize: threshold numpy array to 0/1

# FUNCTIONS / CLASSES
# class MorphologyPanel(QWidget):
#   def __init__: build UI — threshold slider, SE shape selector, SE size spinner, op buttons
#   def on_threshold_changed: binarize image at current threshold → display preview
#   def on_operation_clicked: read SE settings → apply selected op → emit result
# signal: morphology_applied(np.ndarray)

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import validate_grayscale, binarize, normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# validate_grayscale(image)           # call before binarize — morphology only works on grayscale
# binarize(image, threshold)          # call in on_threshold_changed using slider value
# normalize_to_uint8(result)          # call on morphology output before emitting signal
# @wrap_errors                        # decorate on_threshold_changed and on_operation_clicked
# Note: pass the binarized array (not the original) to all erosion/dilation/opening/closing calls

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QSlider, QLabel, QGridLayout, QButtonGroup,
                              QRadioButton, QFrame, QSizePolicy)
from PyQt6.QtCore import pyqtSignal, Qt

from gui.styles import (BG, PANEL, PANEL2, INPUT, BORDER, BORDER2,
                        ACCENT, TEXT, MUTED, MUTED2, btn_style)
from utils import (validate_grayscale, binarize, normalize_to_uint8, wrap_errors, show_error_dialog,)


class MorphologyPanel(QWidget):
    morphology_applied = pyqtSignal(str, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_image: np.ndarray | None = None
        self._binary_image: np.ndarray | None = None

        self.setStyleSheet(f"background:{PANEL};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # ---- header ----
        hdr = QLabel("MORPHOLOGY")
        hdr.setStyleSheet(f"color:{ACCENT};font-size:9px;font-weight:bold;letter-spacing:.15em;")
        layout.addWidget(hdr)

        # ---- binarization section ----
        bin_lbl = QLabel("BINARIZATION")
        bin_lbl.setStyleSheet(f"color:{MUTED};font-size:8px;font-weight:bold;letter-spacing:.12em;")
        layout.addWidget(bin_lbl)

        slider_row = QWidget()
        srl = QHBoxLayout(slider_row)
        srl.setContentsMargins(0, 0, 0, 0)
        srl.setSpacing(8)

        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(0, 255)
        self._threshold_slider.setValue(128)
        self._threshold_slider.setStyleSheet(
            "QSlider::groove:horizontal{height:3px;background:#353730;border-radius:2px;}"
            f"QSlider::handle:horizontal{{width:12px;height:12px;margin:-5px 0;"
            f"background:{ACCENT};border-radius:6px;border:2px solid #0d1002;}}"
            f"QSlider::sub-page:horizontal{{background:{ACCENT};border-radius:2px;}}"
        )
        srl.addWidget(self._threshold_slider)

        self._thresh_val_lbl = QLabel("128")
        self._thresh_val_lbl.setFixedWidth(30)
        self._thresh_val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._thresh_val_lbl.setStyleSheet(f"color:{ACCENT};font-size:10px;font-weight:bold;")
        srl.addWidget(self._thresh_val_lbl)
        layout.addWidget(slider_row)

        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)

        # divider
        layout.addWidget(_hdiv())

        # ---- structuring element ----
        se_lbl = QLabel("STRUCTURING ELEMENT")
        se_lbl.setStyleSheet(f"color:{MUTED};font-size:8px;font-weight:bold;letter-spacing:.12em;")
        layout.addWidget(se_lbl)

        shape_row = QWidget()
        shrl = QHBoxLayout(shape_row)
        shrl.setContentsMargins(0, 0, 0, 0)
        shrl.setSpacing(4)
        shrl.addWidget(QLabel("Shape", styleSheet=f"color:{MUTED};font-size:9px;"))
        self._shape_group = QButtonGroup(self)
        for label in ("Square", "Cross"):
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
        szrl.addWidget(QLabel("Size", styleSheet=f"color:{MUTED};font-size:9px;"))
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

        # divider
        layout.addWidget(_hdiv())

        # ---- operations ----
        ops_lbl = QLabel("OPERATIONS")
        ops_lbl.setStyleSheet(f"color:{MUTED};font-size:8px;font-weight:bold;letter-spacing:.12em;")
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
        boundary_btn.setStyleSheet(btn_style('primary'))
        boundary_btn.clicked.connect(lambda: self._apply_op("Boundary"))
        layout.addWidget(boundary_btn)

        hint = QLabel("Boundary = A − (A ⊖ B)")
        hint.setStyleSheet(f"color:{MUTED2};font-size:9px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        layout.addStretch()

    # ------------------------------------------------------------------ public

    def set_image(self, image: np.ndarray):
        try:
            validate_grayscale(image)
        except ValueError:
            return
        self._source_image = image
        self._on_threshold_changed(self._threshold_slider.value())

    # ------------------------------------------------------------------ private

    def _on_threshold_changed(self, value: int):
        self._thresh_val_lbl.setText(str(value))
        if self._source_image is not None:
            self._binary_image = binarize(self._source_image, value)

    def _get_se(self) -> np.ndarray:
        size = 3
        for btn in self._size_group.buttons():
            if btn.isChecked():
                size = btn.property("sz")
                break
        shape_btn = self._shape_group.checkedButton()
        shape = shape_btn.text() if shape_btn else "Square"
        if shape == "Cross":
            from processing.morphology.structuring_element import get_cross_se
            return get_cross_se(size)
        from processing.morphology.structuring_element import get_square_se
        return get_square_se(size)

    @wrap_errors
    def _apply_op(self, op_name: str):
        if self._binary_image is None:
            show_error_dialog("No image", "Load and threshold an image first.")
            return
        se = self._get_se()

        if op_name == "Erode":
            from processing.morphology.erosion_dilation import erode
            result = erode(self._binary_image, se)
        elif op_name == "Dilate":
            from processing.morphology.erosion_dilation import dilate
            result = dilate(self._binary_image, se)
        elif op_name == "Open":
            from processing.morphology.opening_closing import opening
            result = opening(self._binary_image, se)
        elif op_name == "Close":
            from processing.morphology.opening_closing import closing
            result = closing(self._binary_image, se)
        elif op_name == "Boundary":
            from processing.morphology.boundary_extraction import extract_boundary
            result = extract_boundary(self._binary_image, se)
        else:
            return

        result = normalize_to_uint8(result * 255)
        self.morphology_applied.emit(op_name, result)


def _hdiv() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"background:#2c2e2a;max-height:1px;")
    return f
