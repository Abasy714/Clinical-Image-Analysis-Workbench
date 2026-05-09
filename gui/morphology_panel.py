# STATUS: IMPLEMENTED
"""
Panel for binary morphological operations on thresholded medical images.
User can binarize the image, choose structuring element shape and size, and apply erosion, dilation, opening, closing, or boundary extraction.
"""

import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QSlider, QLabel, QGridLayout, QButtonGroup,
                              QRadioButton, QFrame, QSizePolicy)
from PyQt6.QtCore import pyqtSignal, Qt

from gui.styles import (BG, PANEL, PANEL2, INPUT, BORDER, BORDER2,
                        ACCENT, TEXT, MUTED, MUTED2, btn_style,
                        HEADER_SS, FIELD_SS, APPLY_BTN_SS)
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
        hdr.setStyleSheet(HEADER_SS)
        layout.addWidget(hdr)

        # ---- binarization section ----
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
                height: 3px;
                background: #353730;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 13px;
                height: 13px;
                margin: -5px 0;
                background: #c8f135;
                border-radius: 7px;
                border: 2px solid #0d1002;
            }
            QSlider::sub-page:horizontal {
                background: #c8f135;
                border-radius: 2px;
            }
            QSlider::add-page:horizontal {
                background: #353730;
                border-radius: 2px;
            }
        """)
        srl.addWidget(self._threshold_slider)

        self._thresh_val_lbl = QLabel("128")
        self._thresh_val_lbl.setFixedWidth(30)
        self._thresh_val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._thresh_val_lbl.setStyleSheet("""
            QLabel {
                color: #c8f135;
                font-family: 'JetBrains Mono', Consolas, monospace;
                font-size: 13px;
                font-weight: bold;
            }
        """)
        srl.addWidget(self._thresh_val_lbl)
        layout.addWidget(slider_row)

        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)

        # divider
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

        # divider
        layout.addWidget(_hdiv())

        # ---- operations ----
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

    def _apply_op(self, op_name: str):
        if self._binary_image is None:
            show_error_dialog("No image", "Load and threshold an image first.")
            return
        try: 
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
 
            result = normalize_to_uint8(result.astype(np.uint8) * 255) #changed by sohaila
        
            self.morphology_applied.emit(op_name, result)
        except (ImportError, NotImplementedError, Exception) as e:
            show_error_dialog("Morphology Error", f"Operation '{op_name}' failed.\n{e}")


def _hdiv() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"background:#2c2e2a;max-height:1px;")
    return f
