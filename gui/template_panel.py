import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from gui.theme import get as _get_theme
from gui.styles import btn_style, operation_btn_style, COMBO_SS, APPLY_BTN_SS, HEADER_SS, FIELD_SS
from gui.workers import PipelineWorker


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _qimage_to_gray(path: str) -> np.ndarray | None:
    img = QImage(path)
    if img.isNull():
        return None
    gray = img.convertToFormat(QImage.Format.Format_Grayscale8)
    w, h = gray.width(), gray.height()
    ptr = gray.bits()
    ptr.setsize(h * w)
    return np.frombuffer(ptr, dtype=np.uint8).reshape(h, w).copy()


def _gray_to_qpixmap(arr: np.ndarray, max_size: int = 120) -> QPixmap:
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
    pix = QPixmap.fromImage(img)
    if w > max_size or h > max_size:
        pix = pix.scaled(max_size, max_size,
                         Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.FastTransformation)
    return pix


def _fft_match_fn(image, template):
    from processing.frequency.template_matching import fourier_cross_correlate
    return fourier_cross_correlate(image, template)


def _ncc_match_fn(image, template):
    from processing.frequency.template_matching import normalized_cross_correlation
    return normalized_cross_correlation(image, template)


# ------------------------------------------------------------------
# TemplatePanel
# ------------------------------------------------------------------

class TemplatePanel(QWidget):
    template_applied = pyqtSignal(str, np.ndarray)
    error_occurred   = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._worker: PipelineWorker | None = None
        self._template: np.ndarray | None = None
        self._roi: tuple | None = None
        self._build_ui()

    def set_state(self, state):
        self._state = state

    def set_roi(self, x: int, y: int, w: int, h: int):
        self._roi = (x, y, w, h)
        self._roi_lbl.setText(f"ROI: ({x},{y}) {w}×{h}")
        self._roi_btn.setEnabled(self._state is not None)
        self._load_template_from_roi()

    def _load_template_from_roi(self):
        if self._state is None or self._roi is None:
            return
        image = self._state.current()
        if image is None:
            return
        x, y, w, h = self._roi
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        w = max(1, min(w, image.shape[1] - x))
        h = max(1, min(h, image.shape[0] - y))
        roi = image[y:y+h, x:x+w]
        if roi.ndim == 3:
            roi_gray = (0.299 * roi[:, :, 0] + 0.587 * roi[:, :, 1]
                        + 0.114 * roi[:, :, 2]).astype(np.uint8)
        else:
            roi_gray = roi.copy()
        self._template = roi_gray
        self._show_template_thumbnail(roi_gray)
        self._template_info.setText(f"ROI template: {w}×{h}")
        self._find_btn.setEnabled(True)

    def _show_template_thumbnail(self, arr: np.ndarray):
        pix = _gray_to_qpixmap(arr)
        self._thumb_lbl.setPixmap(pix)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        p = _get_theme()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(self._header("TEMPLATE"))

        load_btn = QPushButton("LOAD TEMPLATE")
        load_btn.setStyleSheet(btn_style('default'))
        load_btn.clicked.connect(self._load_template)
        layout.addWidget(load_btn)

        self._roi_lbl = QLabel("No ROI selected")
        self._roi_lbl.setStyleSheet(
            f"color: {p['MUTED']}; font-family: 'JetBrains Mono', Consolas, monospace; "
            f"font-size: 9px;"
        )
        layout.addWidget(self._roi_lbl)

        self._roi_btn = QPushButton("USE ROI AS TEMPLATE")
        self._roi_btn.setStyleSheet(btn_style('default'))
        self._roi_btn.setEnabled(False)
        self._roi_btn.clicked.connect(self._load_template_from_roi)
        layout.addWidget(self._roi_btn)

        self._thumb_lbl = QLabel()
        self._thumb_lbl.setFixedSize(120, 120)
        self._thumb_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb_lbl.setStyleSheet(
            f"QLabel {{ background: {p['PANEL2']}; border: 1px solid {p['BORDER']}; }}"
        )
        layout.addWidget(self._thumb_lbl)

        self._template_info = QLabel("No template loaded")
        self._template_info.setStyleSheet(FIELD_SS)
        self._template_info.setWordWrap(True)
        layout.addWidget(self._template_info)

        layout.addWidget(self._header("METHOD"))
        self._method_combo = QComboBox()
        self._method_combo.setStyleSheet(COMBO_SS)
        self._method_combo.addItem("FFT Cross-Correlation")
        self._method_combo.addItem("Normalized CC")
        layout.addWidget(self._method_combo)

        self._find_btn = QPushButton("FIND MATCH")
        self._find_btn.setStyleSheet(operation_btn_style())
        self._find_btn.setEnabled(False)
        self._find_btn.clicked.connect(self._find_match)
        layout.addWidget(self._find_btn)

        layout.addWidget(self._header("RESULT"))

        row_w, self._row_lbl   = self._stat_row("Row")
        col_w, self._col_lbl   = self._stat_row("Col")
        conf_w, self._conf_lbl = self._stat_row("Confidence")
        layout.addWidget(row_w)
        layout.addWidget(col_w)
        layout.addWidget(conf_w)

        layout.addStretch()

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

    # ------------------------------------------------------------------
    # Load template
    # ------------------------------------------------------------------

    def _load_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Template", "",
            "Images (*.jpg *.jpeg *.png *.bmp);;All Files (*)",
        )
        if not path:
            return
        arr = _qimage_to_gray(path)
        if arr is None:
            self._template_info.setText("Failed to load image.")
            return
        self._template = arr
        pix = _gray_to_qpixmap(arr)
        self._thumb_lbl.setPixmap(pix)
        import os
        fname = os.path.basename(path)
        self._template_info.setText(f"{fname}\n{arr.shape[1]}×{arr.shape[0]} px")
        self._find_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Find match
    # ------------------------------------------------------------------

    def _find_match(self):
        if self._state is None:
            return
        if self._template is None:
            self._template_info.setText("Load a template first.")
            return
        if self._worker and self._worker.isRunning():
            return

        method = self._method_combo.currentText()
        fn = _fft_match_fn if method == "FFT Cross-Correlation" else _ncc_match_fn

        self._find_btn.setEnabled(False)
        self._find_btn.setText("⟳ Processing...")
        self._worker = PipelineWorker(fn, "Template Match", self._state,
                                      template=self._template.copy())
        self._worker.finished.connect(self._on_match_done)
        self._worker.error.connect(self.error_occurred)
        self._worker.finished.connect(lambda *_: (self._find_btn.setEnabled(True), self._find_btn.setText("FIND MATCH")))
        self._worker.error.connect(lambda *_: (self._find_btn.setEnabled(True), self._find_btn.setText("FIND MATCH")))
        self._worker.start()

    def _on_match_done(self, _op_name: str, corr_map: np.ndarray):
        from processing.frequency.template_matching import find_best_match, draw_matches

        row, col = find_best_match(corr_map)
        row, col = int(row), int(col)

        max_val = float(corr_map.max())
        score   = float(corr_map[row, col])
        confidence = score / max_val if max_val > 1e-10 else 0.0

        t_h, t_w = self._template.shape[:2]
        match = {
            "row": row, "col": col,
            "height": t_h, "width": t_w,
            "score": confidence,
        }

        img = self._state.current()
        if img is None:
            return
        annotated = draw_matches(img, [match])

        self._row_lbl.setText(str(row))
        self._col_lbl.setText(str(col))
        self._conf_lbl.setText(f"{confidence:.3f}")

        self.template_applied.emit("Template Match", annotated)
