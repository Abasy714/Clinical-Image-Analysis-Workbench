import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QTableWidget, QTableWidgetItem, QSpinBox, QHeaderView,
)
from PyQt6.QtCore import Qt, pyqtSignal

from gui.theme import get as _get_theme
from gui.styles import btn_style, SPINBOX_SS, APPLY_BTN_SS, HEADER_SS, FIELD_SS
from gui.workers import PipelineWorker


# ------------------------------------------------------------------
# Module-level worker functions — all processing imports are lazy
# ------------------------------------------------------------------

def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1]
                + 0.114 * image[:, :, 2]).astype(np.uint8)
    return image


def _blob_fn(image, _ref):
    from processing.computer_vision.blob_analysis import analyze_blobs
    result = analyze_blobs(image)
    if isinstance(result, tuple):
        overlay, stats = result
        _ref[0] = stats
        return overlay
    _ref[0] = []
    return result


def _contour_fn(image):
    from processing.computer_vision.contour import trace_contour
    return trace_contour(image)


def _harris_fn(image):
    from processing.computer_vision.feature_detection import detect_harris_corners
    return detect_harris_corners(image)


def _glcm_fn(image, roi_x, roi_y, roi_w, roi_h, _ref):
    from processing.computer_vision.texture_features import compute_glcm_features
    gray = _to_gray(image)
    roi = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    features = compute_glcm_features(roi)
    _ref[0] = features
    return roi


def _sliding_fn(image, window_size, stride):
    from processing.computer_vision.sliding_window import sliding_window_classify
    return sliding_window_classify(image, window_size=window_size, stride=stride)


# ------------------------------------------------------------------
# CVPanel
# ------------------------------------------------------------------

class CVPanel(QWidget):
    cv_applied       = pyqtSignal(str, np.ndarray)
    blob_stats_ready = pyqtSignal(list)
    error_occurred   = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._worker: PipelineWorker | None = None
        self._roi: tuple | None = None
        self._blob_stats_ref: list = []
        self._glcm_ref: list = []
        self._build_ui()

    def set_state(self, state):
        self._state = state
        self._refresh_buttons()

    def set_roi(self, x: int, y: int, w: int, h: int):
        self._roi = (x, y, w, h)
        self._roi_warning.setVisible(False)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_analysis_tab(), "ANALYSIS")
        tabs.addTab(self._build_texture_tab(),  "TEXTURE")
        tabs.addTab(self._build_sliding_tab(),  "SLIDING WINDOW")
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

    # ---- ANALYSIS tab ----

    def _build_analysis_tab(self) -> QWidget:
        p = _get_theme()
        w = QWidget()
        lyt = QVBoxLayout(w)
        lyt.setContentsMargins(8, 8, 8, 8)
        lyt.setSpacing(6)

        lyt.addWidget(self._header("BLOB ANALYSIS"))
        self._blob_btn = QPushButton("ANALYZE BLOBS")
        self._blob_btn.setStyleSheet(APPLY_BTN_SS)
        self._blob_btn.setEnabled(False)
        self._blob_btn.clicked.connect(self._apply_blobs)
        lyt.addWidget(self._blob_btn)

        self._blob_table = QTableWidget(0, 5)
        self._blob_table.setHorizontalHeaderLabels(
            ["ID", "Area", "Centroid", "Circularity", "BBox"]
        )
        self._blob_table.setFixedHeight(110)
        self._blob_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._blob_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._blob_table.setStyleSheet(
            f"QTableWidget {{ background: {p['BG']}; border: 1px solid {p['BORDER']}; "
            f"color: {p['TEXT']}; font-family: 'JetBrains Mono', Consolas, monospace; "
            f"font-size: 9px; gridline-color: {p['BORDER']}; }}"
            f"QHeaderView::section {{ background: {p['PANEL']}; color: {p['MUTED']}; "
            f"border: none; padding: 2px; font-size: 8px; }}"
        )
        lyt.addWidget(self._blob_table)

        lyt.addWidget(self._header("CONTOUR"))
        self._contour_btn = QPushButton("TRACE CONTOUR")
        self._contour_btn.setStyleSheet(btn_style('default'))
        self._contour_btn.setEnabled(False)
        self._contour_btn.clicked.connect(self._apply_contour)
        lyt.addWidget(self._contour_btn)

        lyt.addWidget(self._header("FEATURE DETECTION"))
        self._harris_btn = QPushButton("HARRIS CORNERS")
        self._harris_btn.setStyleSheet(btn_style('default'))
        self._harris_btn.setEnabled(False)
        self._harris_btn.clicked.connect(self._apply_harris)
        lyt.addWidget(self._harris_btn)

        lyt.addStretch()
        return w

    # ---- TEXTURE tab ----

    def _build_texture_tab(self) -> QWidget:
        p = _get_theme()
        w = QWidget()
        lyt = QVBoxLayout(w)
        lyt.setContentsMargins(8, 8, 8, 8)
        lyt.setSpacing(6)

        self._roi_warning = QLabel("Draw an ROI on the image first")
        self._roi_warning.setStyleSheet(
            f"color: {p['AMBER']}; font-family: 'JetBrains Mono', Consolas, monospace; "
            f"font-size: 9px; padding: 4px;"
        )
        self._roi_warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lyt.addWidget(self._roi_warning)

        lyt.addWidget(self._header("GLCM FEATURES"))
        glcm_btn = QPushButton("COMPUTE GLCM")
        glcm_btn.setStyleSheet(APPLY_BTN_SS)
        glcm_btn.clicked.connect(self._apply_glcm)
        lyt.addWidget(glcm_btn)

        self._glcm_labels: dict[str, QLabel] = {}
        for feat in ["Contrast", "Homogeneity", "Energy",
                     "Correlation", "ASM", "Entropy"]:
            row, val_lbl = self._stat_row(feat)
            lyt.addWidget(row)
            self._glcm_labels[feat.lower()] = val_lbl

        lyt.addStretch()
        return w

    # ---- SLIDING WINDOW tab ----

    def _build_sliding_tab(self) -> QWidget:
        import os
        p = _get_theme()
        w = QWidget()
        lyt = QVBoxLayout(w)
        lyt.setContentsMargins(8, 8, 8, 8)
        lyt.setSpacing(6)

        weights = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'processing', 'deep_learning', 'weights',
            'brain_tumor_classifier.h5'
        )
        model_found = os.path.exists(weights)
        self._sw_status = QLabel(
            "MODEL: LOADED" if model_found else "MODEL: NOT FOUND"
        )
        self._sw_status.setStyleSheet(
            f"color: {p['ACCENT'] if model_found else p['RED']}; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; font-size: 9px;"
        )
        lyt.addWidget(self._sw_status)

        lyt.addWidget(self._header("WINDOW SETTINGS"))

        win_row = QHBoxLayout()
        win_row.addWidget(self._lbl("WINDOW SIZE"))
        self._win_spin = QSpinBox()
        self._win_spin.setStyleSheet(SPINBOX_SS)
        self._win_spin.setRange(32, 256)
        self._win_spin.setSingleStep(32)
        self._win_spin.setValue(64)
        win_row.addWidget(self._win_spin)
        lyt.addLayout(win_row)

        stride_row = QHBoxLayout()
        stride_row.addWidget(self._lbl("STRIDE"))
        self._stride_spin = QSpinBox()
        self._stride_spin.setStyleSheet(SPINBOX_SS)
        self._stride_spin.setRange(8, 128)
        self._stride_spin.setSingleStep(8)
        self._stride_spin.setValue(32)
        stride_row.addWidget(self._stride_spin)
        lyt.addLayout(stride_row)

        run_btn = QPushButton("RUN SLIDING WINDOW")
        run_btn.setStyleSheet(APPLY_BTN_SS)
        run_btn.clicked.connect(self._apply_sliding)
        lyt.addWidget(run_btn)
        lyt.addStretch()
        return w

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_buttons(self):
        has_image = (self._state is not None
                     and self._state.current() is not None)
        for btn in (self._blob_btn, self._contour_btn, self._harris_btn):
            btn.setEnabled(has_image)

    # ------------------------------------------------------------------
    # Apply operations
    # ------------------------------------------------------------------

    def _apply_blobs(self):
        if self._state is None or (self._worker and self._worker.isRunning()):
            return
        self._blob_stats_ref = [None]
        ref = self._blob_stats_ref

        def _fn(image):
            return _blob_fn(image, ref)

        self._worker = PipelineWorker(_fn, "Blob Analysis", self._state)
        self._worker.finished.connect(self._on_blobs_done)
        self._worker.error.connect(self.error_occurred)
        self._worker.start()

    def _on_blobs_done(self, op_name: str, overlay: np.ndarray):
        self.cv_applied.emit(op_name, overlay)
        stats = self._blob_stats_ref[0] or []
        if stats:
            self.blob_stats_ready.emit(stats)
            self._populate_blob_table(stats)

    def _populate_blob_table(self, stats: list):
        self._blob_table.setRowCount(len(stats))
        for row, blob in enumerate(stats):
            self._blob_table.setItem(row, 0, QTableWidgetItem(str(blob.get('id', row))))
            self._blob_table.setItem(row, 1, QTableWidgetItem(str(blob.get('area', ''))))
            cy, cx = blob.get('centroid', (0, 0))
            self._blob_table.setItem(row, 2, QTableWidgetItem(f"({cx:.0f},{cy:.0f})"))
            self._blob_table.setItem(row, 3, QTableWidgetItem(
                f"{blob.get('circularity', 0):.3f}"
            ))
            self._blob_table.setItem(row, 4, QTableWidgetItem(
                str(blob.get('bbox', ''))
            ))

    def _apply_contour(self):
        if self._state is None or (self._worker and self._worker.isRunning()):
            return
        self._worker = PipelineWorker(_contour_fn, "Contour Trace", self._state)
        self._worker.finished.connect(lambda n, r: self.cv_applied.emit(n, r))
        self._worker.error.connect(self.error_occurred)
        self._worker.start()

    def _apply_harris(self):
        if self._state is None or (self._worker and self._worker.isRunning()):
            return
        self._worker = PipelineWorker(_harris_fn, "Harris Corners", self._state)
        self._worker.finished.connect(lambda n, r: self.cv_applied.emit(n, r))
        self._worker.error.connect(self.error_occurred)
        self._worker.start()

    def _apply_glcm(self):
        if self._state is None:
            return
        if self._roi is None:
            self._roi_warning.setVisible(True)
            return
        if self._worker and self._worker.isRunning():
            return
        self._glcm_ref = [None]
        ref = self._glcm_ref
        x, y, w, h = self._roi

        def _fn(image):
            return _glcm_fn(image, x, y, w, h, ref)

        self._worker = PipelineWorker(_fn, "GLCM Features", self._state)
        self._worker.finished.connect(self._on_glcm_done)
        self._worker.error.connect(self.error_occurred)
        self._worker.start()

    def _on_glcm_done(self, _op_name: str, _roi: np.ndarray):
        features = self._glcm_ref[0] or {}
        for key in ("contrast", "homogeneity", "energy",
                    "correlation", "asm", "entropy"):
            lbl = self._glcm_labels.get(key)
            if lbl is None:
                continue
            val = features.get(key)
            lbl.setText(f"{val:.4f}" if isinstance(val, (int, float)) else "---")

    def _apply_sliding(self):
        if self._state is None or (self._worker and self._worker.isRunning()):
            return
        self._worker = PipelineWorker(
            _sliding_fn, "Sliding Window", self._state,
            window_size=self._win_spin.value(),
            stride=self._stride_spin.value(),
        )
        self._worker.finished.connect(lambda n, r: self.cv_applied.emit(n, r))
        self._worker.error.connect(self.error_occurred)
        self._worker.start()
