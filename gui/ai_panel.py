import os
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QDoubleSpinBox, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal

from gui.theme import get as _get_theme
from gui.styles import btn_style, operation_btn_style, SPINBOX_SS, APPLY_BTN_SS, HEADER_SS, FIELD_SS
from gui.workers import PipelineWorker

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

_WEIGHTS_DIR    = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'processing', 'deep_learning', 'weights',
)
_CLASSIFIER_H5  = os.path.join(_WEIGHTS_DIR, 'brain_tumor_classifier.h5')
_GRADCAM_H5     = os.path.join(_WEIGHTS_DIR, 'brain_tumor_gradcam.h5')
_METADATA_JSON  = os.path.join(_WEIGHTS_DIR, 'metadata.json')

_CLASS_LABELS   = ["No Tumor", "Meningioma", "Glioma", "Pituitary"]

_SUGGESTIONS = [
    "Denoise (Gaussian)",
    "Sharpen (Laplacian)",
    "Enhance Contrast (CLAHE)",
    "Histogram Equalization",
    "Median Filter",
]

# ------------------------------------------------------------------
# Module-level worker functions (lazy imports)
# ------------------------------------------------------------------

def _predict_fn(image, roi, input_size, _ref):
    from processing.deep_learning.classifier import predict_roi
    result = predict_roi(image, roi, input_size=input_size)
    _ref[0] = result
    if isinstance(result, tuple):
        return result[0]
    if isinstance(result, np.ndarray):
        return result
    return image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]


def _gradcam_fn(image, roi, alpha, _ref):
    from processing.deep_learning.grad_cam import compute_gradcam
    result = compute_gradcam(image, roi, alpha=alpha)
    _ref[0] = result
    return result


def _scorecam_fn(image, roi, alpha, _ref):
    from processing.deep_learning.score_cam import compute_scorecam
    result = compute_scorecam(image, roi, alpha=alpha)
    _ref[0] = result
    return result


# ------------------------------------------------------------------
# Matplotlib canvas (optional)
# ------------------------------------------------------------------

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    _MPL = True
except ImportError:
    _MPL = False


class _BarCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._fig = None
        self._ax  = None
        self._canvas = None
        if _MPL:
            self._fig    = Figure(figsize=(2.6, 1.4), tight_layout=True)
            self._ax     = self._fig.add_subplot(111)
            self._canvas = FigureCanvasQTAgg(self._fig)
            lyt = QVBoxLayout(self)
            lyt.setContentsMargins(0, 0, 0, 0)
            lyt.addWidget(self._canvas)
            self.setFixedHeight(120)
        else:
            self._placeholder = QLabel("matplotlib not available")
            self._placeholder.setStyleSheet(FIELD_SS)
            lyt = QVBoxLayout(self)
            lyt.addWidget(self._placeholder)

    def plot(self, probs: list, labels: list):
        if not _MPL or self._ax is None:
            return
        p = _get_theme()
        self._ax.clear()
        self._fig.patch.set_facecolor(p['BG'])
        self._ax.set_facecolor(p['PANEL'])
        colors = [p['ACCENT'] if v == max(probs) else p['MUTED'] for v in probs]
        self._ax.barh(labels, probs, color=colors)
        self._ax.set_xlim(0, 1)
        self._ax.tick_params(colors=p['TEXT'], labelsize=7)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(p['BORDER'])
        self._canvas.draw()

    def theme_changed(self, _palette):
        pass


# ------------------------------------------------------------------
# AIPanel
# ------------------------------------------------------------------

class AIPanel(QWidget):
    ai_applied          = pyqtSignal(str, np.ndarray)
    classification_done = pyqtSignal(list)
    error_occurred      = pyqtSignal(str)
    suggestion_clicked  = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._worker: PipelineWorker | None = None
        self._preload_worker: PipelineWorker | None = None
        self._roi: tuple | None = None
        self._pending_roi: np.ndarray | None = None
        self._model_loaded = False
        self._input_size   = [224, 224]
        self._predict_ref:  list = [None]
        self._gradcam_ref:  list = [None]
        self._scorecam_ref: list = [None]
        self._build_ui()
        self._load_metadata()
        if os.path.exists(_CLASSIFIER_H5):
            self._start_preload()

    def set_state(self, state):
        self._state = state
        self._refresh()

    def set_roi(self, x: int, y: int, w: int, h: int):
        self._roi = (x, y, w, h)
        self._refresh()

    def clear_roi(self):
        self._roi = None
        self._refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        content = QWidget()
        lyt = QVBoxLayout(content)
        lyt.setContentsMargins(8, 8, 8, 8)
        lyt.setSpacing(8)
        scroll.setWidget(content)

        # ---- Model status ----
        lyt.addWidget(self._header("MODEL STATUS"))
        self._status_lbl = QLabel("NOT LOADED")
        self._status_lbl.setStyleSheet(FIELD_SS)
        lyt.addWidget(self._status_lbl)

        self._input_lbl    = QLabel("")
        self._input_lbl.setStyleSheet(FIELD_SS)
        self._accuracy_lbl = QLabel("")
        self._accuracy_lbl.setStyleSheet(FIELD_SS)
        lyt.addWidget(self._input_lbl)
        lyt.addWidget(self._accuracy_lbl)

        load_btn = QPushButton("LOAD MODEL")
        load_btn.setStyleSheet(btn_style('default'))
        load_btn.clicked.connect(self._load_model)
        lyt.addWidget(load_btn)

        lyt.addWidget(self._divider())

        # ---- Classify ROI ----
        lyt.addWidget(self._header("CLASSIFY ROI"))
        self._classify_btn = QPushButton("RUN CLASSIFIER")
        self._classify_btn.setStyleSheet(operation_btn_style())
        self._classify_btn.setEnabled(False)
        self._classify_btn.clicked.connect(self._apply_classify)
        lyt.addWidget(self._classify_btn)

        self._bar_canvas = _BarCanvas()
        lyt.addWidget(self._bar_canvas)

        self._top_lbl = QLabel("---")
        self._top_lbl.setStyleSheet(FIELD_SS)
        self._top_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lyt.addWidget(self._top_lbl)

        lyt.addWidget(self._divider())

        # ---- Grad-CAM ----
        lyt.addWidget(self._header("GRAD-CAM"))
        gc_row = QHBoxLayout()
        gc_row.addWidget(self._lbl("ALPHA"))
        self._gc_alpha = QDoubleSpinBox()
        self._gc_alpha.setStyleSheet(SPINBOX_SS)
        self._gc_alpha.setRange(0.1, 1.0)
        self._gc_alpha.setSingleStep(0.1)
        self._gc_alpha.setValue(0.5)
        gc_row.addWidget(self._gc_alpha)
        lyt.addLayout(gc_row)

        self._gradcam_btn = QPushButton("GENERATE GRAD-CAM")
        self._gradcam_btn.setStyleSheet(operation_btn_style())
        self._gradcam_btn.setEnabled(False)
        self._gradcam_btn.clicked.connect(self._apply_gradcam)
        lyt.addWidget(self._gradcam_btn)

        lyt.addWidget(self._divider())

        # ---- ScoreCAM ----
        lyt.addWidget(self._header("SCORE-CAM"))
        sc_row = QHBoxLayout()
        sc_row.addWidget(self._lbl("ALPHA"))
        self._sc_alpha = QDoubleSpinBox()
        self._sc_alpha.setStyleSheet(SPINBOX_SS)
        self._sc_alpha.setRange(0.1, 1.0)
        self._sc_alpha.setSingleStep(0.1)
        self._sc_alpha.setValue(0.5)
        sc_row.addWidget(self._sc_alpha)
        lyt.addLayout(sc_row)

        self._scorecam_btn = QPushButton("GENERATE SCORE-CAM")
        self._scorecam_btn.setStyleSheet(operation_btn_style())
        self._scorecam_btn.setEnabled(False)
        self._scorecam_btn.clicked.connect(self._apply_scorecam)
        lyt.addWidget(self._scorecam_btn)

        lyt.addWidget(self._divider())

        # ---- Enhancement suggestions ----
        lyt.addWidget(self._header("SUGGESTIONS"))
        self._analyze_btn = QPushButton("ANALYZE IMAGE")
        self._analyze_btn.setStyleSheet(btn_style('default'))
        self._analyze_btn.clicked.connect(self._analyze_suggestions)
        lyt.addWidget(self._analyze_btn)

        self._suggestions_container = QWidget()
        chips_lyt = QVBoxLayout(self._suggestions_container)
        chips_lyt.setContentsMargins(0, 4, 0, 0)
        chips_lyt.setSpacing(4)
        self._chip_btns: list[QPushButton] = []
        for name in _SUGGESTIONS:
            btn = QPushButton(name)
            btn.setStyleSheet(btn_style('default'))
            btn.setVisible(False)
            btn.clicked.connect(lambda _checked, n=name: self.suggestion_clicked.emit(n))
            self._chip_btns.append(btn)
            chips_lyt.addWidget(btn)
        lyt.addWidget(self._suggestions_container)

        lyt.addStretch()

    def _lbl(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(FIELD_SS)
        return lbl

    def _header(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(HEADER_SS)
        return lbl

    def _divider(self) -> QFrame:
        p = _get_theme()
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {p['BORDER']};")
        return line

    # ------------------------------------------------------------------
    # Metadata / model loading
    # ------------------------------------------------------------------

    def _start_preload(self):
        from processing.deep_learning import predictor

        class _DummyState:
            def get_base_image(self):
                return np.zeros((4, 4, 3), dtype=np.uint8)

        def _preload_fn(image):
            predictor.preload()
            return image

        self._preload_worker = PipelineWorker(_preload_fn, "_preload", _DummyState())
        self._preload_worker.finished.connect(self._on_preload_done)
        self._preload_worker.error.connect(self._on_preload_error)
        self._preload_worker.start()

    def _on_preload_done(self, _op: str, _result):
        from processing.deep_learning import predictor
        if predictor.is_ready():
            p = _get_theme()
            self._status_lbl.setText("READY")
            self._status_lbl.setStyleSheet(
                f"color: {p['ACCENT']}; font-family: 'JetBrains Mono', Consolas, monospace; font-size: 9px;"
            )
            self._model_loaded = True
            self._refresh()

    def _on_preload_error(self, _msg: str):
        pass

    def _load_metadata(self):
        if not os.path.exists(_METADATA_JSON):
            return
        try:
            import json
            global _CLASS_LABELS
            with open(_METADATA_JSON) as f:
                meta = json.load(f)
            self._input_size = meta.get('input_size', [224, 224])
            acc = meta.get('test_accuracy', None)
            classes = meta.get('classes', _CLASS_LABELS)
            self._input_lbl.setText(f"Input: {self._input_size[0]}×{self._input_size[1]}")
            if acc is not None:
                self._accuracy_lbl.setText(f"Test acc: {acc*100:.1f}%")
            if classes:
                _CLASS_LABELS = classes
        except Exception:
            pass

    def _load_model(self):
        p = _get_theme()
        if not os.path.exists(_CLASSIFIER_H5):
            self._status_lbl.setText("WEIGHTS NOT FOUND")
            self._status_lbl.setStyleSheet(
                f"color: {p['RED']}; font-family: 'JetBrains Mono', Consolas, monospace; font-size: 9px;"
            )
            return
        try:
            from processing.deep_learning.classifier import load_model as _load
            _load(_CLASSIFIER_H5)
            self._model_loaded = True
            self._status_lbl.setText("LOADED")
            self._status_lbl.setStyleSheet(
                f"color: {p['ACCENT']}; font-family: 'JetBrains Mono', Consolas, monospace; font-size: 9px;"
            )
        except Exception as e:
            self._status_lbl.setText(f"LOAD ERROR")
            self._status_lbl.setStyleSheet(
                f"color: {p['RED']}; font-family: 'JetBrains Mono', Consolas, monospace; font-size: 9px;"
            )
            self.error_occurred.emit(str(e))
        self._refresh()

    # ------------------------------------------------------------------
    # Enable/disable buttons based on state
    # ------------------------------------------------------------------

    def _refresh(self):
        has_image = self._state is not None and self._state.current() is not None
        has_roi   = self._roi is not None
        ready     = has_image and has_roi and self._model_loaded
        self._classify_btn.setEnabled(ready)
        self._gradcam_btn.setEnabled(ready)
        self._scorecam_btn.setEnabled(ready)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _start_worker(self, fn, op_name: str, on_done, btn=None, btn_label=None, **kwargs):
        if self._state is None or (self._worker and self._worker.isRunning()):
            return
        if btn is not None:
            btn.setEnabled(False)
            btn.setText("⟳ Processing...")
        self._worker = PipelineWorker(fn, op_name, self._state, **kwargs)
        self._worker.finished.connect(on_done)
        self._worker.error.connect(self.error_occurred)
        if btn is not None:
            orig = btn_label or op_name
            self._worker.finished.connect(lambda *_: (btn.setEnabled(True), btn.setText(orig)))
            self._worker.error.connect(lambda *_: (btn.setEnabled(True), btn.setText(orig)))
        self._worker.start()

    def _apply_classify(self):
        if self._state is None:
            self.error_occurred.emit('No state — load an image first')
            return
        img = self._state.get_base_image()
        if img is None or not isinstance(img, np.ndarray):
            self.error_occurred.emit(f'No image loaded (got {type(img).__name__})')
            return
        if self._roi is None:
            self.error_occurred.emit('Select an ROI first')
            return

        x, y, w, h = self._roi
        H_img, W_img = img.shape[:2]
        x = max(0, min(int(x), W_img - 1))
        y = max(0, min(int(y), H_img - 1))
        w = max(1, min(int(w), W_img - x))
        h = max(1, min(int(h), H_img - y))
        roi_crop = img[y:y + h, x:x + w]

        if roi_crop.size == 0 or roi_crop.ndim < 2:
            self.error_occurred.emit(f'ROI is empty: shape={roi_crop.shape}')
            return

        self._pending_roi = np.array(roi_crop, dtype=np.uint8, copy=True)
        import processing.deep_learning.predictor as _pred_mod
        pending = self._pending_roi

        def _classify_fn(image):
            if pending is None or not isinstance(pending, np.ndarray):
                raise ValueError(f'pending_roi invalid: {type(pending).__name__}')
            _pred_mod.predict(pending)
            from utils.image_utils import normalize_to_uint8
            return normalize_to_uint8(pending)

        self._start_worker(_classify_fn, "AI Classify", self._on_classify_done,
                           btn=self._classify_btn, btn_label="RUN CLASSIFIER")

    def _on_classify_done(self, _op: str, result: np.ndarray):
        import processing.deep_learning.predictor as _pred_mod
        data = _pred_mod.get_last_prediction()
        probs: list[float] = []
        if isinstance(data, dict):
            all_scores = data.get('all_scores', {})
            if isinstance(all_scores, dict):
                probs = [float(v) for v in all_scores.values()]
            elif isinstance(all_scores, (list, np.ndarray)):
                probs = [float(v) for v in all_scores]
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            inner = data[0] if isinstance(data[0], (list, np.ndarray)) else data
            probs = [float(v) for v in inner]
        elif isinstance(data, np.ndarray):
            probs = data.flatten().tolist()

        if probs and len(probs) == len(_CLASS_LABELS):
            self._bar_canvas.plot(probs, _CLASS_LABELS)
            top_idx = int(np.argmax(probs))
            top_conf = probs[top_idx]
            self._top_lbl.setText(f"{_CLASS_LABELS[top_idx]}  {top_conf*100:.1f}%")
            self.classification_done.emit(
                [{'label': _CLASS_LABELS[i], 'prob': probs[i]} for i in range(len(probs))]
            )
        self.ai_applied.emit("Classify ROI", result)

    def _apply_gradcam(self):
        if self._state is None or self._roi is None:
            return
        self._gradcam_ref = [None]
        ref   = self._gradcam_ref
        roi   = self._roi
        alpha = self._gc_alpha.value()

        def _fn(image):
            return _gradcam_fn(image, roi, alpha, ref)

        self._start_worker(_fn, "Grad-CAM", lambda n, r: self.ai_applied.emit(n, r), btn=self._gradcam_btn, btn_label="GENERATE GRAD-CAM")

    def _apply_scorecam(self):
        if self._state is None or self._roi is None:
            return
        self._scorecam_ref = [None]
        ref   = self._scorecam_ref
        roi   = self._roi
        alpha = self._sc_alpha.value()

        def _fn(image):
            return _scorecam_fn(image, roi, alpha, ref)

        self._start_worker(_fn, "Score-CAM", lambda n, r: self.ai_applied.emit(n, r), btn=self._scorecam_btn, btn_label="GENERATE SCORE-CAM")

    def _analyze_suggestions(self):
        if self._state is None:
            return
        img = self._state.current()
        if img is None:
            return

        show_flags = [False] * len(_SUGGESTIONS)
        if img.ndim == 3:
            gray = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1]
                    + 0.114 * img[:, :, 2]).astype(np.uint8)
        else:
            gray = img

        std = float(gray.std())
        mean = float(gray.mean())

        if std < 30:
            show_flags[2] = True  # CLAHE
            show_flags[3] = True  # Histogram EQ
        if std > 60:
            show_flags[0] = True  # Denoise
        if mean < 80 or mean > 180:
            show_flags[1] = True  # Sharpen
        if std > 40:
            show_flags[4] = True  # Median

        for btn, visible in zip(self._chip_btns, show_flags):
            btn.setVisible(visible)
