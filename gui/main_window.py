import time
import logging

from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QTabWidget, QWidget, QVBoxLayout,
    QLabel, QMessageBox, QApplication, QFileDialog,
    QFormLayout, QGroupBox,
)
from PyQt6.QtCore import Qt, QSettings, QTimer
from PyQt6.QtGui import QAction, QKeySequence

from utils.pipeline_state import PipelineState
from utils.error_handler import setup_logger

import gui.theme as _theme
from gui.styles import build as _build_ss
from gui.image_viewer import ImageViewer
from gui.pipeline_panel import PipelinePanel
from gui.filter_panel import FilterPanel
from gui.histogram_panel import HistogramPanel
from gui.noise_panel import NoisePanel
from gui.morphology_panel import MorphologyPanel
from gui.fourier_panel import FourierPanel
from gui.template_panel import TemplatePanel
from gui.cv_panel import CVPanel
from gui.ai_panel import AIPanel

_log = setup_logger()
_SETTINGS_ORG = 'ciaw'
_SETTINGS_APP = 'ciaw'
_SETTINGS_THEME = 'ciaw/theme'


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = PipelineState()
        self._redo_stack: list = []
        self._last_op_ms: float = 0.0
        self._op_start: float = 0.0
        self._active_workers: int = 0
        self._spinner_idx: int = 0
        self._theme_name: str = 'Dark Lime'
        self._save_path: str | None = None

        self._apply_saved_theme()
        self._build_ui()
        self._connect_signals()
        self._start_spinner()
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_saved_theme(self):
        settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        name = settings.value(_SETTINGS_THEME, 'Dark Lime')
        if name not in _theme.names():
            name = 'Dark Lime'
        self._theme_name = name
        _theme.set_theme(name)
        app = QApplication.instance()
        if app:
            app.setStyleSheet(_build_ss())

    def _apply_theme(self, name: str):
        if name not in _theme.names():
            return
        self._theme_name = name
        _theme.set_theme(name)
        app = QApplication.instance()
        if app:
            app.setStyleSheet(_build_ss())
        p = _theme.get()
        self._image_viewer.theme_changed(p)
        if hasattr(self._fourier_panel, 'theme_changed'):
            self._fourier_panel.theme_changed(p)
        QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(_SETTINGS_THEME, name)
        self._update_status_bar()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.setWindowTitle("Clinical Image Analysis Workbench")
        self.resize(1400, 900)
        self._build_menu()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # --- Left panel (320px) ---
        left = QWidget()
        left_lyt = QVBoxLayout(left)
        left_lyt.setContentsMargins(0, 0, 0, 0)
        left_lyt.setSpacing(0)

        self._pipeline_panel = PipelinePanel()
        left_lyt.addWidget(self._pipeline_panel, stretch=2)

        self._filter_panel    = FilterPanel()
        self._histogram_panel = HistogramPanel()
        self._noise_panel     = NoisePanel()
        self._morphology_panel = MorphologyPanel()
        self._fourier_panel   = FourierPanel()
        self._template_panel  = TemplatePanel()
        self._cv_panel        = CVPanel()
        self._ai_panel        = AIPanel()

        self._tab_widget = QTabWidget()
        self._tab_widget.setDocumentMode(True)
        self._tab_widget.addTab(self._filter_panel,     "Filter")
        self._tab_widget.addTab(self._histogram_panel,  "Histogram")
        self._tab_widget.addTab(self._noise_panel,      "Noise")
        self._tab_widget.addTab(self._morphology_panel, "Morphology")
        self._tab_widget.addTab(self._fourier_panel,    "Frequency")
        self._tab_widget.addTab(self._template_panel,   "Template")
        self._tab_widget.addTab(self._cv_panel,         "CV")
        self._tab_widget.addTab(self._ai_panel,         "AI")
        left_lyt.addWidget(self._tab_widget, stretch=3)

        for panel in (self._filter_panel, self._histogram_panel,
                      self._noise_panel, self._morphology_panel,
                      self._fourier_panel, self._template_panel,
                      self._cv_panel, self._ai_panel):
            panel.setEnabled(False)

        # --- Center: image viewer ---
        self._image_viewer = ImageViewer()

        # --- Right panel (280px, metadata + options) ---
        right = QWidget()
        right_lyt = QVBoxLayout(right)
        right_lyt.setContentsMargins(8, 8, 8, 8)
        right_lyt.setSpacing(6)
        p = _theme.get()

        hdr_ss = (
            f"color: {p['ACCENT']}; font-family: 'JetBrains Mono', Consolas, monospace; "
            f"font-size: 9px; font-weight: bold; letter-spacing: 2px; "
            f"padding-bottom: 4px; border-bottom: 1px solid {p['BORDER']};"
        )
        val_ss = (
            f"color: {p['TEXT']}; font-family: 'JetBrains Mono', Consolas, monospace; "
            f"font-size: 9px;"
        )
        key_ss = (
            f"color: {p['MUTED']}; font-family: 'JetBrains Mono', Consolas, monospace; "
            f"font-size: 9px;"
        )

        meta_hdr = QLabel("METADATA")
        meta_hdr.setStyleSheet(hdr_ss)
        right_lyt.addWidget(meta_hdr)

        meta_form = QFormLayout()
        meta_form.setSpacing(3)
        meta_form.setContentsMargins(0, 4, 0, 4)

        def _meta_lbl(text="—"):
            lbl = QLabel(text)
            lbl.setStyleSheet(val_ss)
            lbl.setWordWrap(True)
            return lbl

        def _key_lbl(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(key_ss)
            return lbl

        self._meta_file   = _meta_lbl()
        self._meta_dims   = _meta_lbl()
        self._meta_ch     = _meta_lbl()
        self._meta_dtype  = _meta_lbl()
        self._meta_range  = _meta_lbl()
        self._meta_size   = _meta_lbl()

        for key, val in [
            ("File",    self._meta_file),
            ("Dims",    self._meta_dims),
            ("Ch",      self._meta_ch),
            ("DType",   self._meta_dtype),
            ("Range",   self._meta_range),
            ("Size",    self._meta_size),
        ]:
            meta_form.addRow(_key_lbl(key), val)

        right_lyt.addLayout(meta_form)
        right_lyt.addStretch()

        splitter.addWidget(left)
        splitter.addWidget(self._image_viewer)
        splitter.addWidget(right)
        splitter.setSizes([320, 800, 280])

        self.setCentralWidget(splitter)
        self._build_status_bar()

    def _build_menu(self):
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("File")
        self._act(file_menu, "Open",    "Ctrl+O", self._open_file)
        self._act(file_menu, "Save",    "Ctrl+S", self._save_file)
        self._act(file_menu, "Save As", None,     self._save_file_as)

        # Edit
        edit_menu = mb.addMenu("Edit")
        self._act(edit_menu, "Undo",        "Ctrl+Z", self._handle_undo)
        self._act(edit_menu, "Redo",        "Ctrl+Y", self._handle_redo)
        edit_menu.addSeparator()
        self._act(edit_menu, "Preferences", None,     lambda: None)

        # View → Theme submenu
        view_menu = mb.addMenu("View")
        theme_menu = view_menu.addMenu("Theme")
        for name in _theme.names():
            act = QAction(name, self)
            act.triggered.connect(lambda _, n=name: self._apply_theme(n))
            theme_menu.addAction(act)

        # Help
        help_menu = mb.addMenu("Help")
        self._act(help_menu, "About", None, self._show_about)

    def _act(self, menu, label: str, shortcut, slot):
        act = QAction(label, self)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        if slot:
            act.triggered.connect(slot)
        menu.addAction(act)
        return act

    def _build_status_bar(self):
        p = _theme.get()
        seg_ss = (
            f"QLabel {{ color: {p['MUTED']}; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; "
            f"font-size: 9px; padding: 0 5px; }}"
        )
        sep_ss = (
            f"QLabel {{ color: {p['BORDER2']}; font-size: 9px; padding: 0 1px; }}"
        )

        def seg(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet(seg_ss)
            return lbl

        def sep() -> QLabel:
            lbl = QLabel("|")
            lbl.setStyleSheet(sep_ss)
            return lbl

        self._sb_op     = seg("OP: —")
        self._sb_interp = seg("NN")
        self._sb_pipe   = seg("CUM")
        self._sb_stack  = seg("0 ops")
        self._sb_mem    = seg("0.0 MB")
        self._sb_worker = seg("  ")
        self._sb_time   = seg("—")
        self._sb_theme  = seg(self._theme_name)

        sb = self.statusBar()
        for w in [
            self._sb_op, sep(), self._sb_interp, sep(), self._sb_pipe, sep(),
            self._sb_stack, sep(), self._sb_mem, sep(), self._sb_worker, sep(),
            self._sb_time, sep(), self._sb_theme,
        ]:
            sb.addPermanentWidget(w)

    def _update_metadata_panel(self, image: np.ndarray, path: str | None = None):
        if image is None:
            return
        h, w = image.shape[:2]
        ch = image.shape[2] if image.ndim == 3 else 1
        self._meta_file.setText(path.split("\\")[-1].split("/")[-1] if path else "—")
        self._meta_dims.setText(f"{w} × {h}")
        self._meta_ch.setText(str(ch))
        self._meta_dtype.setText(str(image.dtype))
        self._meta_range.setText(f"{int(image.min())} – {int(image.max())}")
        kb = image.nbytes / 1024
        self._meta_size.setText(f"{kb:.0f} KB" if kb < 1024 else f"{kb/1024:.1f} MB")

    def _update_status_bar(self, *_args):
        img = self._state.current()
        stack_len = len(self._state.get_stack_names())
        mode = self._state.get_mode()
        mem_mb = img.nbytes / 1024 / 1024 if img is not None else 0.0
        interp = getattr(self._image_viewer, '_interp', 'nearest')

        self._sb_op.setText(f"OP: {self._state.current_op()}")
        self._sb_interp.setText("BL" if interp == 'bilinear' else "NN")
        self._sb_pipe.setText("CUM" if mode == 'cumulative' else "IND")
        self._sb_stack.setText(f"{stack_len} ops")
        self._sb_mem.setText(f"{mem_mb:.1f} MB")
        self._sb_time.setText(
            f"{self._last_op_ms:.0f} ms" if self._last_op_ms else "—"
        )
        self._sb_theme.setText(self._theme_name)

    # ------------------------------------------------------------------
    # Spinner (WORKER segment)
    # ------------------------------------------------------------------

    _SPINNER = ("●◦◦◦", "◦●◦◦", "◦◦●◦", "◦◦◦●")

    def _start_spinner(self):
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(150)
        self._spinner_timer.timeout.connect(self._tick_spinner)
        self._spinner_timer.start()

    def _tick_spinner(self):
        if self._active_workers > 0:
            self._spinner_idx = (self._spinner_idx + 1) % 4
            self._sb_worker.setText(self._SPINNER[self._spinner_idx])
        else:
            self._sb_worker.setText("  ")

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        pp = self._pipeline_panel
        pp.undo_requested.connect(self._handle_undo)
        pp.redo_requested.connect(self._handle_redo)
        pp.reset_requested.connect(self._handle_reset)
        pp.mode_changed.connect(self._state.set_mode)
        pp.checkpoint_save_requested.connect(self._handle_checkpoint_save)
        pp.checkpoint_restore_requested.connect(self._handle_checkpoint_restore)
        self._image_viewer.interp_changed.connect(self._update_status_bar)
        self._image_viewer.roi_selected.connect(self._on_roi_selected)

        # pass state to all panels
        for panel in (self._filter_panel, self._histogram_panel,
                      self._noise_panel, self._morphology_panel,
                      self._fourier_panel, self._template_panel,
                      self._cv_panel, self._ai_panel):
            panel.set_state(self._state)

        # operation results → pipeline
        self._filter_panel.filter_applied.connect(self._on_operation_done)
        self._histogram_panel.operation_applied.connect(self._on_operation_done)
        self._noise_panel.noise_applied.connect(self._on_operation_done)
        self._morphology_panel.morphology_applied.connect(self._on_operation_done)
        self._morphology_panel.segmentation_applied.connect(self._on_operation_done)
        self._fourier_panel.fourier_applied.connect(self._on_operation_done)
        self._template_panel.template_applied.connect(self._on_operation_done)
        self._cv_panel.cv_applied.connect(self._on_operation_done)
        self._ai_panel.ai_applied.connect(self._on_operation_done)

        # errors
        for panel in (self._filter_panel, self._histogram_panel,
                      self._noise_panel, self._morphology_panel,
                      self._fourier_panel, self._template_panel,
                      self._cv_panel, self._ai_panel):
            panel.error_occurred.connect(self._on_error)

        # interpolation
        self._filter_panel.interp_changed.connect(self._image_viewer.set_interp)

        # AI classification result → status bar
        self._ai_panel.classification_done.connect(self._on_classification_done)

        # theme propagation
        self._fourier_panel_theme = self._fourier_panel

    # ------------------------------------------------------------------
    # Core operation lifecycle  (called by panel workers via _track_worker)
    # ------------------------------------------------------------------

    def _on_roi_selected(self, x: int, y: int, w: int, h: int):
        self._noise_panel.set_roi(x, y, w, h)
        self._cv_panel.set_roi(x, y, w, h)
        self._ai_panel.set_roi(x, y, w, h)

    def _on_classification_done(self, results: list):
        if results:
            top = max(results, key=lambda d: d.get('prob', 0))
            label = top.get('label', '?')
            conf  = top.get('prob', 0.0)
            self._sb_op.setText(f"AI: {label} {conf*100:.1f}%")

    def _on_operation_done(self, op_name: str, result):
        if result is None:
            return
        if self._op_start:
            self._last_op_ms = (time.monotonic() - self._op_start) * 1000
            self._op_start = 0.0
        self._active_workers = max(0, self._active_workers - 1)
        self._redo_stack.clear()
        before = self._state.current()
        self._state.push(op_name, result)
        self._image_viewer.set_image(result)
        if before is not None:
            self._image_viewer.store_before(before)
        self._update_metadata_panel(result)
        if hasattr(self._histogram_panel, 'refresh'):
            self._histogram_panel.refresh()
        if hasattr(self._fourier_panel, 'update_display'):
            self._fourier_panel.update_display(result)
        self._pipeline_panel.refresh(
            self._state.get_stack_names(), self._state.get_mode()
        )
        self._update_status_bar()

    def _on_error(self, msg: str):
        _log.error("Worker error: %s", msg)
        self._active_workers = max(0, self._active_workers - 1)
        QMessageBox.warning(self, "Operation Error", msg)

    def _track_worker(self, worker):
        """Register a PipelineWorker: increments active count, starts timing."""
        self._active_workers += 1
        self._op_start = time.monotonic()
        worker.finished.connect(self._on_operation_done)
        worker.error.connect(self._on_error)

    # ------------------------------------------------------------------
    # Undo / Redo / Reset
    # ------------------------------------------------------------------

    def _handle_undo(self):
        if not self._state.has_image():
            return
        if self._state.get_stack_names():
            self._redo_stack.append(
                (self._state.current_op(), self._state.current().copy())
            )
        result = self._state.undo()
        if result is not None:
            self._image_viewer.set_image(result)
        self._pipeline_panel.refresh(
            self._state.get_stack_names(), self._state.get_mode()
        )
        self._update_status_bar()

    def _handle_redo(self):
        if not self._redo_stack:
            return
        op_name, img = self._redo_stack.pop()
        self._state.push(op_name, img)
        self._image_viewer.set_image(img)
        self._pipeline_panel.refresh(
            self._state.get_stack_names(), self._state.get_mode()
        )
        self._update_status_bar()

    def _handle_reset(self):
        self._state.reset()
        self._redo_stack.clear()
        img = self._state.current()
        if img is not None:
            self._image_viewer.set_image(img)
        self._pipeline_panel.refresh(
            self._state.get_stack_names(), self._state.get_mode()
        )
        self._update_status_bar()

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def _handle_checkpoint_save(self, slot: str):
        if not self._state.has_image():
            return
        try:
            self._state.save_checkpoint(slot)
        except ValueError as exc:
            _log.warning("Checkpoint save failed: %s", exc)

    def _handle_checkpoint_restore(self, slot: str):
        img = self._state.restore_checkpoint(slot)
        if img is not None:
            self._image_viewer.set_image(img)
            self._pipeline_panel.refresh(
                self._state.get_stack_names(), self._state.get_mode()
            )
            self._update_status_bar()

    # ------------------------------------------------------------------
    # File I/O  (processing.io imported lazily to keep top-level clean)
    # ------------------------------------------------------------------

    def _open_file(self):
        from processing.io.image_loader import load_image
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.dcm *.jpg *.jpeg *.png *.bmp);;All Files (*)",
        )
        if not path:
            return
        result = load_image(path)
        if result is None:
            return
        image, _metadata = result
        self._state.set_original(image)
        self._redo_stack.clear()
        self._image_viewer.set_image(image)
        self._image_viewer.store_before(image)
        self._update_metadata_panel(image, path)

        for panel in (self._filter_panel, self._histogram_panel,
                      self._noise_panel, self._morphology_panel,
                      self._fourier_panel, self._template_panel,
                      self._cv_panel, self._ai_panel):
            panel.setEnabled(True)
            panel.set_state(self._state)

        if hasattr(self._histogram_panel, 'refresh'):
            self._histogram_panel.refresh()
        if hasattr(self._fourier_panel, 'update_display'):
            self._fourier_panel.update_display(image)

        self._pipeline_panel.refresh(
            self._state.get_stack_names(), self._state.get_mode()
        )
        self._update_status_bar()
        self.statusBar().showMessage(f"Opened: {path}", 3000)

    def _save_file(self):
        if self._save_path:
            self._do_save(self._save_path)
        else:
            self._save_file_as()

    def _save_file_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "",
            "JPEG (*.jpg);;PNG (*.png);;BMP (*.bmp);;All Files (*)",
        )
        if path:
            self._save_path = path
            self._do_save(path)

    def _do_save(self, path: str):
        from processing.io.image_saver import save_image
        img = self._state.current()
        if img is None:
            QMessageBox.warning(self, "Save", "No image to save.")
            return
        save_image(img, path)
        self.statusBar().showMessage(f"Saved: {path}", 3000)

    # ------------------------------------------------------------------
    # About
    # ------------------------------------------------------------------

    def _show_about(self):
        QMessageBox.about(
            self,
            "About CIAW",
            "Clinical Image Analysis Workbench\n"
            "Digital Image Processing Project\n\n"
            "Spatial · Frequency · Morphology · Noise\n"
            "Deep Learning · Histogram · Segmentation",
        )
