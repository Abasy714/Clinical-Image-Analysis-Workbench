# STATUS: IMPLEMENTED
"""
Main application window for the Clinical Image Analysis Workbench.
Manages the tabbed interface, global pipeline state, and communication between panels.
"""

import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                              QSplitter, QMenuBar, QStatusBar, QLabel,
                              QFileDialog, QApplication, QTabWidget, QFrame)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QFont

from gui.styles import (BG, PANEL, PANEL2, INPUT, BORDER, BORDER2,
                        ACCENT, ACCENT2, TEXT, MUTED, MUTED2, apply_styles)
from utils import (PipelineState, wrap_errors, show_error_dialog, setup_logger,)

APP_TITLE = "Clinical Image Analysis Workbench"
APP_VERSION = "1.0.0"


class MainWindow(QMainWindow):
    """Main application window — owns all panels, pipeline state, and signal routing."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(1280, 720)

        self.pipeline = PipelineState()
        self.logger = setup_logger()

        self._build_menu()
        self._build_central()
        self._build_status_bar()
        self._connect_signals()

        from PyQt6.QtGui import QShortcut, QKeySequence
        ba_shortcut = QShortcut(QKeySequence("B"), self)
        ba_shortcut.activated.connect(self._image_viewer.toggle_before_after)

        self._apply_stylesheet()

    # ------------------------------------------------------------------ build

    def _build_menu(self):
        mb = self.menuBar()
        mb.setStyleSheet(
            f"QMenuBar{{background:#0d0e0c;color:#8a8e84;border-bottom:1px solid {BORDER};padding:2px;}}"
            f"QMenuBar::item:selected{{background:{INPUT};color:{TEXT};}}"
            f"QMenu{{background:{PANEL};border:1px solid {BORDER};color:{TEXT};}}"
            f"QMenu::item:selected{{background:{INPUT};}}"
        )

        file_menu = mb.addMenu("File")
        open_act = QAction("Open…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(lambda: self.load_image())
        file_menu.addAction(open_act)

        save_act = QAction("Save…", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(lambda: self.save_image())
        file_menu.addAction(save_act)

        file_menu.addSeparator()
        exit_act = QAction("Exit", self)
        exit_act.setShortcut("Ctrl+Q")
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        view_menu = mb.addMenu("View")

        pipe_menu = mb.addMenu("Pipeline")
        undo_act = QAction("Undo", self)
        undo_act.setShortcut("Ctrl+Z")
        undo_act.triggered.connect(self._do_undo)
        pipe_menu.addAction(undo_act)

        reset_act = QAction("Reset to original", self)
        reset_act.triggered.connect(self._do_reset)
        pipe_menu.addAction(reset_act)

        mb.addMenu("Tools")
        mb.addMenu("Help")

    def _build_central(self):
        root = QWidget()
        root.setStyleSheet(f"background:{BG};")
        self.setCentralWidget(root)
        vl = QVBoxLayout(root)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        # title bar
        title_bar = self._make_title_bar()
        vl.addWidget(title_bar)

        # splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(
            f"QSplitter::handle{{background:{BORDER};}}"
        )

        # left panel: metadata + pipeline
        left = QWidget()
        left.setFixedWidth(210)
        left.setStyleSheet("""
            QWidget {
                background-color: #181917;
                border-right: 1px solid #2c2e2a;
            }
        """)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(0)

        from gui.metadata_panel import MetadataPanel
        from gui.pipeline_panel import PipelinePanel
        self._metadata_panel = MetadataPanel()
        self._pipeline_panel = PipelinePanel(self.pipeline)
        ll.addWidget(self._metadata_panel)
        ll.addWidget(self._pipeline_panel, stretch=1)
        splitter.addWidget(left)

        # center: image viewer
        from gui.image_viewer import ImageViewer
        self._image_viewer = ImageViewer()
        splitter.addWidget(self._image_viewer)

        # right panel: tabs
        right_tabs = QTabWidget()
        right_tabs.setFixedWidth(270)
        right_tabs.setStyleSheet("""
            QWidget {
                background-color: #181917;
            }
            QTabWidget::pane {
                background: #181917;
                border: none;
                border-left: 1px solid #2c2e2a;
            }
            QTabBar {
                background: #111210;
                border-bottom: 1px solid #2c2e2a;
            }
            QTabBar::tab {
                background: #111210;
                color: #6b6f65;
                border: none;
                border-bottom: 2px solid transparent;
                padding: 8px 0;
                font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
                font-size: 9px;
                font-weight: bold;
                letter-spacing: 2px;
                min-width: 54px;
            }
            QTabBar::tab:selected {
                color: #c8f135;
                border-bottom: 2px solid #c8f135;
                background: #111210;
            }
            QTabBar::tab:hover:!selected {
                color: #eceee8;
                background: #181917;
            }
        """)

        from gui.filter_panel import FilterPanel
        from gui.histogram_panel import HistogramPanel
        from gui.fourier_panel import FourierPanel
        from gui.morphology_panel import MorphologyPanel
        from gui.noise_panel import NoisePanel

        self._filter_panel = FilterPanel()
        self._hist_panel = HistogramPanel()
        self._fourier_panel = FourierPanel()
        self._morph_panel = MorphologyPanel()
        self._noise_panel = NoisePanel()

        right_tabs.addTab(self._filter_panel, "FILTER")
        right_tabs.addTab(self._hist_panel, "HIST")
        right_tabs.addTab(self._fourier_panel, "FREQ")
        right_tabs.addTab(self._morph_panel, "MORPH")
        right_tabs.addTab(self._noise_panel, "NOISE")
        splitter.addWidget(right_tabs)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        self._right_tabs = right_tabs
        vl.addWidget(splitter, stretch=1)

    def _make_title_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(36)
        bar.setStyleSheet(
            f"background:#0d0e0c;border-bottom:1px solid {BORDER};"
        )
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(12, 0, 12, 0)

        ver_lbl = QLabel(f"v{APP_VERSION}")
        ver_lbl.setStyleSheet(f"color:{MUTED2};font-size:9px;")
        hl.addWidget(ver_lbl)

        hl.addStretch()

        title_lbl = QLabel(f"<span style='color:{ACCENT}'>●</span>  {APP_TITLE}")
        title_lbl.setStyleSheet(f"color:{TEXT};font-size:11px;font-weight:bold;")
        hl.addWidget(title_lbl)

        hl.addStretch()

        self._session_lbl = QLabel("SESSION  READY")
        self._session_lbl.setStyleSheet(
            f"background:{INPUT};color:{ACCENT};font-size:8px;font-weight:bold;"
            f"letter-spacing:.1em;padding:2px 8px;border-radius:8px;"
        )
        hl.addWidget(self._session_lbl)

        return bar

    def _build_status_bar(self):
        sb = self.statusBar()
        sb.setStyleSheet(
            f"QStatusBar{{background:#0d0e0c;border-top:1px solid {BORDER};color:{MUTED};font-size:10px;}}"
            f"QStatusBar::item{{border:none;}}"
        )

        def _seg(text: str, fixed_w: int = 0) -> QLabel:
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color:{MUTED};font-size:9px;padding:0 8px;"
                f"border-right:1px solid {BORDER};"
            )
            if fixed_w:
                lbl.setFixedWidth(fixed_w)
            return lbl

        self._sb_op    = _seg("OP: —", 160)
        self._sb_dim   = _seg("—×—", 90)
        self._sb_zoom  = _seg("100%", 50)
        self._sb_zoom.setStyleSheet(
            f"color:{ACCENT};font-size:10px;font-weight:bold;padding:0 8px;"
            f"border-right:1px solid {BORDER};"
        )
        self._sb_interp = _seg("NN", 40)
        self._sb_roi   = _seg("ROI: none", 100)
        self._sb_pipe_mode = _seg("PIPE: Cumulative", 130)
        self._sb_pipe_mode.setStyleSheet(
            f"color:{ACCENT};font-size:10px;font-weight:bold;padding:0 8px;"
            f"border-right:1px solid {BORDER};"
        )

        spacer = QWidget()
        spacer.setSizePolicy(
            spacer.sizePolicy().horizontalPolicy(),
            spacer.sizePolicy().verticalPolicy()
        )

        self._sb_pipe  = _seg("STACK: 0", 80)
        self._sb_mem   = _seg("MEM: 0KB", 80)
        self._sb_ready = QLabel("● READY")
        self._sb_ready.setStyleSheet(f"color:{ACCENT};font-size:9px;font-weight:bold;padding:0 8px;")

        for w in (self._sb_op, self._sb_dim, self._sb_zoom, self._sb_interp, self._sb_roi, self._sb_pipe_mode):
            sb.addWidget(w)
        sb.addWidget(spacer, 1)
        for w in (self._sb_pipe, self._sb_mem, self._sb_ready):
            sb.addPermanentWidget(w)

    def _connect_signals(self):
        # filter panel
        self._filter_panel.apply_btn.clicked.connect(self._on_filter_apply)
        self._filter_panel.kernel_btn.clicked.connect(self._on_kernel_modal)
        self._filter_panel.filter_applied.connect(self.on_operation_applied)

        # histogram panel
        self._hist_panel.apply_btn.clicked.connect(self._on_hist_apply)
        self._hist_panel.equalization_applied.connect(self.on_operation_applied)

        # fourier panel
        self._fourier_panel.apply_btn.clicked.connect(self._fourier_panel.on_apply_clicked)
        self._fourier_panel.notch_applied.connect(self.on_operation_applied)

        # morphology panel
        self._morph_panel.morphology_applied.connect(self.on_operation_applied)

        # noise panel
        self._noise_panel.inject_btn.clicked.connect(self._noise_panel.on_inject_clicked)
        self._noise_panel.noise_applied.connect(self.on_operation_applied)

        # pipeline panel
        self._pipeline_panel.undo_requested.connect(self._do_undo)
        self._pipeline_panel.reset_requested.connect(self._do_reset)
        self._pipeline_panel.checkpoint_save_requested.connect(self._do_checkpoint_save)
        self._pipeline_panel.checkpoint_restore_requested.connect(self._do_checkpoint_restore)
        self._pipeline_panel.mode_changed.connect(self._on_pipeline_mode_changed)

        # image viewer
        self._image_viewer.roi_selected.connect(self._on_roi_selected)
        self._image_viewer.zoom_changed.connect(
            lambda z: self._sb_zoom.setText(f"{z}%")
        )

        # tab change — compute spectrum only when Freq tab is selected
        self._right_tabs.currentChanged.connect(self._on_tab_changed)

    # ------------------------------------------------------------------ actions

    @wrap_errors
    def load_image(self, filepath: str = None):
        if filepath is None:
            import os
            images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Open Image", images_dir,
                "Medical Images (*.dcm *.jpg *.jpeg *.bmp);;All Files (*)"
            )
        if not filepath:
            return
        from processing.io import load_image
        image, metadata = load_image(filepath)
        if image is None:
            show_error_dialog("Load failed", f"Could not load: {filepath}")
            return
        self.pipeline.set_original(image)
        self._image_viewer.set_image(image)
        if hasattr(self._image_viewer, 'set_before_image'):
            self._image_viewer.set_before_image(image)
        self._metadata_panel.update_metadata(metadata)
        self._pipeline_panel.refresh_stack()
        self._pipeline_panel.update_checkpoints()
        # Phase 2 only — frequency domain disabled in Phase 1
        # self._fourier_panel.set_image(image)
        self._morph_panel.set_image(image)
        self._noise_panel.set_image(image)
        self._filter_panel.set_current_image(image)
        h, w = image.shape[:2]
        self._sb_dim.setText(f"{w}×{h}")
        self._sb_op.setText("OP: load")
        self._session_lbl.setText("SESSION  ACTIVE")
        self.logger.info("Loaded image: %s (%dx%d)", filepath, w, h)

    @wrap_errors
    def save_image(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "",
            "JPEG (*.jpg *.jpeg);;BMP (*.bmp);;PNG (*.png);;All Files (*)"
        )
        if not filepath:
            return
        from processing.io import save_image
        save_image(self.pipeline.current(), filepath)
        self.logger.info("Saved image: %s", filepath)

    def on_operation_applied(self, op_name: str, result: np.ndarray):
        self.pipeline.push(op_name, result)
        self._image_viewer.set_image(result)
        self._pipeline_panel.refresh_stack()
        self._pipeline_panel.update_checkpoints()
        base = self.pipeline.get_base_image()
        self._morph_panel.set_image(base)
        self._noise_panel.set_image(base)
        # Phase 2 only — frequency domain disabled in Phase 1
        # self._fourier_panel.set_image(result)
        self._filter_panel.set_current_image(base)

        stack_depth = len(self.pipeline.get_stack_names())
        self._sb_op.setText(f"OP: {op_name[:18]}")
        self._sb_pipe.setText(f"STACK: {stack_depth}")
        mem_kb = result.nbytes // 1024
        self._sb_mem.setText(f"MEM: {mem_kb}KB")

        self._image_viewer.show_processing_overlay(False)

        self.logger.info("Operation applied: %s", op_name)

    def update_status_bar(self, **kwargs):
        mapping = {
            "op":     self._sb_op,
            "dim":    self._sb_dim,
            "zoom":   self._sb_zoom,
            "interp": self._sb_interp,
            "roi":    self._sb_roi,
            "pipe":   self._sb_pipe,
            "mem":    self._sb_mem,
        }
        for key, val in kwargs.items():
            if key in mapping:
                mapping[key].setText(str(val))

    # ------------------------------------------------------------------ panel dispatch

    def _on_filter_apply(self):
        image = self.pipeline.get_base_image()
        if image is None:
            return
        self._image_viewer.show_processing_overlay(True)
        self._filter_panel.set_current_image(image)
        self._filter_panel.on_apply_clicked(image)

    def _on_kernel_modal(self):
        self._filter_panel.open_kernel_modal(self.pipeline.get_base_image())

    def _on_hist_apply(self):
        self._hist_panel.on_apply_clicked(self.pipeline.get_base_image())

    def _on_pipeline_mode_changed(self, mode: str):
        """Switch pipeline mode (cumulative/independent) and update UI."""
        self.pipeline.set_mode(mode)
        if mode == 'cumulative':
            self._sb_pipe_mode.setText("PIPE: Cumulative")
            self._sb_pipe_mode.setStyleSheet(
                f"color:{ACCENT};font-size:10px;font-weight:bold;padding:0 8px;"
                f"border-right:1px solid {BORDER};"
            )
        else:
            self._sb_pipe_mode.setText("PIPE: Independent")
            self._sb_pipe_mode.setStyleSheet(
                "color:#f5a623;font-size:10px;font-weight:bold;padding:0 8px;"
                f"border-right:1px solid {BORDER};"
            )
        # Update panels' source image to the new base
        base = self.pipeline.get_base_image()
        if base is not None:
            try:
                self._morph_panel.set_image(base)
            except Exception:
                pass
            try:
                self._noise_panel.set_image(base)
            except Exception:
                pass
            try:
                self._filter_panel.set_current_image(base)
            except Exception:
                pass
        self.logger.info(f"Pipeline mode: {mode}")

    def _on_roi_selected(self, roi):
        try:
            if roi is not None and roi.width() > 0 and roi.height() > 0:
                self._sb_roi.setText(f"ROI: {roi.width()}×{roi.height()}")

            image = self.pipeline.current()
            if image is None:
                return

            try:
                self._hist_panel.update_roi(image, roi)
            except Exception as e:
                import logging
                logging.getLogger('ciaw').error(f"histogram ROI not ready: {e}")

            try:
                self._noise_panel.update_roi_stats(image, roi)
            except Exception as e:
                import logging
                logging.getLogger('ciaw').error(f"noise ROI stats not ready: {e}")

        except Exception as e:
            import logging
            logging.getLogger('ciaw').error(f"_on_roi_selected error: {e}")

    def _do_undo(self):
        image = self.pipeline.undo()
        self._image_viewer.set_image(image)
        self._pipeline_panel.refresh_stack()
        stack_depth = len(self.pipeline.get_stack_names())
        self._sb_pipe.setText(f"STACK: {stack_depth}")

    def _do_reset(self):
        image = self.pipeline.reset()
        self._image_viewer.set_image(image)
        if hasattr(self._image_viewer, 'set_before_image'):
            self._image_viewer.set_before_image(image)
        self._pipeline_panel.refresh_stack()
        self._sb_op.setText("OP: reset")
        self._sb_pipe.setText("STACK: 0")

    def _do_checkpoint_save(self, slot: str):
        self.pipeline.save_checkpoint(slot)
        self._pipeline_panel.update_checkpoints()

    def _do_checkpoint_restore(self, slot: str):
        image = self.pipeline.restore_checkpoint(slot)
        if image is not None:
            self._image_viewer.set_image(image)
            self._pipeline_panel.refresh_stack()
            self._pipeline_panel.update_checkpoints()

    def _on_tab_changed(self, index: int):
        tab_text = self._right_tabs.tabText(index).strip().upper()
        if tab_text == "FREQ":
            image = self.pipeline.current()
            if image is not None:
                try:
                    from processing.frequency.spectrum import compute_spectrum  # noqa: F401
                    self._fourier_panel.set_image(image)
                except ImportError:
                    # Phase 2 not implemented yet — no dialog
                    pass
                except Exception:
                    pass

    # ------------------------------------------------------------------ stylesheet

    def _apply_stylesheet(self):
        self.setStyleSheet(
            f"QMainWindow{{background:{BG};}}"
            f"QWidget{{background:{BG};color:{TEXT};"
            f"font-family:'JetBrains Mono','Fira Code',Consolas,monospace;font-size:11px;}}"
        )
