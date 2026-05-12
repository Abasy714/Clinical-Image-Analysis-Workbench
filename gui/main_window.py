import time
import logging

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QScrollArea, QTabWidget,
    QGridLayout, QMessageBox, QApplication, QFileDialog,
    QMenuBar, QFrame, QTabBar, QSplitter,
)
from PyQt6.QtCore import Qt, QSettings, QTimer, QPoint, QSize
from PyQt6.QtGui import QAction, QKeySequence, QPainter, QColor, QFont

from utils.pipeline_state import PipelineState
from utils.error_handler import setup_logger

import gui.theme as _theme
from gui.styles import build as _build_ss, btn_style, tab_style, vertical_tab_style
from gui.widgets import OpStackList, ColorLogWidget, StatGrid
from gui.image_viewer import ImageViewer
from gui.filter_panel import FilterPanel
from gui.histogram_panel import HistogramPanel
from gui.noise_panel import NoisePanel
from gui.morphology_panel import MorphologyPanel
from gui.fourier_panel import FourierPanel
from gui.template_panel import TemplatePanel
from gui.cv_panel import CVPanel
from gui.ai_panel import AIPanel

_log = setup_logger()
_SETTINGS_ORG   = 'ciaw'
_SETTINGS_APP   = 'ciaw'
_SETTINGS_THEME = 'ciaw/theme'


# ── Medical cross icon ─────────────────────────────────────────────────────────

class _CrossIcon(QWidget):
    def __init__(self, size: int = 14, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, event):
        t = _theme.get()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(t['ACCENT']))
        s = self.width()
        b = max(3, s // 3)
        cx = (s - b) // 2
        p.drawRect(cx, 0, b, s)
        p.drawRect(0, cx, s, b)
        p.end()

    def theme_changed(self, _palette):
        self.update()


# ── Custom title bar ───────────────────────────────────────────────────────────

class _TitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TitleBar")
        self.setFixedHeight(38)
        self._drag_pos: QPoint | None = None
        self._session_start = time.time()
        self._build_ui()
        self._session_timer = QTimer(self)
        self._session_timer.setInterval(1000)
        self._session_timer.timeout.connect(self._tick_session)
        self._session_timer.start()
        self._apply_theme()

    def _build_ui(self):
        lyt = QHBoxLayout(self)
        lyt.setContentsMargins(10, 0, 4, 0)
        lyt.setSpacing(0)

        self._cross = _CrossIcon(14, self)
        lyt.addWidget(self._cross)
        lyt.addSpacing(7)

        self._ciaw_lbl = QLabel("CIAW")
        lyt.addWidget(self._ciaw_lbl)
        lyt.addSpacing(10)

        lyt.addWidget(self._vsep())
        lyt.addSpacing(10)

        self._sub_lbl = QLabel("Clinical Image Analysis Workbench")
        lyt.addWidget(self._sub_lbl)
        lyt.addStretch()

        self._session_lbl = QLabel("SESSION  00:00:00")
        lyt.addWidget(self._session_lbl)
        lyt.addSpacing(10)

        lyt.addWidget(self._vsep())
        lyt.addSpacing(8)

        self._theme_btn = QPushButton(_theme.names()[0])
        self._theme_btn.setFixedHeight(22)
        self._theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._theme_btn.clicked.connect(self._cycle_theme)
        lyt.addWidget(self._theme_btn)
        lyt.addSpacing(8)

        lyt.addWidget(self._vsep())

        self._min_btn   = self._ctrl_btn("─")
        self._max_btn   = self._ctrl_btn("□")
        self._close_btn = self._ctrl_btn("✕")
        self._close_btn.setObjectName("CloseBtn")
        self._min_btn.clicked.connect(lambda: self.window().showMinimized())
        self._max_btn.clicked.connect(self._toggle_max)
        self._close_btn.clicked.connect(self.window().close)
        lyt.addWidget(self._min_btn)
        lyt.addWidget(self._max_btn)
        lyt.addWidget(self._close_btn)

    def _vsep(self) -> QFrame:
        f = QFrame(self)
        f.setFrameShape(QFrame.Shape.VLine)
        f.setFixedHeight(16)
        f.setFixedWidth(1)
        return f

    def _ctrl_btn(self, text: str) -> QPushButton:
        btn = QPushButton(text, self)
        btn.setFixedSize(38, 38)
        btn.setFlat(True)
        return btn

    def _toggle_max(self):
        w = self.window()
        w.showNormal() if w.isMaximized() else w.showMaximized()

    def _tick_session(self):
        elapsed = int(time.time() - self._session_start)
        h, rem = divmod(elapsed, 3600)
        m, s   = divmod(rem, 60)
        self._session_lbl.setText(f"SESSION  {h:02d}:{m:02d}:{s:02d}")

    def _cycle_theme(self):
        names = _theme.names()
        try:
            idx = names.index(_theme._current)
        except (ValueError, AttributeError):
            idx = 0
        mw = self.window()
        if hasattr(mw, '_apply_theme'):
            mw._apply_theme(names[(idx + 1) % len(names)])

    def _apply_theme(self):
        t = _theme.get()
        self.setStyleSheet(
            f"#TitleBar {{ background: {t['PANEL']}; "
            f"border-bottom: 1px solid {t['BORDER']}; }}"
            f"#TitleBar QFrame {{ background: {t['BORDER2']}; }}"
        )
        self._ciaw_lbl.setStyleSheet(
            f"color: {t['ACCENT']}; font-size: 13px; font-weight: bold; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; background: transparent;"
        )
        self._sub_lbl.setStyleSheet(
            f"color: {t['MUTED']}; font-size: 10px; background: transparent; "
            f"font-family: 'JetBrains Mono', Consolas, monospace;"
        )
        self._session_lbl.setStyleSheet(
            f"color: {t['MUTED']}; font-size: 9px; background: transparent; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; letter-spacing: 1px;"
        )
        ctrl_ss = (
            f"QPushButton {{ background: transparent; color: {t['MUTED']}; border: none; "
            f"font-size: 12px; }}"
            f"QPushButton:hover {{ color: {t['TEXT']}; background: {t['PANEL2']}; }}"
        )
        self._min_btn.setStyleSheet(ctrl_ss)
        self._max_btn.setStyleSheet(ctrl_ss)
        self._close_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {t['MUTED']}; border: none; "
            f"font-size: 12px; }}"
            f"QPushButton:hover {{ color: {t['WHITE']}; background: {t['RED']}; }}"
        )
        self._theme_btn.setStyleSheet(
            f"QPushButton {{ background: {t['INPUT']}; color: {t['ACCENT']}; "
            f"border: 1px solid {t['BORDER2']}; border-radius: 4px; "
            f"font-size: 9px; padding: 2px 8px; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; }}"
            f"QPushButton:hover {{ background: {t['PANEL2']}; }}"
        )
        self._cross.update()

    def theme_changed(self, palette: dict):
        self._apply_theme()
        try:
            self._theme_btn.setText(_theme._current)
        except AttributeError:
            pass

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (event.globalPosition().toPoint()
                              - self.window().frameGeometry().topLeft())
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self.window().move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_max()


# ── Vertical tab bar (non-rotated text) ───────────────────────────────────────

class _WestTabBar(QTabBar):
    """Left-side tab bar with readable horizontal text."""

    def tabSizeHint(self, index: int) -> QSize:
        return QSize(80, 28)

    def minimumTabSizeHint(self, index: int) -> QSize:
        return QSize(60, 22)

    def paintEvent(self, event):
        t = _theme.get()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for i in range(self.count()):
            r = self.tabRect(i)
            if self.currentIndex() == i:
                p.fillRect(r, QColor(t['INPUT']))
                p.fillRect(r.x(), r.y(), 2, r.height(), QColor(t['ACCENT']))
                text_color = QColor(t['TEXT'])
            else:
                p.fillRect(r, QColor(t['PANEL']))
                text_color = QColor(t['MUTED'])
            p.setPen(text_color)
            p.setFont(QFont("JetBrains Mono", 8))
            p.drawText(
                r.adjusted(10, 0, 0, 0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                self.tabText(i),
            )
        p.end()


# ── MainWindow ─────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        self._state           = PipelineState()
        self._redo_stack:  list  = []
        self._last_op_ms:  float = 0.0
        self._op_start:    float = 0.0
        self._active_workers: int = 0
        self._spinner_idx:    int = 0
        self._theme_name:     str = 'Dark Lime'
        self._save_path: str | None = None
        self._roi:       tuple | None = None
        self._last_metadata: dict = {}
        self._last_filepath: str  = ''

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
        t = _theme.get()
        for panel in self._all_panels + [self._image_viewer]:
            if hasattr(panel, 'theme_changed'):
                panel.theme_changed(t)
        if hasattr(self, '_title_bar'):
            self._title_bar.theme_changed(t)
        if hasattr(self, '_opstack_list'):
            self._opstack_list.theme_changed(t)
        if hasattr(self, 'log_widget') and hasattr(self.log_widget, 'theme_changed'):
            self.log_widget.theme_changed(t)
        if hasattr(self, '_right_tabs'):
            self._right_tabs.setStyleSheet(tab_style())
            self._right_tabs.update()
        if hasattr(self, '_ops_tabs'):
            self._ops_tabs.setStyleSheet(vertical_tab_style())
            self._ops_tabs.tabBar().update()
            self._ops_tabs.update()
        self._update_pipeline_btn_style()
        QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(_SETTINGS_THEME, name)
        self._update_status_bar()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.setWindowTitle("Clinical Image Analysis Workbench")
        self.resize(1400, 900)

        # Hide native menu bar; we embed a QMenuBar in the central widget
        self.menuBar().hide()

        central = QWidget()
        central.setObjectName("CentralRoot")
        root_lyt = QVBoxLayout(central)
        root_lyt.setContentsMargins(1, 1, 1, 1)
        root_lyt.setSpacing(0)

        self._title_bar = _TitleBar(self)
        root_lyt.addWidget(self._title_bar)

        self._menu_bar = self._build_menu()
        root_lyt.addWidget(self._menu_bar)

        content = QWidget()
        content_lyt = QHBoxLayout(content)
        content_lyt.setContentsMargins(0, 0, 0, 0)
        content_lyt.setSpacing(0)

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setChildrenCollapsible(False)

        self._left_panel = self._build_left_panel()
        self._main_splitter.addWidget(self._left_panel)

        center = QWidget()
        center_lyt = QVBoxLayout(center)
        center_lyt.setContentsMargins(0, 0, 0, 0)
        center_lyt.setSpacing(0)
        self._image_viewer = ImageViewer()
        center_lyt.addWidget(self._image_viewer, stretch=1)
        self._main_splitter.addWidget(center)

        right_panel = self._build_right_panel()
        self._main_splitter.addWidget(right_panel)

        self._main_splitter.setSizes([240, 800, 300])
        self._main_splitter.setStretchFactor(0, 0)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setStretchFactor(2, 0)

        content_lyt.addWidget(self._main_splitter)
        root_lyt.addWidget(content, stretch=1)
        self.setCentralWidget(central)
        self._build_status_bar()

    # ---- Left panel ----

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        lyt = QVBoxLayout(w)
        lyt.setContentsMargins(8, 8, 8, 8)
        lyt.setSpacing(6)

        lyt.addWidget(self._build_metadata_box())
        lyt.addWidget(self._build_checkpoint_box())
        lyt.addWidget(self._build_pipeline_box())
        self._opstack_box = self._build_opstack_box()
        lyt.addWidget(self._opstack_box, stretch=1)

        undo_row = QHBoxLayout()
        undo_row.setSpacing(0)
        self._undo_btn  = QPushButton("↩ UNDO")
        self._redo_btn  = QPushButton("↪ REDO")
        self._reset_btn = QPushButton("✕ RESET")
        for btn in (self._undo_btn, self._redo_btn, self._reset_btn):
            btn.setFixedHeight(26)
        self._undo_btn.setStyleSheet(btn_style('ghost'))
        self._redo_btn.setStyleSheet(btn_style('ghost'))
        self._reset_btn.setStyleSheet(btn_style('danger'))
        undo_row.addWidget(self._undo_btn)
        undo_row.addWidget(self._redo_btn)
        undo_row.addWidget(self._reset_btn)
        lyt.addLayout(undo_row)

        return w

    def _build_metadata_box(self) -> QGroupBox:
        box = QGroupBox("IMAGE INFO")
        form = QFormLayout(box)
        form.setSpacing(2)
        form.setContentsMargins(6, 4, 6, 4)

        t = _theme.get()
        key_ss = (f"color: {t['MUTED']}; font-family: 'JetBrains Mono', Consolas, monospace;"
                  f" font-size: 9px;")
        val_ss = (f"color: {t['TEXT']}; font-family: 'JetBrains Mono', Consolas, monospace;"
                  f" font-size: 9px;")

        def _kl(text):
            l = QLabel(text); l.setStyleSheet(key_ss); l.setFixedWidth(60); return l

        def _vl():
            l = QLabel("—"); l.setStyleSheet(val_ss)
            l.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            l.setMaximumWidth(130); return l

        self._meta_file     = _vl()
        self._meta_dims     = _vl()
        self._meta_ch       = _vl()
        self._meta_dtype    = _vl()
        self._meta_range    = _vl()
        self._meta_size     = _vl()
        self._meta_modality = _vl()
        self._meta_patient  = _vl()
        self._meta_age      = _vl()
        self._meta_body     = _vl()
        self._meta_kvp      = _vl()

        self._dicom_rows: list[tuple] = []
        for key, val in [
            ("File",  self._meta_file), ("Dims",  self._meta_dims),
            ("Ch",    self._meta_ch),   ("DType", self._meta_dtype),
            ("Range", self._meta_range),("Size",  self._meta_size),
        ]:
            form.addRow(_kl(key), val)

        for key, val in [
            ("Modality", self._meta_modality), ("Patient", self._meta_patient),
            ("Age",      self._meta_age),       ("Body",    self._meta_body),
            ("KVP",      self._meta_kvp),
        ]:
            kl = _kl(key)
            form.addRow(kl, val)
            kl.hide(); val.hide()
            self._dicom_rows.append((kl, val))

        return box

    def _build_checkpoint_box(self) -> QGroupBox:
        box = QGroupBox("CHECKPOINTS")
        grid = QGridLayout(box)
        grid.setSpacing(3)
        grid.setContentsMargins(6, 4, 6, 4)

        t = _theme.get()
        slot_ss = (f"color: {t['ACCENT']}; font-family: 'JetBrains Mono', Consolas, monospace;"
                   f" font-weight: bold; font-size: 10px;")

        self._cp_save:    dict[str, QPushButton] = {}
        self._cp_restore: dict[str, QPushButton] = {}
        self._cp_dot:     dict[str, QLabel]      = {}

        for row, slot in enumerate(('A', 'B', 'C', 'D')):
            lbl = QLabel(slot)
            lbl.setFixedWidth(16)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(slot_ss)

            dot = QLabel("○")
            dot.setFixedWidth(14)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dot.setStyleSheet(f"color: {t['MUTED2']}; font-size: 10px;")
            self._cp_dot[slot] = dot

            save_btn    = QPushButton("SAVE")
            restore_btn = QPushButton("RESTORE")
            for btn in (save_btn, restore_btn):
                btn.setStyleSheet(btn_style('ghost'))
                btn.setFixedHeight(22)
            self._cp_save[slot]    = save_btn
            self._cp_restore[slot] = restore_btn

            grid.addWidget(lbl,         row, 0)
            grid.addWidget(dot,         row, 1)
            grid.addWidget(save_btn,    row, 2)
            grid.addWidget(restore_btn, row, 3)

        return box

    def _build_pipeline_box(self) -> QGroupBox:
        box = QGroupBox("PIPELINE")
        lyt = QVBoxLayout(box)
        lyt.setContentsMargins(6, 4, 6, 4)
        lyt.setSpacing(4)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(0)
        self._cum_btn = QPushButton("CUMULATIVE")
        self._ind_btn = QPushButton("INDEPENDENT")
        for btn in (self._cum_btn, self._ind_btn):
            btn.setCheckable(True)
            btn.setFixedHeight(26)
        self._cum_btn.setChecked(True)
        self._cum_btn.clicked.connect(lambda: self._set_pipeline_mode('cumulative'))
        self._ind_btn.clicked.connect(lambda: self._set_pipeline_mode('independent'))
        mode_row.addWidget(self._cum_btn)
        mode_row.addWidget(self._ind_btn)
        lyt.addLayout(mode_row)
        self._update_pipeline_btn_style()
        return box

    def _build_opstack_box(self) -> QGroupBox:
        box = QGroupBox("OP STACK (0)")
        lyt = QVBoxLayout(box)
        lyt.setContentsMargins(4, 4, 4, 4)
        lyt.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._opstack_list = OpStackList()
        scroll.setWidget(self._opstack_list)
        lyt.addWidget(scroll)
        return box

    # ---- Right panel ----

    def _build_right_panel(self) -> QTabWidget:
        self._right_tabs = QTabWidget()
        self._right_tabs.setDocumentMode(True)
        self._right_tabs.addTab(self._build_ops_tab(), "OPERATIONS")
        self._right_tabs.addTab(self._build_log_tab(), "LOG")
        self._right_tabs.addTab(self._build_roi_tab(), "ROI VIEW")
        return self._right_tabs

    def _build_ops_tab(self) -> QWidget:
        self._filter_panel     = FilterPanel()
        self._histogram_panel  = HistogramPanel()
        self._noise_panel      = NoisePanel()
        self._morphology_panel = MorphologyPanel()
        self._fourier_panel    = FourierPanel()
        self._template_panel   = TemplatePanel()
        self._cv_panel         = CVPanel()
        self._ai_panel         = AIPanel()

        self._all_panels = [
            self._filter_panel, self._histogram_panel, self._noise_panel,
            self._morphology_panel, self._fourier_panel, self._template_panel,
            self._cv_panel, self._ai_panel,
        ]

        west_bar = _WestTabBar()
        self._ops_tabs = QTabWidget()
        self._ops_tabs.setTabBar(west_bar)
        self._ops_tabs.setTabPosition(QTabWidget.TabPosition.West)
        self._ops_tabs.setDocumentMode(True)
        self._ops_tabs.setUsesScrollButtons(False)

        for label, panel in [
            ("FILTER",     self._filter_panel),
            ("HISTOGRAM",  self._histogram_panel),
            ("NOISE",      self._noise_panel),
            ("MORPHOLOGY", self._morphology_panel),
            ("FREQUENCY",  self._fourier_panel),
            ("TEMPLATE",   self._template_panel),
            ("CV",         self._cv_panel),
            ("AI",         self._ai_panel),
        ]:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setWidget(panel)
            self._ops_tabs.addTab(scroll, label)

        for panel in self._all_panels:
            panel.setEnabled(False)
            panel.set_state(self._state)

        return self._ops_tabs

    def _build_log_tab(self) -> QWidget:
        t = _theme.get()
        w = QWidget()
        lyt = QVBoxLayout(w)
        lyt.setContentsMargins(6, 6, 6, 6)
        lyt.setSpacing(4)

        top_row = QHBoxLayout()
        hdr = QLabel("APPLICATION LOG")
        hdr.setStyleSheet(
            f"color: {t['ACCENT']}; font-family: 'JetBrains Mono', Consolas, monospace;"
            f" font-size: 9px; font-weight: bold; letter-spacing: 2px;"
        )
        top_row.addWidget(hdr)
        top_row.addStretch()

        export_btn = QPushButton("EXPORT")
        export_btn.setFixedHeight(22)
        export_btn.setStyleSheet(btn_style('ghost'))
        export_btn.clicked.connect(self._export_log)
        top_row.addWidget(export_btn)

        clear_btn = QPushButton("CLEAR")
        clear_btn.setFixedHeight(22)
        clear_btn.setStyleSheet(btn_style('ghost'))
        top_row.addWidget(clear_btn)
        lyt.addLayout(top_row)

        self.log_widget = ColorLogWidget()
        lyt.addWidget(self.log_widget, stretch=1)
        clear_btn.clicked.connect(self.log_widget.clear)
        return w

    def _build_roi_tab(self) -> QWidget:
        t = _theme.get()
        w = QWidget()
        lyt = QVBoxLayout(w)
        lyt.setContentsMargins(8, 8, 8, 8)
        lyt.setSpacing(6)

        self._roi_img_lbl = QLabel("Draw an ROI on the image")
        self._roi_img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._roi_img_lbl.setStyleSheet(
            f"QLabel {{ color: {t['MUTED']}; font-size: 9px; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; "
            f"border: 1px solid {t['BORDER']}; background: {t['BG']}; }}"
        )
        self._roi_img_lbl.setMinimumHeight(200)
        lyt.addWidget(self._roi_img_lbl, stretch=1)

        self._roi_stats_grid = StatGrid(["W", "H", "Mean", "Std", "Min", "Max"], cols=3)
        self._roi_stats_grid.setFixedHeight(80)
        self._roi_stats_grid.hide()
        lyt.addWidget(self._roi_stats_grid)
        return w

    # ---- Menu (embedded) ----

    def _build_menu(self) -> QMenuBar:
        mb = QMenuBar(self)
        mb.setNativeMenuBar(False)

        file_menu = mb.addMenu("File")
        self._act(file_menu, "Open",    "Ctrl+O", self._open_file)
        self._act(file_menu, "Save",    "Ctrl+S", self._save_file)
        self._act(file_menu, "Save As", None,     self._save_file_as)

        edit_menu = mb.addMenu("Edit")
        self._act(edit_menu, "Undo", "Ctrl+Z", self._do_undo)
        self._act(edit_menu, "Redo", "Ctrl+Y", self._do_redo)

        view_menu = mb.addMenu("View")
        theme_menu = view_menu.addMenu("Theme")
        for name in _theme.names():
            act = QAction(name, self)
            act.triggered.connect(lambda _, n=name: self._apply_theme(n))
            theme_menu.addAction(act)

        help_menu = mb.addMenu("Help")
        self._act(help_menu, "About", None, self._show_about)
        return mb

    def _act(self, menu, label: str, shortcut, slot):
        act = QAction(label, self)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        if slot:
            act.triggered.connect(slot)
        menu.addAction(act)
        return act

    # ---- Status bar ----

    def _build_status_bar(self):
        t = _theme.get()
        seg_ss = (
            f"QLabel {{ color: {t['MUTED']}; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; "
            f"font-size: 9px; padding: 0 5px; }}"
        )
        sep_ss = f"QLabel {{ color: {t['BORDER2']}; font-size: 9px; padding: 0 1px; }}"

        def seg(text):
            l = QLabel(text); l.setStyleSheet(seg_ss); return l

        def sep():
            l = QLabel("|"); l.setStyleSheet(sep_ss); return l

        self._sb_op     = seg("OP: —")
        self._sb_zoom   = seg("ZOOM: 100%")
        self._sb_interp = seg("NN")
        self._sb_pipe   = seg("CUM")
        self._sb_mem    = seg("0 MB")
        self._sb_worker = seg("  ")

        sb = self.statusBar()
        for widget in [
            self._sb_op,    sep(), self._sb_zoom,  sep(),
            self._sb_interp, sep(), self._sb_pipe, sep(),
            self._sb_mem,   sep(), self._sb_worker,
        ]:
            sb.addPermanentWidget(widget)

    def _update_status_bar(self, *_args):
        img    = self._state.current()
        mode   = self._state.get_mode()
        zoom   = int(getattr(self._image_viewer, '_zoom', 1.0) * 100)
        interp = getattr(self._image_viewer, '_interp', 'nearest')
        op     = str(self._state.current_op() or '—')
        if len(op) > 20:
            op = op[:19] + '…'
        try:
            import psutil, os
            mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        except Exception:
            mem_mb = img.nbytes / 1024 / 1024 if img is not None else 0.0

        self._sb_op.setText(f"OP: {op}")
        self._sb_zoom.setText(f"ZOOM: {zoom}%")
        self._sb_interp.setText("BL" if interp == 'bilinear' else "NN")
        self._sb_pipe.setText("CUM" if mode == 'cumulative' else "IND")
        self._sb_mem.setText(f"{mem_mb:.0f} MB")

    # ------------------------------------------------------------------
    # Spinner
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
        self._image_viewer.roi_selected.connect(self._on_roi_selected)
        self._image_viewer.roi_cleared.connect(self._on_roi_cleared)
        self._image_viewer.interp_changed.connect(
            lambda m: self._sb_interp.setText("BL" if m == 'bilinear' else "NN")
        )
        self._image_viewer.set_state(self._state)

        self._filter_panel.filter_applied.connect(self._on_operation_done)
        self._histogram_panel.operation_applied.connect(self._on_operation_done)
        self._noise_panel.noise_applied.connect(self._on_operation_done)
        self._morphology_panel.morphology_applied.connect(self._on_operation_done)
        self._fourier_panel.fourier_applied.connect(self._on_operation_done)
        self._template_panel.template_applied.connect(self._on_operation_done)
        self._cv_panel.cv_applied.connect(self._on_operation_done)
        self._ai_panel.ai_applied.connect(self._on_operation_done)

        for panel in self._all_panels:
            panel.error_occurred.connect(self._on_error)

        self._ai_panel.classification_done.connect(self._on_classification_done)

        for slot in ('A', 'B', 'C', 'D'):
            self._cp_save[slot].clicked.connect(
                lambda _, s=slot: self._do_save_checkpoint(s)
            )
            self._cp_restore[slot].clicked.connect(
                lambda _, s=slot: self._do_restore_checkpoint(s)
            )

        self._undo_btn.clicked.connect(self._do_undo)
        self._redo_btn.clicked.connect(self._do_redo)
        self._reset_btn.clicked.connect(self._do_reset)

    # ------------------------------------------------------------------
    # Pipeline mode
    # ------------------------------------------------------------------

    def _update_pipeline_btn_style(self):
        t = _theme.get()
        mode = self._state.get_mode()
        active_ss = (
            f"QPushButton {{ background: {t['ACCENT']}; color: {t['DARK']}; border: none; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; font-size: 9px; "
            f"font-weight: bold; }}"
            f"QPushButton:hover {{ background: {t['ACCENT2']}; }}"
        )
        inactive_ss = (
            f"QPushButton {{ background: {t['INPUT']}; color: {t['MUTED']}; "
            f"border: 1px solid {t['BORDER2']}; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; font-size: 9px; }}"
            f"QPushButton:hover {{ color: {t['TEXT']}; }}"
        )
        if mode == 'cumulative':
            self._cum_btn.setStyleSheet(active_ss)
            self._ind_btn.setStyleSheet(inactive_ss)
        else:
            self._cum_btn.setStyleSheet(inactive_ss)
            self._ind_btn.setStyleSheet(active_ss)

    def _set_pipeline_mode(self, mode: str):
        self._state.set_mode(mode)
        self._update_pipeline_btn_style()

    # ------------------------------------------------------------------
    # Op stack
    # ------------------------------------------------------------------

    def _refresh_opstack(self, flash_new: bool = False):
        names = self._state.get_stack_names()
        self._opstack_list.clear()
        for i, name in enumerate(reversed(names), 1):
            self._opstack_list.addItem(f"{i:02d}  {name}")
        self._opstack_box.setTitle(f"OP STACK ({len(names)})")
        if flash_new and self._opstack_list.count() > 0:
            self._opstack_list.flash_top()

    # ------------------------------------------------------------------
    # Metadata panel
    # ------------------------------------------------------------------

    def _update_metadata_display(self, image: np.ndarray,
                                  metadata: dict | None = None,
                                  filepath: str = ''):
        if image is None:
            return
        h, w = image.shape[:2]
        ch = image.shape[2] if image.ndim == 3 else 1
        dt = image.dtype
        bit_depth = '8-bit' if dt == np.uint8 else ('16-bit' if dt == np.uint16 else str(dt))
        kb = image.nbytes / 1024

        fname = filepath.replace('\\', '/').split('/')[-1] if filepath else ''
        self._meta_file.setText(fname or '—')
        self._meta_file.setToolTip(filepath)
        self._meta_dims.setText(f"{w} × {h}")
        self._meta_ch.setText(str(ch))
        self._meta_dtype.setText(bit_depth)
        self._meta_range.setText(f"{int(image.min())} – {int(image.max())}")
        self._meta_size.setText(
            f"{kb:.0f} KB" if kb < 1024 else f"{kb / 1024:.1f} MB"
        )

        if metadata:
            self._meta_modality.setText(str(metadata.get('Modality', '—')))
            self._meta_patient.setText(str(metadata.get('PatientName', '—')))
            self._meta_age.setText(str(metadata.get('PatientAge', '—')))
            self._meta_body.setText(str(metadata.get('BodyPartExamined', '—')))
            self._meta_kvp.setText(str(metadata.get('KVP', '—')))
            for lbl, val in self._dicom_rows:
                lbl.show(); val.show()
        else:
            for lbl, val in self._dicom_rows:
                lbl.hide(); val.hide()

    def _update_image_stats(self, image: np.ndarray):
        """Update only image-derived fields — preserves File/DICOM fields."""
        if image is None:
            return
        h, w = image.shape[:2]
        ch = image.shape[2] if image.ndim == 3 else 1
        dt = image.dtype
        bit_depth = ('8-bit' if dt == np.uint8 else
                     '16-bit' if dt == np.uint16 else str(dt))
        kb = image.nbytes / 1024
        self._meta_dims.setText(f"{w} × {h}")
        self._meta_ch.setText(str(ch))
        self._meta_dtype.setText(bit_depth)
        self._meta_range.setText(f"{int(image.min())} – {int(image.max())}")
        self._meta_size.setText(f"{kb:.0f} KB" if kb < 1024 else f"{kb / 1024:.1f} MB")

    # ------------------------------------------------------------------
    # Core operation lifecycle
    # ------------------------------------------------------------------

    def _on_operation_done(self, op_name: str, result):
        if result is None:
            return
        if self._op_start:
            self._last_op_ms = (time.monotonic() - self._op_start) * 1000
            self._op_start = 0.0
        self._active_workers = max(0, self._active_workers - 1)
        self._redo_stack.clear()
        before = self._state.current()
        if before is not None:
            self._image_viewer.store_before(before)
        self._state.push(op_name, result)
        self._image_viewer.set_image(result)
        self._refresh_opstack(flash_new=True)
        self._update_image_stats(result)
        try:
            if (self._right_tabs.currentIndex() == 0
                    and self._ops_tabs.currentIndex() == 1):
                self._histogram_panel.update_histogram(result)
        except Exception:
            pass
        try:
            if (self._right_tabs.currentIndex() == 0
                    and self._ops_tabs.currentIndex() == 4):
                self._fourier_panel.update_display(result)
        except Exception:
            pass
        self._update_status_bar()

    def _on_error(self, msg: str):
        _log.error("Worker error: %s", msg)
        self._active_workers = max(0, self._active_workers - 1)
        QMessageBox.warning(self, "Operation Error", msg)

    def _on_classification_done(self, results: list):
        if results:
            top = max(results, key=lambda d: d.get('prob', 0))
            self._sb_op.setText(f"AI: {top.get('label','?')} {top.get('prob',0)*100:.1f}%")

    # ------------------------------------------------------------------
    # ROI
    # ------------------------------------------------------------------

    def _on_roi_selected(self, x: int, y: int, w: int, h: int):
        self._roi = (x, y, w, h)
        image = self._state.current()
        if image is not None:
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = max(1, min(w, image.shape[1] - x))
            h = max(1, min(h, image.shape[0] - y))
            roi_crop = image[y:y + h, x:x + w]
            from utils.image_utils import to_qpixmap
            pix = to_qpixmap(roi_crop).scaled(
                280, 380,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._roi_img_lbl.setPixmap(pix)
            for key, val in [
                ("W", str(w)), ("H", str(h)),
                ("Mean", f"{roi_crop.mean():.1f}"), ("Std",  f"{roi_crop.std():.1f}"),
                ("Min",  str(int(roi_crop.min()))), ("Max",  str(int(roi_crop.max()))),
            ]:
                self._roi_stats_grid.set_value(key, val)
            self._roi_stats_grid.show()
            self._right_tabs.setCurrentIndex(2)

        for panel in (self._noise_panel, self._cv_panel,
                      self._template_panel, self._ai_panel,
                      self._histogram_panel):
            try:
                panel.set_roi(x, y, w, h)
            except Exception as e:
                _log.error("set_roi: %s", e)

    def _on_roi_cleared(self):
        self._roi = None
        for panel in (self._noise_panel, self._cv_panel,
                      self._histogram_panel, self._ai_panel):
            try:
                panel.clear_roi()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Undo / Redo / Reset
    # ------------------------------------------------------------------

    def _do_undo(self):
        if not self._state.has_image():
            return
        before = self._state.current()
        if self._state.get_stack_names():
            self._redo_stack.append(
                (self._state.current_op(), self._state.current().copy())
            )
        result = self._state.undo()
        if result is not None:
            if before is not None:
                self._image_viewer.store_before(before)
            self._image_viewer.set_image(result)
            try:
                self._histogram_panel.update_histogram(result)
            except Exception:
                pass
        self._refresh_opstack()
        self._update_status_bar()

    def _do_redo(self):
        if not self._redo_stack:
            return
        before = self._state.current()
        op_name, img = self._redo_stack.pop()
        if before is not None:
            self._image_viewer.store_before(before)
        self._state.push(op_name, img)
        self._image_viewer.set_image(img)
        self._refresh_opstack()
        self._update_status_bar()

    def _do_reset(self):
        before = self._state.current()
        self._state.reset()
        self._redo_stack.clear()
        img = self._state.current()
        if img is not None:
            if before is not None:
                self._image_viewer.store_before(before)
            self._image_viewer.set_image(img)
        self._refresh_opstack()
        self._update_status_bar()

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def _do_save_checkpoint(self, slot: str):
        if not self._state.has_image():
            return
        try:
            self._state.save_checkpoint(slot)
            t = _theme.get()
            self._cp_dot[slot].setText("●")
            self._cp_dot[slot].setStyleSheet(f"color: {t['ACCENT']}; font-size: 10px;")
        except Exception as exc:
            _log.warning("Checkpoint save failed: %s", exc)

    def _do_restore_checkpoint(self, slot: str):
        before = self._state.current()
        img = self._state.restore_checkpoint(slot)
        if img is not None:
            if before is not None:
                self._image_viewer.store_before(before)
            self._image_viewer.set_image(img)
            self._refresh_opstack()
            self._update_status_bar()

    # ------------------------------------------------------------------
    # File I/O
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
        image, metadata = result
        self._last_metadata = metadata or {}
        self._last_filepath  = path

        self._state.set_original(image)
        self._redo_stack.clear()
        self._on_roi_cleared()
        if hasattr(self._image_viewer, 'clear_roi'):
            self._image_viewer.clear_roi(emit_signal=False)

        self._image_viewer.set_image(image)
        self._image_viewer.store_before(image)
        self._noise_panel.set_reference_image(image)
        self._noise_panel.clear_roi()

        for panel in self._all_panels:
            panel.setEnabled(True)
            panel.set_state(self._state)

        self._image_viewer.set_state(self._state)
        self._update_metadata_display(image, metadata=self._last_metadata, filepath=path)

        try:
            self._histogram_panel.update_histogram(image)
        except Exception:
            pass
        try:
            self._fourier_panel.update_display(image)
        except Exception:
            pass

        self._roi_img_lbl.setText("Draw an ROI on the image")
        self._roi_stats_grid.hide()
        self._refresh_opstack()
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

    def _export_log(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Log", "ciaw_log.txt", "Text (*.txt);;All Files (*)"
        )
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(self.log_widget.toPlainText())
            except Exception as exc:
                _log.error("Log export failed: %s", exc)

    # ------------------------------------------------------------------
    # About
    # ------------------------------------------------------------------

    def _show_about(self):
        QMessageBox.about(
            self, "About CIAW",
            "Clinical Image Analysis Workbench\n"
            "Digital Image Processing Project\n\n"
            "Spatial · Frequency · Morphology · Noise\n"
            "Deep Learning · Histogram · Segmentation",
        )
