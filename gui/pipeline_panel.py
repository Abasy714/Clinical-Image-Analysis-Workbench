from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence

from gui.theme import get as _get_theme
from gui.styles import btn_style


class PipelinePanel(QWidget):
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    mode_changed = pyqtSignal(str)
    checkpoint_save_requested = pyqtSignal(str)
    checkpoint_restore_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = 'cumulative'
        self._build_ui()

    def _build_ui(self):
        p = _get_theme()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Section header
        header = QLabel("PIPELINE")
        header.setStyleSheet(
            f"color: {p['ACCENT']}; font-family: 'JetBrains Mono', Consolas, monospace; "
            f"font-size: 9px; font-weight: bold; letter-spacing: 2px; "
            f"padding-bottom: 4px; border-bottom: 1px solid {p['BORDER']};"
        )
        layout.addWidget(header)

        # Mode toggle button
        self._mode_btn = QPushButton("MODE: CUMULATIVE")
        self._mode_btn.setCheckable(True)
        self._mode_btn.setChecked(True)
        self._mode_btn.clicked.connect(self._on_mode_toggle)
        self._mode_btn.setStyleSheet(self._mode_btn_style(True))
        layout.addWidget(self._mode_btn)

        # Op stack list (most recent at top)
        self._list = QListWidget()
        self._list.setStyleSheet(
            f"QListWidget {{ background: {p['BG']}; border: 1px solid {p['BORDER']}; "
            f"color: {p['TEXT']}; font-family: 'JetBrains Mono', Consolas, monospace; "
            f"font-size: 10px; outline: none; }}"
            f"QListWidget::item {{ padding: 4px 8px; border-bottom: 1px solid {p['BORDER']}; }}"
            f"QListWidget::item:selected {{ background: {p['PANEL2']}; color: {p['ACCENT']}; }}"
        )
        layout.addWidget(self._list, stretch=1)

        # Undo / Redo / Reset row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self._undo_btn = QPushButton("UNDO")
        self._undo_btn.setStyleSheet(btn_style('default'))
        self._undo_btn.clicked.connect(self.undo_requested)

        self._redo_btn = QPushButton("REDO")
        self._redo_btn.setStyleSheet(btn_style('default'))
        self._redo_btn.clicked.connect(self.redo_requested)

        self._reset_btn = QPushButton("RESET")
        self._reset_btn.setStyleSheet(btn_style('danger'))
        self._reset_btn.clicked.connect(self.reset_requested)

        btn_row.addWidget(self._undo_btn)
        btn_row.addWidget(self._redo_btn)
        btn_row.addWidget(self._reset_btn)
        layout.addLayout(btn_row)

        # Checkpoints section
        ck_header = QLabel("CHECKPOINTS")
        ck_header.setStyleSheet(
            f"color: {p['MUTED']}; font-family: 'JetBrains Mono', Consolas, monospace; "
            f"font-size: 9px; font-weight: bold; letter-spacing: 2px; "
            f"margin-top: 4px; padding-top: 4px; border-top: 1px solid {p['BORDER']};"
        )
        layout.addWidget(ck_header)

        for slot in ('A', 'B', 'C', 'D'):
            row = QHBoxLayout()
            row.setSpacing(4)

            lbl = QLabel(slot)
            lbl.setFixedWidth(16)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                f"color: {p['ACCENT']}; font-family: 'JetBrains Mono', Consolas, monospace; "
                f"font-weight: bold; font-size: 10px;"
            )

            save_btn = QPushButton("SAVE")
            save_btn.setStyleSheet(btn_style('ghost'))
            save_btn.clicked.connect(lambda _, s=slot: self.checkpoint_save_requested.emit(s))

            restore_btn = QPushButton("RESTORE")
            restore_btn.setStyleSheet(btn_style('ghost'))
            restore_btn.clicked.connect(lambda _, s=slot: self.checkpoint_restore_requested.emit(s))

            row.addWidget(lbl)
            row.addWidget(save_btn)
            row.addWidget(restore_btn)
            layout.addLayout(row)

    # ------------------------------------------------------------------
    # Styling helpers
    # ------------------------------------------------------------------

    def _mode_btn_style(self, on: bool) -> str:
        p = _get_theme()
        if on:
            return (
                f"QPushButton {{ background: {p['PANEL2']}; color: {p['ACCENT']}; "
                f"border: 1px solid {p['ACCENT_DIM']}; border-radius: 2px; "
                f"font-family: 'JetBrains Mono', Consolas, monospace; "
                f"font-size: 9px; font-weight: bold; letter-spacing: 1px; padding: 5px; }}"
                f"QPushButton:hover {{ border-color: {p['ACCENT']}; }}"
            )
        return (
            f"QPushButton {{ background: {p['PANEL2']}; color: {p['AMBER']}; "
            f"border: 1px solid {p['AMBER']}; border-radius: 2px; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; "
            f"font-size: 9px; font-weight: bold; letter-spacing: 1px; padding: 5px; }}"
            f"QPushButton:hover {{ color: {p['TEXT']}; }}"
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_mode_toggle(self, checked: bool):
        self._mode = 'cumulative' if checked else 'independent'
        self._mode_btn.setText(
            "MODE: CUMULATIVE" if checked else "MODE: INDEPENDENT"
        )
        self._mode_btn.setStyleSheet(self._mode_btn_style(checked))
        self.mode_changed.emit(self._mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self, stack_names: list, mode: str):
        self._list.clear()
        total = len(stack_names)
        for i, name in enumerate(reversed(stack_names)):
            item = QListWidgetItem(f"{total - i:02d}  {name}")
            self._list.addItem(item)
        on = (mode == 'cumulative')
        self._mode = mode
        self._mode_btn.setChecked(on)
        self._mode_btn.setText("MODE: CUMULATIVE" if on else "MODE: INDEPENDENT")
        self._mode_btn.setStyleSheet(self._mode_btn_style(on))
