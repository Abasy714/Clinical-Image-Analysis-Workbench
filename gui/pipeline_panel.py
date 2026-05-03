# STATUS: IMPLEMENTED
"""
Pipeline management panel for sequential image enhancement operations.
Provides undo last step and reset to original buttons, and displays the current op stack.
"""

# PyQt6.QtWidgets — QWidget, QVBoxLayout, QListWidget, QPushButton, QLabel
# PyQt6.QtCore — pyqtSignal
# utils.pipeline_state — PipelineState

# FUNCTIONS / CLASSES
# class PipelinePanel(QWidget):
#   def __init__: build UI — op history list, undo button, reset button
#   def refresh_stack: read PipelineState and repopulate the list widget
#   def on_undo_clicked: call pipeline_state.undo → emit current image
#   def on_reset_clicked: call pipeline_state.reset → emit original image
# signal: pipeline_changed(np.ndarray) — emits the image after undo or reset

# --- UTIL USAGE GUIDE ---
# from utils.pipeline_state import PipelineState
# from utils.image_utils import to_qpixmap
# from utils.error_handler import wrap_errors
#
# pipeline.undo()                     # call inside on_undo_clicked — returns previous image
# pipeline.reset()                    # call inside on_reset_clicked — returns original image
# pipeline.get_stack_names()          # call inside refresh_stack to populate the list widget
# to_qpixmap(image)                   # call on the returned image before emitting pipeline_changed
# @wrap_errors                        # decorate on_undo_clicked and on_reset_clicked

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
                              QListWidgetItem, QPushButton, QLabel, QFrame,
                              QGridLayout, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from gui.styles import (BG, PANEL, PANEL2, INPUT, BORDER, BORDER2,
                        ACCENT, ACCENT_DIM, TEXT, MUTED, MUTED2, btn_style)
from utils import wrap_errors


class PipelinePanel(QWidget):
    undo_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    checkpoint_save_requested = pyqtSignal(str)
    checkpoint_restore_requested = pyqtSignal(str)

    _SLOTS = ['A', 'B', 'C', 'D']

    def __init__(self, pipeline_state, parent=None):
        super().__init__(parent)
        self._pipeline = pipeline_state
        self._slot_frames: dict = {}
        self._slot_labels: dict = {}
        self._checkpoints: dict = {}

        self.setStyleSheet(f"background: {PANEL}; border: none;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # section header
        hdr = QPushButton("▾  PIPELINE")
        hdr.setStyleSheet(
            f"QPushButton {{ background: {PANEL2}; color: {ACCENT}; border: none; "
            f"border-bottom: 1px solid {BORDER}; padding: 6px 10px; "
            f"font-size: 9px; font-weight: bold; letter-spacing: 0.15em; text-align: left; }}"
            f"QPushButton:hover {{ background: {INPUT}; }}"
        )
        layout.addWidget(hdr)

        # checkpoint 2×2 grid
        ckpt_w = QWidget()
        ckpt_w.setStyleSheet(f"background: {PANEL};")
        ckpt_layout = QGridLayout(ckpt_w)
        ckpt_layout.setContentsMargins(8, 8, 8, 4)
        ckpt_layout.setSpacing(4)

        slot_lbl = QLabel("CHECKPOINTS")
        slot_lbl.setStyleSheet(
            f"color: {MUTED}; font-size: 8px; font-weight: bold; letter-spacing: 0.12em;"
        )
        ckpt_layout.addWidget(slot_lbl, 0, 0, 1, 2)

        positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
        for i, slot in enumerate(self._SLOTS):
            r, c = positions[i]
            frame = QFrame()
            frame.setFixedHeight(36)
            frame.setCursor(Qt.CursorShape.PointingHandCursor)
            lbl = QLabel(f"{slot}  ·  empty")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(4, 2, 4, 2)
            fl.addWidget(lbl)
            self._style_slot(frame, lbl, saved=False)
            self._slot_frames[slot] = frame
            self._slot_labels[slot] = lbl
            frame.mousePressEvent = self._make_slot_click(slot)
            ckpt_layout.addWidget(frame, r, c)

        save_btn = QPushButton("Save to next slot")
        save_btn.setStyleSheet(btn_style('ghost'))
        save_btn.clicked.connect(self._save_to_next_slot)
        ckpt_layout.addWidget(save_btn, 3, 0, 1, 2)
        layout.addWidget(ckpt_w)

        # branch divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"background: {BORDER}; max-height: 1px;")
        layout.addWidget(div)

        # op stack header
        stack_hdr = QWidget()
        stack_hdr.setStyleSheet(f"background: {PANEL};")
        shl = QHBoxLayout(stack_hdr)
        shl.setContentsMargins(10, 6, 8, 4)
        shl.addWidget(QLabel("OP STACK", styleSheet=f"color:{MUTED};font-size:8px;font-weight:bold;letter-spacing:.12em;"))
        shl.addStretch()
        self._count_badge = QLabel("0")
        self._count_badge.setStyleSheet(
            f"background:{INPUT};color:{ACCENT};font-size:9px;font-weight:bold;padding:1px 6px;border-radius:8px;"
        )
        shl.addWidget(self._count_badge)
        layout.addWidget(stack_hdr)

        self._stack_list = QListWidget()
        self._stack_list.setStyleSheet(
            f"QListWidget{{background:{BG};border:none;color:{TEXT};}}"
            f"QListWidget::item{{padding:4px 8px;border-bottom:1px solid {BORDER};font-size:10px;}}"
            f"QListWidget::item:selected{{background:#1a2208;color:{ACCENT};}}"
        )
        self._stack_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._stack_list)

        # undo / reset
        btn_row = QWidget()
        btn_row.setStyleSheet(f"background:{PANEL};")
        brl = QHBoxLayout(btn_row)
        brl.setContentsMargins(8, 6, 8, 8)
        brl.setSpacing(6)
        self._undo_btn = QPushButton("↩  Undo")
        self._undo_btn.setStyleSheet(btn_style('danger'))
        self._undo_btn.clicked.connect(self.undo_requested)
        brl.addWidget(self._undo_btn)
        self._reset_btn = QPushButton("⟳  Reset")
        self._reset_btn.setStyleSheet(btn_style('ghost'))
        self._reset_btn.clicked.connect(self.reset_requested)
        brl.addWidget(self._reset_btn)
        layout.addWidget(btn_row)

    def _style_slot(self, frame: QFrame, label: QLabel, saved: bool, name: str = ""):
        if saved:
            frame.setStyleSheet(
                f"QFrame{{background:#0f1f05;border:1px solid {ACCENT_DIM};border-radius:2px;}}"
            )
            label.setStyleSheet(f"color:{ACCENT};font-size:9px;font-style:normal;font-weight:bold;")
        else:
            frame.setStyleSheet(
                f"QFrame{{background:{INPUT};border:1px solid {BORDER};border-radius:2px;}}"
                f"QFrame:hover{{border-color:{BORDER2};}}"
            )
            label.setStyleSheet(f"color:{MUTED2};font-size:9px;font-style:italic;")

    def _make_slot_click(self, slot: str):
        def handler(event):
            if slot in self._checkpoints:
                self.checkpoint_restore_requested.emit(slot)
        return handler

    def _save_to_next_slot(self):
        for slot in self._SLOTS:
            if slot not in self._checkpoints:
                self.checkpoint_save_requested.emit(slot)
                return
        self.checkpoint_save_requested.emit(self._SLOTS[-1])

    def refresh_stack(self):
        names = self._pipeline.get_stack_names()
        self._stack_list.clear()
        self._count_badge.setText(str(len(names)))
        for i, name in enumerate(reversed(names)):
            idx = len(names) - i
            item = QListWidgetItem(f"  {idx:02d}  {name}")
            if i == 0:
                item.setForeground(QColor(ACCENT))
                item.setBackground(QColor("#0f1f05"))
            else:
                item.setForeground(QColor(TEXT))
            self._stack_list.addItem(item)

    def update_checkpoints(self):
        self._checkpoints = self._pipeline.get_checkpoints()
        for slot in self._SLOTS:
            frame = self._slot_frames[slot]
            lbl = self._slot_labels[slot]
            if slot in self._checkpoints:
                op = self._checkpoints[slot]
                short = op[:10] + "…" if len(op) > 10 else op
                lbl.setText(f"{slot}  ·  {short}")
                self._style_slot(frame, lbl, saved=True, name=op)
            else:
                lbl.setText(f"{slot}  ·  empty")
                self._style_slot(frame, lbl, saved=False)
