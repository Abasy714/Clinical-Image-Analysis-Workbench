# STATUS: IMPLEMENTED
"""
Panel for displaying image metadata extracted from DICOM, JPEG, and BMP files.
Shows width, height, bit depth, and DICOM-specific tags when available.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel,
                              QTableWidget, QTableWidgetItem, QHeaderView,
                              QPushButton)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from gui.styles import (BG, PANEL, PANEL2, INPUT, BORDER, BORDER2,
                        ACCENT, ACCENT_DIM, TEXT, MUTED, MUTED2)
from utils import wrap_errors


class MetadataPanel(QWidget):
    """
    Displays image metadata in a two-column table (key / value).
    The 'IMAGE METADATA' section header is collapsible.
    """

    DICOM_DISPLAY_KEYS = [
        ('File',       'filename'),
        ('Modality',   'Modality'),
        ('Width',      'width'),
        ('Height',     'height'),
        ('Bit depth',  'BitsStored'),
        ('Patient',    'PatientName'),
        ('Age',        'PatientAge'),
        ('Body part',  'BodyPartExamined'),
        ('KVP',        'KVP'),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {PANEL}; border: none;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header_btn = QPushButton("▾  IMAGE METADATA")
        self._header_btn.setStyleSheet(
            f"QPushButton {{ background: {PANEL2}; color: {ACCENT}; border: none; "
            f"border-bottom: 1px solid {BORDER}; padding: 6px 10px; "
            f"font-size: 9px; font-weight: bold; letter-spacing: 0.15em; text-align: left; }}"
            f"QPushButton:hover {{ background: {INPUT}; }}"
        )
        self._header_btn.clicked.connect(self._toggle)
        layout.addWidget(self._header_btn)

        self._table = QTableWidget(len(self.DICOM_DISPLAY_KEYS), 2)
        self._table.horizontalHeader().setVisible(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.setColumnWidth(0, 72)
        self._table.setStyleSheet(
            f"QTableWidget {{ background: {PANEL}; border: none; color: {TEXT}; }}"
            f"QTableWidget::item {{ padding: 3px 8px; border-bottom: 1px solid {BORDER}; }}"
        )

        row_h = 20
        for row, (label, _) in enumerate(self.DICOM_DISPLAY_KEYS):
            self._table.setRowHeight(row, row_h)
            key_item = QTableWidgetItem(label)
            key_item.setForeground(QColor(MUTED))
            key_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._table.setItem(row, 0, key_item)

            val_item = QTableWidgetItem("—")
            val_item.setForeground(QColor(MUTED2))
            val_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._table.setItem(row, 1, val_item)

        self._table.setFixedHeight(row_h * len(self.DICOM_DISPLAY_KEYS) + 2)
        layout.addWidget(self._table)
        self._collapsed = False

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._table.setVisible(not self._collapsed)
        arrow = "▸" if self._collapsed else "▾"
        self._header_btn.setText(f"{arrow}  IMAGE METADATA")

    @wrap_errors
    def update_metadata(self, metadata: dict):
        for row, (_, key) in enumerate(self.DICOM_DISPLAY_KEYS):
            value = metadata.get(key, "")
            text = str(value).strip() if value is not None else ""
            text = text or "—"

            val_item = self._table.item(row, 1)
            if val_item is None:
                val_item = QTableWidgetItem()
                self._table.setItem(row, 1, val_item)
            val_item.setFlags(Qt.ItemFlag.ItemIsEnabled)

            if key == 'Modality' and text != "—":
                val_item.setText(f"  {text}  ")
                val_item.setBackground(QColor(ACCENT_DIM))
                val_item.setForeground(QColor(ACCENT))
            else:
                val_item.setText(text)
                val_item.setBackground(QColor(PANEL))
                val_item.setForeground(QColor(TEXT))

    def clear(self):
        for row in range(len(self.DICOM_DISPLAY_KEYS)):
            val_item = self._table.item(row, 1)
            if val_item:
                val_item.setText("—")
                val_item.setBackground(QColor(PANEL))
                val_item.setForeground(QColor(MUTED2))
