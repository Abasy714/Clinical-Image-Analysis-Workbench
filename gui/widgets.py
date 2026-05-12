"""Shared reusable UI components for CIAW."""
import logging
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QListWidget, QPlainTextEdit, QFrame, QGridLayout,
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QRectF
from PyQt6.QtGui import (
    QColor, QTextCharFormat, QTextCursor, QPainter, QLinearGradient, QPen,
)

import gui.theme as _theme


# ── SectionHeader ──────────────────────────────────────────────────────────────

class SectionHeader(QWidget):
    """Accent bar + small-caps title + optional badge/collapse."""
    toggled = pyqtSignal(bool)

    def __init__(self, title: str, collapsible: bool = False,
                 badge: str | None = None, parent=None):
        super().__init__(parent)
        self._expanded = True
        lyt = QHBoxLayout(self)
        lyt.setContentsMargins(0, 4, 0, 4)
        lyt.setSpacing(6)

        self._bar = QFrame()
        self._bar.setFixedSize(3, 14)
        lyt.addWidget(self._bar)

        self._title_lbl = QLabel(title.upper())
        lyt.addWidget(self._title_lbl)

        self._badge_lbl = QLabel(badge) if badge is not None else None
        if self._badge_lbl:
            lyt.addWidget(self._badge_lbl)

        lyt.addStretch()

        if collapsible:
            self._toggle_btn = QPushButton("▾")
            self._toggle_btn.setFixedSize(16, 16)
            self._toggle_btn.setFlat(True)
            self._toggle_btn.clicked.connect(self._on_toggle)
            lyt.addWidget(self._toggle_btn)
        else:
            self._toggle_btn = None

        self._apply_theme()

    def _apply_theme(self):
        t = _theme.get()
        self._bar.setStyleSheet(f"background: {t['ACCENT']}; border: none;")
        self._title_lbl.setStyleSheet(
            f"color: {t['MUTED']}; font-size: 9px; font-weight: bold; "
            f"letter-spacing: 1.5px; font-family: 'JetBrains Mono', Consolas, monospace; "
            f"background: transparent;"
        )
        if self._badge_lbl:
            self._badge_lbl.setStyleSheet(
                f"color: {t['ACCENT']}; font-size: 9px; background: transparent; "
                f"font-family: 'JetBrains Mono', Consolas, monospace;"
            )
        if self._toggle_btn:
            self._toggle_btn.setStyleSheet(
                f"QPushButton {{ color: {t['MUTED']}; background: transparent; "
                f"border: none; font-size: 10px; }}"
                f"QPushButton:hover {{ color: {t['TEXT']}; }}"
            )

    def set_badge(self, text: str):
        if self._badge_lbl:
            self._badge_lbl.setText(text)

    def _on_toggle(self):
        self._expanded = not self._expanded
        if self._toggle_btn:
            self._toggle_btn.setText("▾" if self._expanded else "▸")
        self.toggled.emit(self._expanded)

    def theme_changed(self, palette: dict):
        self._apply_theme()


# ── CollapsibleSection ─────────────────────────────────────────────────────────

class CollapsibleSection(QWidget):
    """Collapsible panel with SectionHeader."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        lyt = QVBoxLayout(self)
        lyt.setContentsMargins(0, 0, 0, 0)
        lyt.setSpacing(0)

        self._header = SectionHeader(title, collapsible=True)
        self._header.toggled.connect(self._on_toggled)
        lyt.addWidget(self._header)

        self._body = QWidget()
        self._body_lyt = QVBoxLayout(self._body)
        self._body_lyt.setContentsMargins(0, 2, 0, 2)
        lyt.addWidget(self._body)

    def set_content(self, widget: QWidget):
        self._body_lyt.addWidget(widget)

    def toggle(self):
        self._header._on_toggle()

    def _on_toggled(self, expanded: bool):
        self._body.setVisible(expanded)

    def theme_changed(self, palette: dict):
        self._header.theme_changed(palette)


# ── PillGroup ──────────────────────────────────────────────────────────────────

class PillGroup(QWidget):
    """Horizontal group of buttons sharing a pill container border."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lyt = QHBoxLayout(self)
        self._lyt.setContentsMargins(0, 0, 0, 0)
        self._lyt.setSpacing(0)
        self._buttons: list = []

    def add_button(self, text: str, checkable: bool = False) -> QPushButton:
        btn = QPushButton(text)
        btn.setCheckable(checkable)
        self._buttons.append(btn)
        self._lyt.addWidget(btn)
        self._refresh_styles()
        return btn

    def _refresh_styles(self):
        t = _theme.get()
        n = len(self._buttons)
        for i, btn in enumerate(self._buttons):
            if n == 1:
                rr = "4px"
            elif i == 0:
                rr = "4px 0 0 4px"
            elif i == n - 1:
                rr = "0 4px 4px 0"
            else:
                rr = "0"
            border_l = "none" if i > 0 else f"1px solid {t['BORDER2']}"
            btn.setStyleSheet(
                f"QPushButton {{ background: {t['INPUT']}; color: {t['TEXT']}; "
                f"border: 1px solid {t['BORDER2']}; border-left: {border_l}; "
                f"border-radius: {rr}; padding: 4px 8px; font-size: 10px; "
                f"font-family: 'JetBrains Mono', Consolas, monospace; }}"
                f"QPushButton:hover {{ background: {t['PANEL2']}; }}"
                f"QPushButton:checked {{ background: {t['ACCENT']}; color: {t['DARK']}; "
                f"border-color: {t['ACCENT']}; }}"
            )

    def theme_changed(self, palette: dict):
        self._refresh_styles()


# ── StatGrid ───────────────────────────────────────────────────────────────────

class StatGrid(QWidget):
    """N×M grid of (label, value) pairs with bg_input background."""

    def __init__(self, fields: list, cols: int = 3, parent=None):
        super().__init__(parent)
        self._values: dict = {}
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        t = _theme.get()
        for idx, key in enumerate(fields):
            row, col = divmod(idx, cols)
            cell = QWidget()
            cell.setObjectName("StatCell")
            cell.setStyleSheet(
                f"QWidget#StatCell {{ background: {t['INPUT']}; border-radius: 4px; }}"
            )
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(4, 4, 4, 4)
            cl.setSpacing(1)
            lbl = QLabel(key)
            lbl.setStyleSheet(
                f"color: {t['MUTED']}; font-size: 8px; background: transparent; "
                f"font-family: 'JetBrains Mono', Consolas, monospace;"
            )
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val = QLabel("—")
            val.setStyleSheet(
                f"color: {t['TEXT']}; font-size: 11px; font-weight: bold; "
                f"background: transparent; "
                f"font-family: 'JetBrains Mono', Consolas, monospace;"
            )
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cl.addWidget(lbl)
            cl.addWidget(val)
            grid.addWidget(cell, row, col)
            self._values[key] = val

    def set_value(self, key: str, value: str):
        if key in self._values:
            self._values[key].setText(str(value))


# ── ProgressStrip ──────────────────────────────────────────────────────────────

class ProgressStrip(QWidget):
    """2px animated progress bar for operation feedback."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(2)
        self._pos = 0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self.hide()

    def start(self):
        self._pos = 0
        self.show()
        self._timer.start()

    def stop(self):
        self._timer.stop()
        self.hide()

    def _tick(self):
        w = max(1, self.width() + 200)
        self._pos = (self._pos + 4) % w
        self.update()

    def paintEvent(self, event):
        t = _theme.get()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(t['BG']))
        bar_w = 200
        x = float(self._pos - bar_w)
        grad = QLinearGradient(x, 0.0, x + bar_w, 0.0)
        grad.setColorAt(0.0, QColor(t['BG']))
        grad.setColorAt(0.5, QColor(t['ACCENT']))
        grad.setColorAt(1.0, QColor(t['BG']))
        painter.fillRect(QRectF(x, 0.0, float(bar_w), 2.0), grad)
        painter.end()


# ── OpStackList ────────────────────────────────────────────────────────────────

class OpStackList(QListWidget):
    """Op stack list with flash animation on new entry."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._unflash)
        self._apply_theme()

    def _apply_theme(self):
        t = _theme.get()
        self.setStyleSheet(
            f"QListWidget {{ background: {t['BG']}; border: none; color: {t['TEXT']}; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; font-size: 9px; "
            f"outline: none; }}"
            f"QListWidget::item {{ padding: 3px 6px; border-bottom: 1px solid {t['BORDER']}; }}"
            f"QListWidget::item:selected {{ background: {t['PANEL2']}; color: {t['ACCENT']}; }}"
        )

    def flash_top(self):
        if self.count() == 0:
            return
        t = _theme.get()
        item = self.item(0)
        item.setBackground(QColor(t['ACCENT']))
        item.setForeground(QColor(t['DARK']))
        self._flash_timer.start(600)

    def _unflash(self):
        if self.count() == 0:
            return
        t = _theme.get()
        item = self.item(0)
        item.setBackground(QColor(t['BG']))
        item.setForeground(QColor(t['TEXT']))

    def theme_changed(self, palette: dict):
        self._apply_theme()


# ── ColorLogWidget ─────────────────────────────────────────────────────────────

class ColorLogWidget(QPlainTextEdit):
    """Log widget with per-level text coloring."""

    _LEVEL_KEYS = {
        logging.DEBUG:    'MUTED',
        logging.INFO:     'TEXT',
        logging.WARNING:  'AMBER',
        logging.ERROR:    'RED',
        logging.CRITICAL: 'RED',
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(500)
        self._apply_theme()

    def _apply_theme(self):
        t = _theme.get()
        self.setStyleSheet(
            f"QPlainTextEdit {{ background: {t['BG']}; border: none; "
            f"font-family: 'JetBrains Mono', Consolas, monospace; font-size: 9px; "
            f"padding: 4px; }}"
        )

    @pyqtSlot(str, str)
    def _append_colored(self, msg: str, color: str):
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.setCharFormat(fmt)
        cursor.insertText(msg + '\n')
        self.setTextCursor(cursor)
        self.ensureCursorVisible()

    def theme_changed(self, palette: dict):
        self._apply_theme()


# ── RoiOverlay ─────────────────────────────────────────────────────────────────

class RoiOverlay(QWidget):
    """Marching-ants ROI overlay, placed as child of the image label."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._rect = None
        self._dash_offset = 0
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)

    def set_rect(self, rect):
        self._rect = rect
        if rect is not None and not rect.isEmpty():
            self._timer.start()
            self.show()
        else:
            self._timer.stop()
            self.hide()
        self.update()

    def _tick(self):
        self._dash_offset = (self._dash_offset + 1) % 16
        self.update()

    def paintEvent(self, event):
        if self._rect is None or self._rect.isEmpty():
            return
        t = _theme.get()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        fill = QColor(t['ACCENT'])
        fill.setAlpha(20)
        p.fillRect(self._rect, fill)

        pen = QPen(QColor(t['ACCENT']), 2)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setDashOffset(self._dash_offset)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(self._rect)

        h = 6
        accent = QColor(t['ACCENT'])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(accent)
        for cx, cy in [
            (self._rect.left() - h // 2,  self._rect.top() - h // 2),
            (self._rect.right() - h // 2, self._rect.top() - h // 2),
            (self._rect.left() - h // 2,  self._rect.bottom() - h // 2),
            (self._rect.right() - h // 2, self._rect.bottom() - h // 2),
        ]:
            p.drawRect(cx, cy, h, h)

        p.end()
