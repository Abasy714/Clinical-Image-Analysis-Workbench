# STATUS: IMPLEMENTED
"""Palette constants, global stylesheet, and button style helpers for the Clinical Image Analysis Workbench."""

BG         = "#111210"
PANEL      = "#181917"
PANEL2     = "#1e1f1d"
INPUT      = "#252623"
BORDER     = "#2c2e2a"
BORDER2    = "#353730"
ACCENT     = "#c8f135"
ACCENT2    = "#a8d420"
ACCENT_DIM = "#6a8a10"
RED        = "#ff4d3a"
AMBER      = "#f5a623"
TEXT       = "#eceee8"
MUTED      = "#6b6f65"
MUTED2     = "#4a4d46"
WHITE      = "#f4f6f0"

HEADER_SS = """
    QLabel {
        color: #c8f135;
        font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
        font-size: 9px;
        font-weight: bold;
        letter-spacing: 2px;
        padding-bottom: 4px;
        border-bottom: 1px solid #2c2e2a;
    }
"""

FIELD_SS = """
    QLabel {
        color: #6b6f65;
        font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
        font-size: 9px;
        letter-spacing: 1px;
        padding-top: 4px;
    }
"""

COMBO_SS = """
    QComboBox {
        background: #252623;
        color: #eceee8;
        border: 1px solid #353730;
        border-radius: 2px;
        padding: 5px 10px;
        font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
        font-size: 11px;
        min-height: 28px;
    }
    QComboBox:focus {
        border-color: #c8f135;
    }
    QComboBox::drop-down {
        border: none;
        width: 24px;
        padding-right: 6px;
    }
    QComboBox::down-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid #6b6f65;
        width: 0;
        height: 0;
        margin-right: 6px;
    }
    QComboBox QAbstractItemView {
        background: #181917;
        border: 1px solid #353730;
        color: #eceee8;
        selection-background-color: #1a2208;
        selection-color: #c8f135;
        padding: 2px;
        outline: none;
    }
    QComboBox QAbstractItemView::item {
        padding: 5px 10px;
        min-height: 24px;
    }
    QComboBox QAbstractItemView::item:hover {
        background: #252623;
    }
"""

SPINBOX_SS = """
    QDoubleSpinBox, QSpinBox {
        background: #252623;
        color: #eceee8;
        border: 1px solid #353730;
        border-radius: 2px;
        padding: 4px 8px;
        font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
        font-size: 11px;
        min-height: 26px;
    }
    QDoubleSpinBox:focus, QSpinBox:focus {
        border-color: #c8f135;
    }
    QDoubleSpinBox::up-button, QSpinBox::up-button,
    QDoubleSpinBox::down-button, QSpinBox::down-button {
        background: #353730;
        border: none;
        width: 16px;
    }
    QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
    QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {
        background: #484b44;
    }
    QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {
        border-left: 3px solid transparent;
        border-right: 3px solid transparent;
        border-bottom: 4px solid #6b6f65;
        width: 0; height: 0;
    }
    QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {
        border-left: 3px solid transparent;
        border-right: 3px solid transparent;
        border-top: 4px solid #6b6f65;
        width: 0; height: 0;
    }
"""

APPLY_BTN_SS = """
    QPushButton {
        background: #c8f135;
        color: #0d1002;
        border: 1px solid #a8d420;
        border-radius: 2px;
        padding: 10px;
        font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
        font-size: 11px;
        font-weight: bold;
        letter-spacing: 1px;
        min-height: 36px;
    }
    QPushButton:hover {
        background: #a8d420;
        border-color: #8ab810;
    }
    QPushButton:pressed {
        background: #8ab810;
    }
    QPushButton:disabled {
        background: #2c2e2a;
        color: #4a4d46;
        border-color: #2c2e2a;
    }
"""

MAIN_STYLE = """
QMainWindow, QWidget {
    background-color: #111210;
    color: #eceee8;
    font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
    font-size: 11px;
}
QMenuBar {
    background-color: #0d0e0c;
    color: #8a8e84;
    border-bottom: 1px solid #2c2e2a;
    padding: 2px;
}
QMenuBar::item:selected { background: #252623; color: #eceee8; }
QMenu { background: #181917; border: 1px solid #2c2e2a; color: #eceee8; }
QMenu::item:selected { background: #252623; }
QStatusBar {
    background: #0d0e0c;
    border-top: 1px solid #2c2e2a;
    color: #6b6f65;
    font-size: 10px;
}
QStatusBar::item { border: none; }
QScrollBar:vertical { background: #111210; width: 6px; border: none; }
QScrollBar::handle:vertical { background: #2c2e2a; border-radius: 3px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #111210; height: 6px; border: none; }
QScrollBar::handle:horizontal { background: #2c2e2a; border-radius: 3px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QComboBox {
    background: #252623; border: 1px solid #353730; border-radius: 2px;
    color: #eceee8; padding: 4px 8px; font-family: inherit;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background: #181917; border: 1px solid #353730; color: #eceee8;
    selection-background-color: #252623;
}
QLineEdit, QSpinBox, QDoubleSpinBox {
    background: #252623; border: 1px solid #353730; border-radius: 2px;
    color: #eceee8; padding: 4px 8px; font-family: inherit;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #c8f135; }
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background: #353730; border: none; width: 14px;
}
QSlider::groove:horizontal { height: 3px; background: #353730; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 12px; height: 12px; margin: -5px 0;
    background: #c8f135; border-radius: 6px; border: 2px solid #0d1002;
}
QSlider::sub-page:horizontal { background: #c8f135; border-radius: 2px; }
QTabWidget::pane { border: none; background: #181917; }
QTabBar::tab {
    background: #181917; color: #6b6f65; border: none;
    padding: 8px 12px; font-size: 9px; font-weight: bold;
    letter-spacing: 0.12em; border-bottom: 2px solid transparent;
}
QTabBar::tab:selected { color: #c8f135; border-bottom-color: #c8f135; }
QTabBar::tab:hover { color: #eceee8; }
QListWidget { background: #111210; border: none; color: #eceee8; outline: none; }
QListWidget::item { padding: 5px 8px; border-bottom: 1px solid #2c2e2a; }
QListWidget::item:selected { background: #1a2208; color: #c8f135; border: none; }
QTableWidget { background: #111210; border: none; gridline-color: #2c2e2a; color: #eceee8; }
QTableWidget::item { padding: 3px 6px; border: none; }
QHeaderView::section { background: #181917; color: #6b6f65; border: none; padding: 4px; }
QDialog { background: #111210; color: #eceee8; }
QGroupBox {
    border: 1px solid #2c2e2a; border-radius: 2px;
    margin-top: 6px; padding-top: 6px; color: #6b6f65; font-size: 9px;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #6b6f65; }
QToolTip {
    background: #181917; color: #eceee8;
    border: 1px solid #2c2e2a; font-size: 10px;
}
"""


def btn_style(variant: str = 'default') -> str:
    base = (
        "border-radius: 2px; padding: 6px 10px; font-family: inherit; "
        "font-size: 10px; font-weight: bold; letter-spacing: 0.06em; border: 1px solid;"
    )
    if variant == 'primary':
        return (
            f"QPushButton {{ {base} background: #c8f135; color: #0d1002; border-color: #a8d420; }}"
            "QPushButton:hover { background: #a8d420; }"
            "QPushButton:pressed { background: #8ab818; }"
            "QPushButton:disabled { background: #2c2e2a; color: #4a4d46; border-color: #2c2e2a; }"
        )
    if variant == 'danger':
        return (
            f"QPushButton {{ {base} background: #2d1210; color: #ff4d3a; border-color: #5a1f1a; }}"
            "QPushButton:hover { background: #3d1815; }"
            "QPushButton:pressed { background: #1e0c0a; }"
        )
    if variant == 'ghost':
        return (
            f"QPushButton {{ {base} background: transparent; color: #6b6f65; border-color: #353730; }}"
            "QPushButton:hover { color: #eceee8; border-color: #484b44; }"
            "QPushButton:pressed { background: #1e1f1d; }"
        )
    return (
        f"QPushButton {{ {base} background: #252623; color: #eceee8; border-color: #353730; }}"
        "QPushButton:hover { background: #353730; }"
        "QPushButton:pressed { background: #1e1f1d; }"
    )


def apply_styles(app) -> None:
    from PyQt6.QtGui import QFont
    font = QFont("JetBrains Mono")
    font.setStyleHint(QFont.StyleHint.Monospace)
    app.setFont(font)
    app.setStyleSheet(MAIN_STYLE)


def build() -> str:
    from gui.theme import get
    p = get()
    return f"""
QMainWindow, QWidget {{
    background-color: {p['BG']};
    color: {p['TEXT']};
    font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
    font-size: 11px;
}}
QMenuBar {{
    background-color: {p['BG']};
    color: {p['MUTED']};
    border-bottom: 1px solid {p['BORDER']};
    padding: 2px;
}}
QMenuBar::item:selected {{ background: {p['INPUT']}; color: {p['TEXT']}; }}
QMenu {{ background: {p['PANEL']}; border: 1px solid {p['BORDER']}; color: {p['TEXT']}; }}
QMenu::item:selected {{ background: {p['INPUT']}; }}
QStatusBar {{
    background: {p['BG']};
    border-top: 1px solid {p['BORDER']};
    color: {p['MUTED']};
    font-size: 10px;
}}
QStatusBar::item {{ border: none; }}
QScrollBar:vertical {{ background: {p['BG']}; width: 6px; border: none; }}
QScrollBar::handle:vertical {{ background: {p['BORDER']}; border-radius: 3px; min-height: 20px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: {p['BG']}; height: 6px; border: none; }}
QScrollBar::handle:horizontal {{ background: {p['BORDER']}; border-radius: 3px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QComboBox {{
    background: {p['INPUT']}; border: 1px solid {p['BORDER2']}; border-radius: 2px;
    color: {p['TEXT']}; padding: 4px 8px; font-family: inherit;
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {p['PANEL']}; border: 1px solid {p['BORDER2']}; color: {p['TEXT']};
    selection-background-color: {p['INPUT']};
}}
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {p['INPUT']}; border: 1px solid {p['BORDER2']}; border-radius: 2px;
    color: {p['TEXT']}; padding: 4px 8px; font-family: inherit;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {p['ACCENT']}; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {p['BORDER2']}; border: none; width: 14px;
}}
QSlider::groove:horizontal {{ height: 3px; background: {p['BORDER2']}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 12px; height: 12px; margin: -5px 0;
    background: {p['ACCENT']}; border-radius: 6px; border: 2px solid {p['BG']};
}}
QSlider::sub-page:horizontal {{ background: {p['ACCENT']}; border-radius: 2px; }}
QTabWidget::pane {{ border: none; background: {p['PANEL']}; }}
QTabBar::tab {{
    background: {p['PANEL']}; color: {p['MUTED']}; border: none;
    padding: 8px 12px; font-size: 9px; font-weight: bold;
    letter-spacing: 0.12em; border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {p['ACCENT']}; border-bottom-color: {p['ACCENT']}; }}
QTabBar::tab:hover {{ color: {p['TEXT']}; }}
QListWidget {{ background: {p['BG']}; border: none; color: {p['TEXT']}; outline: none; }}
QListWidget::item {{ padding: 5px 8px; border-bottom: 1px solid {p['BORDER']}; }}
QListWidget::item:selected {{ background: {p['PANEL2']}; color: {p['ACCENT']}; border: none; }}
QTableWidget {{ background: {p['BG']}; border: none; gridline-color: {p['BORDER']}; color: {p['TEXT']}; }}
QTableWidget::item {{ padding: 3px 6px; border: none; }}
QHeaderView::section {{ background: {p['PANEL']}; color: {p['MUTED']}; border: none; padding: 4px; }}
QDialog {{ background: {p['BG']}; color: {p['TEXT']}; }}
QGroupBox {{
    border: 1px solid {p['BORDER']}; border-radius: 2px;
    margin-top: 6px; padding-top: 6px; color: {p['MUTED']}; font-size: 9px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; color: {p['MUTED']}; }}
QToolTip {{
    background: {p['PANEL']}; color: {p['TEXT']};
    border: 1px solid {p['BORDER']}; font-size: 10px;
}}
"""
