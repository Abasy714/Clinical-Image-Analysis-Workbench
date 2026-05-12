"""Palette constants, global stylesheet, and button style helpers."""

# Module-level constants kept for panels that apply them at build time.
# These are Dark Lime defaults; dynamic theming uses build() + btn_style().
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

HEADER_SS = (
    "QLabel {"
    "color: #c8f135;"
    "font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;"
    "font-size: 9px; font-weight: bold; letter-spacing: 2px;"
    "padding-bottom: 4px; border-bottom: 1px solid #2c2e2a;}"
)

FIELD_SS = (
    "QLabel {"
    "color: #6b6f65;"
    "font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;"
    "font-size: 9px; letter-spacing: 1px; padding-top: 4px;}"
)

COMBO_SS = (
    "QComboBox {background: #252623; color: #eceee8; border: 1px solid #353730;"
    "border-radius: 2px; padding: 5px 10px;"
    "font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;"
    "font-size: 11px; min-height: 28px;}"
    "QComboBox:focus {border-color: #c8f135;}"
    "QComboBox::drop-down {border: none; width: 24px; padding-right: 6px;}"
    "QComboBox::down-arrow {image: none; border-left: 4px solid transparent;"
    "border-right: 4px solid transparent; border-top: 5px solid #6b6f65;"
    "width: 0; height: 0; margin-right: 6px;}"
    "QComboBox QAbstractItemView {background: #181917; border: 1px solid #353730;"
    "color: #eceee8; selection-background-color: #1a2208;"
    "selection-color: #c8f135; padding: 2px; outline: none;}"
    "QComboBox QAbstractItemView::item {padding: 5px 10px; min-height: 24px;}"
    "QComboBox QAbstractItemView::item:hover {background: #252623;}"
)

SPINBOX_SS = (
    "QDoubleSpinBox, QSpinBox {background: #252623; color: #eceee8;"
    "border: 1px solid #353730; border-radius: 2px; padding: 4px 8px;"
    "font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;"
    "font-size: 11px; min-height: 26px;}"
    "QDoubleSpinBox:focus, QSpinBox:focus {border-color: #c8f135;}"
    "QDoubleSpinBox::up-button, QSpinBox::up-button,"
    "QDoubleSpinBox::down-button, QSpinBox::down-button"
    " {background: #353730; border: none; width: 16px;}"
    "QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,"
    "QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover"
    " {background: #484b44;}"
    "QDoubleSpinBox::up-arrow, QSpinBox::up-arrow"
    " {border-left: 3px solid transparent; border-right: 3px solid transparent;"
    "border-bottom: 4px solid #6b6f65; width: 0; height: 0;}"
    "QDoubleSpinBox::down-arrow, QSpinBox::down-arrow"
    " {border-left: 3px solid transparent; border-right: 3px solid transparent;"
    "border-top: 4px solid #6b6f65; width: 0; height: 0;}"
)

APPLY_BTN_SS = (
    "QPushButton {background: #c8f135; color: #0d1002; border: 1px solid #a8d420;"
    "border-radius: 2px; padding: 10px;"
    "font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace;"
    "font-size: 11px; font-weight: bold; letter-spacing: 1px; min-height: 36px;}"
    "QPushButton:hover {background: #a8d420; border-color: #8ab810;}"
    "QPushButton:pressed {background: #8ab810;}"
    "QPushButton:disabled {background: #2c2e2a; color: #4a4d46; border-color: #2c2e2a;}"
)

MAIN_STYLE = (
    "QMainWindow, QWidget {background-color: #111210; color: #eceee8;"
    "font-family: 'JetBrains Mono', 'Fira Code', Consolas, monospace; font-size: 11px;}"
    "QMenuBar {background-color: #0d0e0c; color: #8a8e84;"
    "border-bottom: 1px solid #2c2e2a; padding: 2px;}"
    "QMenuBar::item:selected {background: #252623; color: #eceee8;}"
    "QMenu {background: #181917; border: 1px solid #2c2e2a; color: #eceee8;}"
    "QMenu::item:selected {background: #252623;}"
    "QStatusBar {background: #0d0e0c; border-top: 1px solid #2c2e2a;"
    "color: #6b6f65; font-size: 10px;}"
    "QStatusBar::item {border: none;}"
    "QScrollBar:vertical {background: #111210; width: 6px; border: none;}"
    "QScrollBar::handle:vertical {background: #2c2e2a; border-radius: 3px; min-height: 20px;}"
    "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {height: 0;}"
    "QScrollBar:horizontal {background: #111210; height: 6px; border: none;}"
    "QScrollBar::handle:horizontal {background: #2c2e2a; border-radius: 3px;}"
    "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {width: 0;}"
    "QTabWidget::pane {border: none; background: #181917;}"
    "QTabBar::tab {background: #181917; color: #6b6f65; border: none;"
    "padding: 8px 12px; font-size: 9px; font-weight: bold;"
    "letter-spacing: 0.12em; border-bottom: 2px solid transparent;}"
    "QTabBar::tab:selected {color: #c8f135; border-bottom-color: #c8f135;}"
    "QTabBar::tab:hover {color: #eceee8;}"
)


def btn_style(variant: str = 'default') -> str:
    from gui.theme import get
    p = get()
    base = (
        "border-radius: 2px; padding: 6px 10px; font-family: inherit; "
        "font-size: 10px; font-weight: bold; letter-spacing: 0.06em; border: 1px solid;"
    )
    if variant == 'primary':
        return (
            f"QPushButton {{ {base} background: {p['ACCENT']}; color: {p['DARK']};"
            f" border-color: {p['ACCENT2']}; }}"
            f"QPushButton:hover {{ background: {p['ACCENT2']}; }}"
            f"QPushButton:pressed {{ background: {p['ACCENT_DIM']}; }}"
            f"QPushButton:disabled {{ background: {p['BORDER']}; color: {p['MUTED2']};"
            f" border-color: {p['BORDER']}; }}"
        )
    if variant == 'danger':
        return (
            f"QPushButton {{ {base} background: {p['PANEL2']}; color: {p['RED']};"
            f" border-color: {p['RED']}; }}"
            f"QPushButton:hover {{ background: {p['INPUT']}; }}"
            f"QPushButton:pressed {{ background: {p['PANEL']}; }}"
        )
    if variant == 'ghost':
        return (
            f"QPushButton {{ {base} background: transparent; color: {p['MUTED']};"
            f" border-color: {p['BORDER2']}; }}"
            f"QPushButton:hover {{ color: {p['TEXT']}; border-color: {p['MUTED']}; }}"
            f"QPushButton:pressed {{ background: {p['PANEL2']}; }}"
        )
    # default
    return (
        f"QPushButton {{ {base} background: {p['INPUT']}; color: {p['TEXT']};"
        f" border-color: {p['BORDER2']}; }}"
        f"QPushButton:hover {{ background: {p['PANEL2']}; }}"
        f"QPushButton:pressed {{ background: {p['BORDER']}; }}"
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
QPushButton {{
    background: {p['INPUT']}; color: {p['TEXT']}; border: 1px solid {p['BORDER2']};
    border-radius: 2px; padding: 5px 10px;
    font-family: inherit; font-size: 10px; font-weight: bold;
}}
QPushButton:hover {{ background: {p['PANEL2']}; border-color: {p['MUTED']}; }}
QPushButton:pressed {{ background: {p['BORDER']}; }}
QPushButton:checked {{ background: {p['ACCENT_DIM']}; color: {p['ACCENT']}; border-color: {p['ACCENT']}; }}
QPushButton:disabled {{ background: {p['BORDER']}; color: {p['MUTED2']}; border-color: {p['BORDER']}; }}
QCheckBox {{ color: {p['TEXT']}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {p['BORDER2']}; border-radius: 2px;
    background: {p['INPUT']};
}}
QCheckBox::indicator:checked {{ background: {p['ACCENT']}; border-color: {p['ACCENT']}; }}
QCheckBox::indicator:hover {{ border-color: {p['ACCENT']}; }}
QRadioButton {{ color: {p['TEXT']}; spacing: 6px; }}
QRadioButton::indicator {{
    width: 12px; height: 12px;
    border: 1px solid {p['BORDER2']}; border-radius: 6px;
    background: {p['INPUT']};
}}
QRadioButton::indicator:checked {{ background: {p['ACCENT']}; border-color: {p['ACCENT']}; }}
QRadioButton::indicator:hover {{ border-color: {p['ACCENT']}; }}
QComboBox {{
    background: {p['INPUT']}; border: 1px solid {p['BORDER2']}; border-radius: 2px;
    color: {p['TEXT']}; padding: 4px 8px; font-family: inherit;
}}
QComboBox:focus {{ border-color: {p['ACCENT']}; }}
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
QPlainTextEdit {{
    background: {p['INPUT']}; border: 1px solid {p['BORDER2']}; border-radius: 2px;
    color: {p['MUTED']}; padding: 4px; font-family: inherit; font-size: 9px;
}}
QPlainTextEdit:focus {{ border-color: {p['ACCENT']}; }}
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
QScrollArea {{ background: {p['BG']}; border: none; }}
QListWidget {{ background: {p['BG']}; border: none; color: {p['TEXT']}; outline: none; }}
QListWidget::item {{ padding: 5px 8px; border-bottom: 1px solid {p['BORDER']}; }}
QListWidget::item:selected {{ background: {p['PANEL2']}; color: {p['ACCENT']}; border: none; }}
QTableWidget {{ background: {p['BG']}; border: none; gridline-color: {p['BORDER']}; color: {p['TEXT']}; }}
QTableWidget::item {{ padding: 3px 6px; border: none; }}
QHeaderView::section {{ background: {p['PANEL']}; color: {p['MUTED']}; border: none; padding: 4px; }}
QDialog {{ background: {p['BG']}; color: {p['TEXT']}; }}
QGroupBox {{
    border: 1px solid {p['BORDER']}; border-radius: 3px;
    margin-top: 8px; padding-top: 8px;
    font-size: 9px; font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 8px; padding: 0 4px;
    color: {p['ACCENT']}; font-size: 9px; font-weight: bold;
}}
QProgressBar {{
    background: {p['BORDER']}; border: 1px solid {p['BORDER2']};
    border-radius: 2px; height: 6px;
}}
QProgressBar::chunk {{ background: {p['ACCENT']}; border-radius: 2px; }}
QSplitter::handle {{ background: {p['BORDER']}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QRubberBand {{ border: 2px solid {p['ACCENT']}; background: transparent; }}
QToolTip {{
    background: {p['PANEL']}; color: {p['TEXT']};
    border: 1px solid {p['BORDER']}; font-size: 10px;
}}
"""


def tab_style() -> str:
    from gui.theme import get
    p = get()
    return (
        f"QTabWidget::pane {{ border: none; background: {p['PANEL']}; }}"
        f"QTabBar::tab {{ background: {p['PANEL']}; color: {p['MUTED']}; border: none;"
        f" padding: 8px 12px; font-size: 9px; font-weight: bold;"
        f" letter-spacing: 0.12em; border-bottom: 2px solid transparent; }}"
        f"QTabBar::tab:selected {{ color: {p['ACCENT']}; border-bottom-color: {p['ACCENT']}; }}"
        f"QTabBar::tab:hover {{ color: {p['TEXT']}; }}"
    )


def operation_btn_style() -> str:
    from gui.theme import get
    p = get()
    return (
        f"QPushButton {{ background: {p['INPUT']}; color: {p['TEXT']};"
        f" border: 1px solid {p['BORDER2']}; border-radius: 4px;"
        f" font-size: 10px; font-weight: 500; padding: 6px 8px;"
        f" letter-spacing: 0.05em; font-family: inherit; }}"
        f"QPushButton:hover {{ color: {p['ACCENT']}; border-color: {p['ACCENT']}; }}"
        f"QPushButton:pressed {{ background: {p['ACCENT']}; color: {p['DARK']};"
        f" border-color: {p['ACCENT']}; }}"
        f"QPushButton:checked {{ background: {p['ACCENT']}; color: {p['DARK']};"
        f" border-color: {p['ACCENT']}; font-weight: bold; }}"
        f"QPushButton:disabled {{ background: {p['BG']}; color: {p['MUTED']};"
        f" border-color: {p['BORDER']}; }}"
    )


def vertical_tab_style() -> str:
    from gui.theme import get
    p = get()
    return (
        f"QTabWidget::pane {{ border: none; background: {p['PANEL']}; }}"
        f"QTabBar::tab {{ background: {p['PANEL']}; color: {p['MUTED']}; border: none;"
        f" min-width: 80px; max-width: 80px; padding: 6px 4px;"
        f" font-size: 8px; font-weight: bold; letter-spacing: 0.12em;"
        f" border-right: 2px solid transparent; }}"
        f"QTabBar::tab:selected {{ color: {p['ACCENT']}; background: {p['INPUT']};"
        f" border-right-color: {p['ACCENT']}; }}"
        f"QTabBar::tab:hover {{ color: {p['TEXT']}; }}"
    )
