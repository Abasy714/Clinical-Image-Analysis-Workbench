_PALETTES = {
    "Dark Lime": {
        "BG": "#111210", "PANEL": "#181917", "PANEL2": "#1e1f1d",
        "INPUT": "#252623", "BORDER": "#2c2e2a", "BORDER2": "#353730",
        "ACCENT": "#c8f135", "ACCENT2": "#a8d420", "ACCENT_DIM": "#6a8a10",
        "RED": "#ff4d3a", "AMBER": "#f5a623",
        "TEXT": "#eceee8", "MUTED": "#6b6f65", "MUTED2": "#4a4d46",
        "WHITE": "#f4f6f0",
    },
    "Dark Blue": {
        "BG": "#0e1117", "PANEL": "#161b26", "PANEL2": "#1c2333",
        "INPUT": "#222d3d", "BORDER": "#2a3548", "BORDER2": "#354260",
        "ACCENT": "#4fa3e0", "ACCENT2": "#3a88c8", "ACCENT_DIM": "#2060a0",
        "RED": "#ff4d3a", "AMBER": "#f5a623",
        "TEXT": "#e8ecf4", "MUTED": "#6b7a9f", "MUTED2": "#4a5470",
        "WHITE": "#f0f4f8",
    },
    "Dark Purple": {
        "BG": "#110e17", "PANEL": "#181526", "PANEL2": "#1e1a30",
        "INPUT": "#25213a", "BORDER": "#2c2848", "BORDER2": "#3a3460",
        "ACCENT": "#b07aff", "ACCENT2": "#9060e0", "ACCENT_DIM": "#6040a0",
        "RED": "#ff4d3a", "AMBER": "#f5a623",
        "TEXT": "#ece8f4", "MUTED": "#6b659f", "MUTED2": "#4a4570",
        "WHITE": "#f4f0f8",
    },
    "DICOM Gray": {
        "BG": "#0a0a0a", "PANEL": "#141414", "PANEL2": "#1a1a1a",
        "INPUT": "#222222", "BORDER": "#2a2a2a", "BORDER2": "#333333",
        "ACCENT": "#d0d0d0", "ACCENT2": "#b0b0b0", "ACCENT_DIM": "#808080",
        "RED": "#cc3322", "AMBER": "#cc8800",
        "TEXT": "#e0e0e0", "MUTED": "#606060", "MUTED2": "#404040",
        "WHITE": "#f0f0f0",
    },
    "Light": {
        "BG": "#f5f5f5", "PANEL": "#ffffff", "PANEL2": "#eeeeee",
        "INPUT": "#e8e8e8", "BORDER": "#cccccc", "BORDER2": "#aaaaaa",
        "ACCENT": "#2a7a2a", "ACCENT2": "#1a6a1a", "ACCENT_DIM": "#4a9a4a",
        "RED": "#cc2222", "AMBER": "#cc7700",
        "TEXT": "#111111", "MUTED": "#666666", "MUTED2": "#999999",
        "WHITE": "#ffffff",
    },
}

_current = "Dark Lime"


def get() -> dict:
    return _PALETTES[_current]


def set_theme(name: str):
    global _current
    if name not in _PALETTES:
        raise ValueError(f"Unknown theme: {name}")
    _current = name


def names() -> list:
    return list(_PALETTES.keys())
