import numpy as np
from utils.image_utils import normalize_to_uint8


def sliding_window_classify(image: np.ndarray, window_size: int = 128,
                             stride: int = 32) -> np.ndarray:
    if image.ndim == 3:
        gray = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1]
                + 0.114 * image[:, :, 2]).astype(np.uint8)
    else:
        gray = normalize_to_uint8(image)

    from processing.deep_learning.predictor import predict, is_available
    if not is_available():
        raise RuntimeError("Model weights not found — cannot run sliding window classifier")

    H, W = gray.shape
    row_positions = list(range(0, H - window_size + 1, stride))
    col_positions = list(range(0, W - window_size + 1, stride))

    if not row_positions or not col_positions:
        raise ValueError(f"Image ({H}×{W}) too small for window_size={window_size}")

    n_rows = len(row_positions)
    n_cols = len(col_positions)
    conf_map = np.zeros((n_rows, n_cols), dtype=np.float64)

    rgb_input = np.stack([gray, gray, gray], axis=-1)

    for ri, r in enumerate(row_positions):
        for ci, c in enumerate(col_positions):
            crop = rgb_input[r:r + window_size, c:c + window_size]
            result = predict(crop)
            conf_map[ri, ci] = result.get('confidence', 0.0)

    full_map = np.zeros((H, W), dtype=np.float64)
    scale_r = H / n_rows
    scale_c = W / n_cols
    for ri in range(n_rows):
        r0 = int(ri * scale_r)
        r1 = int((ri + 1) * scale_r) if ri < n_rows - 1 else H
        for ci in range(n_cols):
            c0 = int(ci * scale_c)
            c1 = int((ci + 1) * scale_c) if ci < n_cols - 1 else W
            full_map[r0:r1, c0:c1] = conf_map[ri, ci]

    mn, mx = full_map.min(), full_map.max()
    if mx > mn:
        full_map = (full_map - mn) / (mx - mn)

    try:
        import matplotlib.pyplot as plt
        colored = (plt.get_cmap('jet')(full_map)[:, :, :3] * 255).astype(np.uint8)
    except ImportError:
        heat = (full_map * 255).astype(np.uint8)
        colored = np.stack([heat, np.zeros_like(heat), np.zeros_like(heat)], axis=-1)

    overlay = np.clip(0.5 * rgb_input.astype(np.float64)
                      + 0.5 * colored.astype(np.float64), 0, 255).astype(np.uint8)
    return normalize_to_uint8(overlay)
