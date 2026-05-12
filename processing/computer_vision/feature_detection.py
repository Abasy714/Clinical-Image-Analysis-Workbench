import numpy as np
from utils.image_utils import normalize_to_uint8


def detect_harris_corners(image: np.ndarray, k: float = 0.04,
                           threshold: float = 0.01) -> np.ndarray:
    if image.ndim == 3:
        gray = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1]
                + 0.114 * image[:, :, 2]).astype(np.uint8)
    else:
        gray = normalize_to_uint8(image)

    from processing.spatial.edge_detection import sobel
    from processing.spatial.smoothing import gaussian_filter

    gx, gy, _ = sobel(gray)
    Ix = gx.astype(np.float64)
    Iy = gy.astype(np.float64)

    Ix2_u8  = np.clip(Ix * Ix / 255.0, 0, 255).astype(np.uint8)
    Iy2_u8  = np.clip(Iy * Iy / 255.0, 0, 255).astype(np.uint8)
    IxIy_u8 = np.clip(np.abs(Ix * Iy) / 255.0, 0, 255).astype(np.uint8)

    Ix2_s  = gaussian_filter(Ix2_u8,  3, 1.0).astype(np.float64)
    Iy2_s  = gaussian_filter(Iy2_u8,  3, 1.0).astype(np.float64)
    IxIy_s = gaussian_filter(IxIy_u8, 3, 1.0).astype(np.float64)

    det   = Ix2_s * Iy2_s - IxIy_s ** 2
    trace = Ix2_s + Iy2_s
    R     = det - k * (trace ** 2)

    r_max = R.max()
    if r_max <= 0:
        return np.stack([gray, gray, gray], axis=-1).astype(np.uint8)

    corners = (R > threshold * r_max) & (R > 0)

    H, W = corners.shape
    suppressed = np.zeros_like(corners, dtype=bool)
    pad = 2
    R_pad = np.pad(R, pad, mode='constant', constant_values=-np.inf)
    for r in range(H):
        for c in range(W):
            if corners[r, c]:
                window = R_pad[r:r + 2*pad + 1, c:c + 2*pad + 1]
                if R[r, c] == window.max():
                    suppressed[r, c] = True

    rgb = np.stack([gray, gray, gray], axis=-1).copy()

    ys, xs = np.where(suppressed)
    for yr, xc in zip(ys, xs):
        for dr in range(-4, 5):
            for dc in range(-4, 5):
                if dr ** 2 + dc ** 2 <= 16:
                    nr, nc = yr + dr, xc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        rgb[nr, nc] = [255, 60, 60]

    return rgb.astype(np.uint8)
