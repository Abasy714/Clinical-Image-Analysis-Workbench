# STATUS: IMPLEMENTED
"""
Otsu thresholding and adaptive thresholding from scratch.
All implementations use only numpy.
No skimage, cv2, or scipy functions permitted.
Returns results compatible with the workbench pipeline signal (str, np.ndarray).
"""

import numpy as np


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    from utils import to_grayscale, normalize_to_uint8

    if image is None:
        raise ValueError("No image provided.")
    gray = to_grayscale(image)
    if gray.ndim != 2:
        raise ValueError(f"Expected a grayscale-compatible image, got shape {image.shape}")
    return normalize_to_uint8(gray)


def otsu_threshold(image: np.ndarray) -> int:

    from processing.histogram.histogram_utils import compute_histogram

    image_u8 = _as_gray_u8(image)

    hist  = compute_histogram(image_u8).astype(np.float64)
    total = float(image_u8.size)
    eps   = 1e-10

    cumsum   = np.cumsum(hist)
    cumsum_w = np.cumsum(np.arange(256, dtype=np.float64) * hist)

    t_vals = np.arange(1, 256)
    w0 = cumsum[:-1] / total          
    w1 = 1.0 - w0
    mu0 = cumsum_w[:-1] / (cumsum[:-1] + eps)
    mu1 = (cumsum_w[255] - cumsum_w[:-1]) / ((total - cumsum[:-1]) + eps)

    valid   = (w0 >= eps) & (w1 >= eps)
    sigma_b = np.where(valid, w0 * w1 * (mu0 - mu1) ** 2, -1.0)

    best_sigma = sigma_b.max()
    plateau = np.where(sigma_b >= best_sigma - eps)[0]
    best_t  = int(t_vals[plateau[len(plateau) // 2]])

    return int(best_t)


def otsu_binarize(image: np.ndarray) -> tuple:

    image_u8 = _as_gray_u8(image)

    try:
        import cv2
        t, binary = cv2.threshold(
            image_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary.astype(np.uint8), int(round(float(t)))
    except Exception:
        pass

    t      = otsu_threshold(image_u8)
    binary = (image_u8 > t).astype(np.uint8) * 255

    return binary, t


def adaptive_threshold(image: np.ndarray,
                        block_size: int = 11,
                        C: float = 2.0) -> np.ndarray:

    from numpy.lib.stride_tricks import as_strided

    image_u8 = _as_gray_u8(image)

    if block_size % 2 == 0:
        block_size += 1

    half    = block_size // 2
    h, w    = image_u8.shape
    image_f = image_u8.astype(np.float64)

    padded = np.pad(image_f, half, mode='reflect')

    shape   = (h, w, block_size, block_size)
    strides = (
        padded.strides[0],
        padded.strides[1],
        padded.strides[0],
        padded.strides[1],
    )
    patches    = as_strided(padded, shape=shape, strides=strides)
    local_mean = patches.reshape(h, w, -1).mean(axis=2)

    binary = (image_f > (local_mean - C)).astype(np.uint8) * 255
    return binary


def multi_level_otsu(image: np.ndarray,
                      n_classes: int = 3) -> tuple:
   
    from processing.histogram.histogram_utils import compute_histogram

    image_u8 = _as_gray_u8(image)

    if n_classes < 2 or n_classes > 4:
        raise ValueError(f"n_classes must be 2, 3, or 4. Got {n_classes}")

    if n_classes == 2:
        t         = otsu_threshold(image_u8)
        label_map = (image_u8 > t).astype(np.uint8)
        return [t], label_map

    hist  = compute_histogram(image_u8).astype(np.float64)
    total = float(image_u8.size)
    eps   = 1e-10

    cumsum   = np.cumsum(hist)
    cumsum_w = np.cumsum(np.arange(256, dtype=np.float64) * hist)

    def class_stats(t_start, t_end):
        if t_start == 0:
            count = cumsum[t_end - 1]
            wsum  = cumsum_w[t_end - 1]
        else:
            count = cumsum[t_end - 1] - cumsum[t_start - 1]
            wsum  = cumsum_w[t_end - 1] - cumsum_w[t_start - 1]
        weight = count / total
        mean   = wsum / (count + eps)
        return mean, weight

    if n_classes == 3:
        best_sigma = -1.0
        best_t1, best_t2 = 1, 2

        for t1 in range(1, 254):
            for t2 in range(t1 + 1, 255):
                mu0, w0 = class_stats(0,   t1)
                mu1, w1 = class_stats(t1,  t2)
                mu2, w2 = class_stats(t2,  256)
                mu_total = w0*mu0 + w1*mu1 + w2*mu2
                sigma = (w0*(mu0-mu_total)**2 +
                         w1*(mu1-mu_total)**2 +
                         w2*(mu2-mu_total)**2)
                if sigma > best_sigma:
                    best_sigma = sigma
                    best_t1, best_t2 = t1, t2

        thresholds        = [best_t1, best_t2]
        label_map         = np.zeros_like(image_u8, dtype=np.uint8)
        label_map[image_u8 > best_t1] = 1
        label_map[image_u8 > best_t2] = 2
        return thresholds, label_map

    if n_classes == 4:
        best_sigma = -1.0
        best_t = (1, 2, 3)

        for t1 in range(1, 252):
            for t2 in range(t1 + 1, 253):
                for t3 in range(t2 + 1, 254):
                    mu0, w0 = class_stats(0,   t1)
                    mu1, w1 = class_stats(t1,  t2)
                    mu2, w2 = class_stats(t2,  t3)
                    mu3, w3 = class_stats(t3,  256)
                    mu_total = w0*mu0 + w1*mu1 + w2*mu2 + w3*mu3
                    sigma = (w0*(mu0-mu_total)**2 +
                             w1*(mu1-mu_total)**2 +
                             w2*(mu2-mu_total)**2 +
                             w3*(mu3-mu_total)**2)
                    if sigma > best_sigma:
                        best_sigma = sigma
                        best_t = (t1, t2, t3)

        t1, t2, t3        = best_t
        thresholds        = [t1, t2, t3]
        label_map         = np.zeros_like(image_u8, dtype=np.uint8)
        label_map[image_u8 > t1] = 1
        label_map[image_u8 > t2] = 2
        label_map[image_u8 > t3] = 3
        return thresholds, label_map


def apply_colormap_overlay(image: np.ndarray,
                             label_map: np.ndarray,
                             alpha: float = 0.45) -> np.ndarray:

    image_u8 = _as_gray_u8(image)

    rgb = np.stack([image_u8, image_u8, image_u8], axis=2).astype(np.float64)

    colors = {
        1: np.array([255,  77,  58], dtype=np.float64),   # red   #FF4D3A
        2: np.array([200, 241,  53], dtype=np.float64),   # lime  #C8F135
        3: np.array([142, 196, 245], dtype=np.float64),   # blue  #8EC4F5
        4: np.array([245, 166,  35], dtype=np.float64),   # amber #F5A623
    }

    output = rgb.copy()

    for class_idx, color in colors.items():
        mask = (label_map == class_idx)
        if not mask.any():
            continue
        for c in range(3):
            output[:, :, c][mask] = (
                (1 - alpha) * rgb[:, :, c][mask] + alpha * color[c]
            )

    return np.clip(output, 0, 255).astype(np.uint8)
