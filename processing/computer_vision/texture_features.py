import numpy as np
from utils.image_utils import normalize_to_uint8


def compute_glcm(image: np.ndarray, d: int = 1) -> np.ndarray:
    if image.ndim == 3:
        image = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1]
                 + 0.114 * image[:, :, 2]).astype(np.uint8)
    gray = normalize_to_uint8(image)
    quantized = (gray // 16).astype(np.int32)
    N = 16
    glcm = np.zeros((N, N), dtype=np.float64)

    rows0 = quantized[:, :-d]
    cols0 = quantized[:, d:]
    np.add.at(glcm, (rows0.ravel(), cols0.ravel()), 1)

    rows90 = quantized[:-d, :]
    cols90 = quantized[d:, :]
    np.add.at(glcm, (rows90.ravel(), cols90.ravel()), 1)

    glcm = glcm + glcm.T
    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm


def compute_glcm_features(image: np.ndarray) -> dict:
    glcm = compute_glcm(image)
    N = glcm.shape[0]
    i, j = np.ogrid[0:N, 0:N]

    contrast    = float(np.sum((i - j) ** 2 * glcm))
    homogeneity = float(np.sum(glcm / (1 + np.abs(i - j))))
    asm         = float(np.sum(glcm ** 2))
    energy      = float(np.sqrt(asm))
    entropy     = float(-np.sum(glcm * np.log(glcm + 1e-10)))

    mu_i = float(np.sum(i * glcm))
    mu_j = float(np.sum(j * glcm))
    si   = float(np.sqrt(np.sum((i - mu_i) ** 2 * glcm)))
    sj   = float(np.sqrt(np.sum((j - mu_j) ** 2 * glcm)))
    correlation = float(
        np.sum((i - mu_i) * (j - mu_j) * glcm) / (si * sj + 1e-10)
    )

    return {
        'contrast':     contrast,
        'homogeneity':  homogeneity,
        'energy':       energy,
        'asm':          asm,
        'entropy':      entropy,
        'correlation':  correlation,
    }
