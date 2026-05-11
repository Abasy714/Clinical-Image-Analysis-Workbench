import numpy as np
from utils.image_utils import normalize_to_uint8


def trace_contour(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1]
                + 0.114 * image[:, :, 2]).astype(np.uint8)
    else:
        gray = normalize_to_uint8(image)

    binary = gray > 127

    eroded = binary.copy()
    eroded[1:, :]  &= binary[:-1, :]
    eroded[:-1, :] &= binary[1:, :]
    eroded[:, 1:]  &= binary[:, :-1]
    eroded[:, :-1] &= binary[:, 1:]

    contour = binary & ~eroded

    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[contour] = [200, 241, 53]
    return rgb.astype(np.uint8)
