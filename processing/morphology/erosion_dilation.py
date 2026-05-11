import numpy as np
import logging

_log = logging.getLogger('ciaw')


def _pad_binary(binary: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    rows, cols = binary.shape
    padded = np.zeros((rows + 2 * pad_h, cols + 2 * pad_w), dtype=binary.dtype)
    padded[pad_h:pad_h + rows, pad_w:pad_w + cols] = binary
    return padded


def erode(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    binary = binary.astype(bool)
    se = se.astype(bool)

    _log.debug("erode input sum: %s", int(binary.sum()))

    se_h, se_w = se.shape
    pad_h, pad_w = se_h // 2, se_w // 2
    padded = _pad_binary(binary, pad_h, pad_w)
    rows, cols = binary.shape

    shifts = []
    for dr in range(se_h):
        for dc in range(se_w):
            if se[dr, dc]:
                shift = padded[dr:dr + rows, dc:dc + cols]
                shifts.append(shift)

    stacked = np.stack(shifts, axis=0)
    result = np.all(stacked, axis=0)
    _log.debug("erode output sum: %s", int(result.sum()))
    return result


def dilate(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    binary = binary.astype(bool)
    se = se.astype(bool)

    se_h, se_w = se.shape
    pad_h, pad_w = se_h // 2, se_w // 2
    padded = _pad_binary(binary, pad_h, pad_w)
    rows, cols = binary.shape

    se_reflected = se[::-1, ::-1]
    shifts = []
    for dr in range(se_h):
        for dc in range(se_w):
            if se_reflected[dr, dc]:
                shift = padded[dr:dr + rows, dc:dc + cols]
                shifts.append(shift)

    # dilation = any neighbor must be True
    stacked = np.stack(shifts, axis=0)
    return np.any(stacked, axis=0)
