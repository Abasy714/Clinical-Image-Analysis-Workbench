import numpy as np


def _pad_binary(binary: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    rows, cols = binary.shape
    padded = np.zeros((rows + 2 * pad_h, cols + 2 * pad_w), dtype=binary.dtype)
    padded[pad_h:pad_h + rows, pad_w:pad_w + cols] = binary
    return padded


def erode(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    binary = binary.astype(bool)
    se = se.astype(bool)

    print("erode input unique:", np.unique(binary))
    print("erode input sum:", binary.sum())

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

    print("number of shifts:", len(shifts))
    stacked = np.stack(shifts, axis=0)
    print("stacked shape:", stacked.shape)
    result = np.all(stacked, axis=0)
    print("erode output sum:", result.sum())
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