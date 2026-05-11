import numpy as np
import logging

_log = logging.getLogger('ciaw')


def _pad_binary(binary: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    rows, cols = binary.shape #original image dimensions
    padded = np.zeros((rows + 2 * pad_h, cols + 2 * pad_w), dtype=binary.dtype) #bigger empty array
    padded[pad_h:pad_h + rows, pad_w:pad_w + cols] = binary #copies original image into center of padded array
    return padded


def erode(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    binary = binary.astype(bool) #converts to t/f
    se = se.astype(bool)#converts to t/f
    se_h, se_w = se.shape #shape of se
    pad_h, pad_w = se_h // 2, se_w // 2 #calculates padding needed
    padded = _pad_binary(binary, pad_h, pad_w) #pads image
    rows, cols = binary.shape #saves original shape

    _log.debug("erode input sum: %s", int(binary.sum()))


    shifts = []
    for dr in range(se_h):
        for dc in range(se_w):
            if se[dr, dc]: #for all true positions
                shift = padded[dr:dr + rows, dc:dc + cols] #slice image at dr,dc then offset by dr,dc
                shifts.append(shift)
    stacked = np.stack(shifts, axis=0) #stack all shifts in 3d array, each layer is img as seen from se neighbor position
    result = np.all(stacked, axis=0)#erosion rule: if all true,pixel stays true

    _log.debug("erode output sum: %s", int(result.sum()))
    return result


def dilate(binary: np.ndarray, se: np.ndarray) -> np.ndarray:
    binary = binary.astype(bool)
    se = se.astype(bool)

    se_h, se_w = se.shape
    pad_h, pad_w = se_h // 2, se_w // 2
    padded = _pad_binary(binary, pad_h, pad_w)
    rows, cols = binary.shape

    se_reflected = se[::-1, ::-1] #mathematically required for dilation
    shifts = []
    for dr in range(se_h):
        for dc in range(se_w):
            if se_reflected[dr, dc]:
                shift = padded[dr:dr + rows, dc:dc + cols]
                shifts.append(shift)
    stacked = np.stack(shifts, axis=0)
    return np.any(stacked, axis=0) #if any neighbor is true, pixel stays true 
