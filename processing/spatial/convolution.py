"""
2D convolution implemented entirely from scratch using nested loops and numpy.
This is the core engine used by all spatial filters.
"""

import numpy as np
from utils.error_handler import wrap_errors

PADDING_MODES = ['zero', 'reflect', 'replicate']


def _pad_image(image: np.ndarray, pad_h: int, pad_w: int, mode: str) -> np.ndarray:
    """
    Pad a 2D image array on all sides.

    Parameters
    ----------
    image   : 2D float64 array (H, W)
    pad_h   : number of rows to add on each of top and bottom
    pad_w   : number of cols to add on each of left and right
    mode    : 'zero' | 'reflect' | 'replicate'

    Returns
    -------
    padded  : 2D float64 array (H + 2*pad_h, W + 2*pad_w)
    """
    if mode == 'zero':
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    if mode == 'reflect':
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    if mode == 'replicate':
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    raise ValueError(f"Unknown padding mode '{mode}'. Choose from {PADDING_MODES}.")


@wrap_errors
def convolve2d(image: np.ndarray, kernel: np.ndarray,
               padding: str = 'zero') -> np.ndarray:
    """
    Apply a 2D convolution kernel to a grayscale image from scratch.

    The kernel is flipped (true convolution, not correlation) before sliding.
    Works for any odd or even kernel size.

    Parameters
    ----------
    image   : 2D numpy array (H, W), any numeric dtype
    kernel  : 2D numpy array (kH, kW), the convolution kernel
    padding : padding strategy — 'zero' (default), 'replicate', or 'reflect'

    Returns
    -------
    output  : 2D float64 array (H, W), same spatial size as input
    """
    if image.ndim != 2:
        raise ValueError(f"convolve2d expects a 2D image, got shape {image.shape}")
    if kernel.ndim != 2:
        raise ValueError(f"convolve2d expects a 2D kernel, got shape {kernel.shape}")

    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    # true convolution = correlation with flipped kernel
    kernel_flipped = np.flip(kernel)

    img_h, img_w = image.shape
    k_h, k_w = kernel_flipped.shape

    pad_h = k_h // 2
    pad_w = k_w // 2

    padded = _pad_image(image, pad_h, pad_w, padding)

    # Vectorized sliding window — replaces O(H*W) Python loop
    from numpy.lib.stride_tricks import as_strided
    shape = (img_h, img_w, k_h, k_w)
    strides = (
        padded.strides[0],
        padded.strides[1],
        padded.strides[0],
        padded.strides[1],
    )
    patches = as_strided(padded, shape=shape, strides=strides)
    output = np.einsum('ijkl,kl->ij', patches, kernel_flipped)
    return output