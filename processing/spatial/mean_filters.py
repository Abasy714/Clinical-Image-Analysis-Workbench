
import numpy as np
from utils.error_handler import wrap_errors
from utils.image_utils import validate_grayscale, normalize_to_uint8


@wrap_errors
def arithmetic_mean_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply an arithmetic mean filter to a grayscale image from scratch.

    Each output pixel is the simple average of all pixels in the
    kernel_size x kernel_size neighborhood:

        g(x, y) = (1 / mn) * sum of f(s, t) over the neighborhood

    Best for: reducing Gaussian noise while keeping overall brightness.
    Drawback: blurs edges.

    Parameters
    ----------
    image       : 2D numpy array (H, W), grayscale
    kernel_size : side length of the square window (must be odd, e.g. 3, 5, 7)

    Returns
    -------
    output : 2D uint8 array (H, W)
    """
    validate_grayscale(image)

    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    if kernel_size == 1:
        return normalize_to_uint8(image)

    image = image.astype(np.float64)
    img_h, img_w = image.shape
    pad = kernel_size // 2

    padded = np.pad(image, pad, mode='edge')  # replicate border pixels to avoid black edges

    # Vectorized sliding window — replaces O(H*W) Python loop
    from numpy.lib.stride_tricks import as_strided
    shape = (img_h, img_w, kernel_size, kernel_size)
    strides = (
        padded.strides[0],  # move to next row of center pixels
        padded.strides[1],  # move to next col of center pixels
        padded.strides[0],  # move down inside the patch
        padded.strides[1],  # move right inside the patch
    )
    patches = as_strided(padded, shape=shape, strides=strides)
    patches = patches.reshape(img_h, img_w, -1)  # flatten each window → (H, W, k*k)

    output = patches.mean(axis=2)  # mean over all pixels in the neighborhood
    return normalize_to_uint8(output)


@wrap_errors
def harmonic_mean_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply a harmonic mean filter to a grayscale image from scratch.

    Formula:
        g(x, y) = mn / sum(1 / f(s, t))

    A small epsilon is added before inverting to avoid division by zero
    on black (zero) pixels.

    Best for: removing salt noise (bright impulses).
    Fails for: pepper noise (dark impulses / zeros).

    Parameters
    ----------
    image       : 2D numpy array (H, W), grayscale
    kernel_size : side length of the square window (must be odd)

    Returns
    -------
    output : 2D uint8 array (H, W)
    """
    validate_grayscale(image)

    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    if kernel_size == 1:
        return normalize_to_uint8(image)

    image = image.astype(np.float64)
    img_h, img_w = image.shape
    pad = kernel_size // 2

    padded = np.pad(image, pad, mode='edge')  # replicate border pixels to avoid black edges

    # Vectorized sliding window — replaces O(H*W) Python loop
    from numpy.lib.stride_tricks import as_strided
    shape = (img_h, img_w, kernel_size, kernel_size)
    strides = (
        padded.strides[0],  # move to next row of center pixels
        padded.strides[1],  # move to next col of center pixels
        padded.strides[0],  # move down inside the patch
        padded.strides[1],  # move right inside the patch
    )
    patches = as_strided(padded, shape=shape, strides=strides)
    patches = patches.reshape(img_h, img_w, -1)  # flatten each window → (H, W, k*k)

    n = patches.shape[2]                    # number of pixels in window = k*k
    eps = 1e-6                              # avoid division by zero on black pixels

    # harmonic mean = n / sum(1/x) -- el harmonic mean bey7seb el reciprocal of each pixel, yakhod el sum, we yedrab fi n
    output = n / np.sum(1.0 / (patches + eps), axis=2)
    return normalize_to_uint8(output)


@wrap_errors
def contraharmonic_mean_filter(
    image: np.ndarray, kernel_size: int, Q: float
) -> np.ndarray:
    """
    Apply a contra-harmonic mean filter to a grayscale image from scratch.

    Formula:
        g(x, y) = sum(f(s,t)^(Q+1)) / sum(f(s,t)^Q)

    The order Q controls the filter behavior:
        Q > 0  → eliminates pepper noise (dark impulses)
        Q < 0  → eliminates salt noise (bright impulses)
        Q = 0  → reduces to arithmetic mean
        Q = -1 → reduces to harmonic mean

    Parameters
    ----------
    image       : 2D numpy array (H, W), grayscale
    kernel_size : side length of the square window (must be odd)
    Q           : order of the filter (float, can be negative)

    Returns
    -------
    output : 2D uint8 array (H, W)
    """
    validate_grayscale(image)

    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    if kernel_size == 1:
        return normalize_to_uint8(image)

    image = image.astype(np.float64)
    img_h, img_w = image.shape
    pad = kernel_size // 2

    padded = np.pad(image, pad, mode='edge')  # replicate border pixels to avoid black edges

    # Vectorized sliding window — replaces O(H*W) Python loop
    from numpy.lib.stride_tricks import as_strided
    shape = (img_h, img_w, kernel_size, kernel_size)
    strides = (
        padded.strides[0],  # move to next row of center pixels
        padded.strides[1],  # move to next col of center pixels
        padded.strides[0],  # move down inside the patch
        padded.strides[1],  # move right inside the patch
    )
    patches = as_strided(padded, shape=shape, strides=strides)
    patches = patches.reshape(img_h, img_w, -1)  # flatten each window → (H, W, k*k)

    eps = 1e-6  # avoid 0^negative = inf when Q is negative

    safe_patches = patches + eps  # shift all values slightly above zero

    # numerator = sum of p^(Q+1), denominator = sum of p^Q
    numerator   = np.sum(np.power(safe_patches, Q + 1.0), axis=2)
    denominator = np.sum(np.power(safe_patches, Q),       axis=2)

    # guard against near-zero denominators to avoid NaN
    denominator = np.where(np.abs(denominator) < eps, eps, denominator)

    output = numerator / denominator
    return normalize_to_uint8(output)