"""
Adaptive median filter implemented entirely from scratch.
Window grows from S_min up to S_max when the local median is detected
as an impulse. Preserves edges better than a fixed-size median filter.
"""

import numpy as np
from utils.error_handler import wrap_errors
from utils.image_utils import validate_grayscale


@wrap_errors
def adaptive_median_filter(
    image: np.ndarray,
    max_window: int = 7,
) -> np.ndarray:
    """
    Apply an adaptive median filter to a grayscale image from scratch.

    Unlike the standard median filter which uses a fixed window size,
    this filter grows the window when the median itself is detected as
    an impulse noise pixel — preserving more edge detail.

    Parameters
    ----------
    image      : numpy array, grayscale or RGB (RGB converted to grayscale)
    max_window : maximum allowed window size (default: 7, forced odd)

    Returns
    -------
    output : 2D uint8 array (H, W)
    """
    from utils.image_utils import to_grayscale
    if image is None:
        raise ValueError('adaptive_median: image is None')
    if not isinstance(image, np.ndarray):
        raise ValueError(f'adaptive_median: got {type(image)}, need ndarray')
    if image.ndim == 3:
        image = to_grayscale(image)
    validate_grayscale(image)

    if max_window % 2 == 0:
        max_window += 1
    if max_window < 3:
        max_window = 3
    s_min = 3
    s_max = max_window

    image = image.astype(np.float64)
    img_h, img_w = image.shape

    output = np.zeros((img_h, img_w), dtype=np.float64)

    # maximum padding we will ever need — based on s_max
    max_pad = s_max // 2
    padded = np.pad(image, max_pad, mode='edge')  # replicate border pixels to avoid black edges

    for row in range(img_h):
        for col in range(img_w):

            window_size = s_min  # start with the smallest window for each pixel

            while window_size <= s_max:
                pad = window_size // 2

                # extract the neighborhood of current window size around (row, col)
                # offset by max_pad because padded image is larger than original
                r = row + max_pad
                c = col + max_pad
                neighborhood = padded[r - pad: r + pad + 1,
                                      c - pad: c + pad + 1].flatten()

                z_min = neighborhood.min()    # darkest pixel in neighborhood
                z_med = np.median(neighborhood)  # median of neighborhood
                z_max = neighborhood.max()    # brightest pixel in neighborhood
                z_xy  = image[row, col]       # center pixel value

                # Stage A: check if median is itself an impulse
                if z_min < z_med < z_max:
                    # median is clean — go to Stage B
                    # Stage B: check if center pixel is an impulse
                    if z_min < z_xy < z_max:
                        output[row, col] = z_xy   # center pixel is clean, keep it
                    else:
                        output[row, col] = z_med  # center pixel is impulse, replace with median
                    break  # done with this pixel

                else:
                    # median is an impulse — grow the window and try again
                    window_size += 2  # always grow by 2 to keep window odd

                    if window_size > s_max:
                        # window maxed out — best we can do is output the median
                        output[row, col] = z_med
                        break

    return np.clip(output, 0, 255).astype(np.uint8)