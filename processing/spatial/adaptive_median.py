"""
Adaptive median filter implemented entirely from scratch.
Window grows from S_min up to S_max when the local median is detected
as an impulse. Preserves edges better than a fixed-size median filter.
"""

import numpy as np
from utils.error_handler import wrap_errors
from utils.image_utils import validate_grayscale, normalize_to_uint8


@wrap_errors
def adaptive_median_filter(
    image: np.ndarray,
    s_min: int = 3,
    s_max: int = 7,
) -> np.ndarray:
    """
    Apply an adaptive median filter to a grayscale image from scratch.

    Unlike the standard median filter which uses a fixed window size,
    this filter grows the window when the median itself is detected as
    an impulse noise pixel — preserving more edge detail.

    Two-stage algorithm per pixel:

        Stage A — check if median is an impulse:
            z_min = min of neighborhood
            z_med = median of neighborhood
            z_max = max of neighborhood
            if z_min < z_med < z_max → median is clean → go to Stage B
            else                     → window too small, grow and repeat A
            if window exceeds s_max  → output z_med and stop

        Stage B — check if center pixel is an impulse:
            z_xy  = center pixel value
            if z_min < z_xy < z_max → center pixel is clean → output z_xy
            else                    → center pixel is impulse → output z_med

    Parameters
    ----------
    image : 2D numpy array (H, W), grayscale
    s_min : starting (minimum) window size — must be odd (default: 3)
    s_max : maximum allowed window size — must be odd (default: 7)

    Returns
    -------
    output : 2D uint8 array (H, W), same size as input
    """
    validate_grayscale(image)

    if s_min < 3 or s_min % 2 == 0:
        raise ValueError(f"s_min must be an odd number >= 3, got {s_min}")
    if s_max < s_min or s_max % 2 == 0:
        raise ValueError(f"s_max must be an odd number >= s_min, got {s_max}")

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

    return normalize_to_uint8(output)