import numpy as np
from utils.image_utils import validate_grayscale, normalize_to_uint8
from utils.error_handler import wrap_errors
from processing.histogram.histogram_utils import compute_histogram, compute_cdf


def _compute_tile_lut(block: np.ndarray, clip_limit: float = 3.0, num_levels: int = 256) -> np.ndarray:
    hist = compute_histogram(block, bins=num_levels)

    avg_count = block.size / num_levels #for clip limiting, divides total no of pixels in tile with levels 
    max_count = max(1, int(clip_limit * avg_count)) #no bin can be more than 3 times the average (clipping)

    excess = int(np.sum(np.maximum(hist - max_count, 0)))
    hist = np.minimum(hist, max_count)
    redistribute = excess // num_levels
    remainder = excess % num_levels
    hist = hist + redistribute
    hist[:remainder] += 1

    cdf = compute_cdf(hist)
    lut = np.clip(np.round(cdf * (num_levels - 1)).astype(np.int32), 0, num_levels - 1).astype(np.uint8)
    return lut


def _interpolate_luts(
    image: np.ndarray,
    luts: np.ndarray,
    tile_centers_y: np.ndarray,
    tile_centers_x: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    n_tiles_y = luts.shape[0]
    n_tiles_x = luts.shape[1]

    # build coordinate grids for every pixel
    rows = np.arange(height, dtype=np.float64)
    cols = np.arange(width,  dtype=np.float64)

    # for each pixel row, find which two tile centers it sits between
    # searchsorted returns the index to the right, so subtract 1 for left
    ty1 = np.searchsorted(tile_centers_y, rows, side='right') - 1
    ty1 = np.clip(ty1, 0, n_tiles_y - 2)
    ty2 = ty1 + 1

    tx1 = np.searchsorted(tile_centers_x, cols, side='right') - 1
    tx1 = np.clip(tx1, 0, n_tiles_x - 2)
    tx2 = tx1 + 1

    # fractional distances — shape (height,) and (width,)
    span_y = tile_centers_y[ty2] - tile_centers_y[ty1]
    span_x = tile_centers_x[tx2] - tile_centers_x[tx1]

    span_y = np.where(span_y == 0, 1.0, span_y)
    span_x = np.where(span_x == 0, 1.0, span_x)

    dy = np.clip((rows - tile_centers_y[ty1]) / span_y, 0.0, 1.0)  # shape (height,)
    dx = np.clip((cols - tile_centers_x[tx1]) / span_x, 0.0, 1.0)  # shape (width,)

    # expand to 2D grids — shape (height, width)
    dy2d = dy[:, np.newaxis]  # column vector
    dx2d = dx[np.newaxis, :]  # row vector

    # pixel values — shape (height, width)
    pv = image.astype(np.int32)

    # look up each pixel value in the 4 surrounding tile LUTs
    # luts shape is (n_tiles_y, n_tiles_x, 256)
    ty1_2d = ty1[:, np.newaxis]  # shape (height, 1)
    ty2_2d = ty2[:, np.newaxis]
    tx1_2d = tx1[np.newaxis, :]  # shape (1, width)
    tx2_2d = tx2[np.newaxis, :]

    v11 = luts[ty1_2d, tx1_2d, pv].astype(np.float64)  # top-left
    v12 = luts[ty1_2d, tx2_2d, pv].astype(np.float64)  # top-right
    v21 = luts[ty2_2d, tx1_2d, pv].astype(np.float64)  # bottom-left
    v22 = luts[ty2_2d, tx2_2d, pv].astype(np.float64)  # bottom-right

    # bilinear blend — same formula as before, now applied to entire image at once
    output = (
        v11 * (1.0 - dx2d) * (1.0 - dy2d)
        + v12 *       dx2d  * (1.0 - dy2d)
        + v21 * (1.0 - dx2d) *       dy2d
        + v22 *       dx2d  *        dy2d
    )

    return output


@wrap_errors
def local_histogram_equalization(image: np.ndarray, block_size: int, clip_limit: float = 3.0) -> np.ndarray:
    validate_grayscale(image) #ensures img is 2d

#validations for block size
    if block_size < 2:
        raise ValueError(f"block_size must be at least 2, got {block_size}.")
    if block_size > image.shape[0] or block_size > image.shape[1]:
        raise ValueError(f"block_size ({block_size}) exceeds image dimensions {image.shape}.")
    

    image = normalize_to_uint8(image)
    height, width = image.shape #gets dimensions
    num_levels = 256

    
    n_tiles_y = max(1, height // block_size)#gets number of tiles that fit according to block size
    n_tiles_x = max(1, width // block_size)

    
    luts = np.zeros((n_tiles_y, n_tiles_x, num_levels), dtype=np.uint8) #gets lut for tile
    for ty in range(n_tiles_y): #loops on tiles 
        for tx in range(n_tiles_x):
            row_start = ty * block_size
            col_start = tx * block_size
            block = image[row_start : row_start + block_size, col_start : col_start + block_size]
            luts[ty, tx] = _compute_tile_lut(block, clip_limit=clip_limit) #Computes the clipped equalized LUT for this tile and stores it.

    tile_centers_y = (np.arange(n_tiles_y, dtype=np.float64) + 0.5) * block_size
    tile_centers_x = (np.arange(n_tiles_x, dtype=np.float64) + 0.5) * block_size

    output = _interpolate_luts(image, luts, tile_centers_y, tile_centers_x)

    return normalize_to_uint8(output.astype(np.uint8))