"""Processing package for the Clinical Image Analysis Workbench."""
from .io import load_image, save_image
from .spatial import (
    convolve2d,
    average_filter,
    gaussian_filter,
    sobel,
    prewitt,
    combined_magnitude,
    median_filter,
)
from .interpolation import nearest_neighbor_resize, bilinear_resize, bilinear_sample, apply_zoom
from .geometric import rotate_image, shear_image
from .histogram import compute_histogram, compute_cdf, local_histogram_equalization
from .segmentation import (
    otsu_threshold,
    otsu_binarize,
    adaptive_threshold,
    multi_level_otsu,
    apply_colormap_overlay,
)
