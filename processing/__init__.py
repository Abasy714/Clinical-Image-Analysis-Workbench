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
from .interpolation import nearest_neighbor_resize
