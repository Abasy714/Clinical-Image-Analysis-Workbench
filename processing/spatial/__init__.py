"""Spatial filtering subpackage: convolution, smoothing, edge detection, median, adaptive median."""
from .convolution import convolve2d
from .smoothing import average_filter, gaussian_filter
from .edge_detection import sobel, prewitt, combined_magnitude
from .median_filter import median_filter
