"""Spatial filtering subpackage: convolution, smoothing, edge detection, median, adaptive median."""
from .convolution import convolve2d
from .smoothing import average_filter, gaussian_filter
from .edge_detection import sobel, prewitt, combined_magnitude
from .median_filter import median_filter
from .mean_filters import  arithmetic_mean_filter, harmonic_mean_filter, contraharmonic_mean_filter

from .order_statistic_filters import  min_filter, max_filter, midpoint_filter
from .adaptive_median import adaptive_median_filter