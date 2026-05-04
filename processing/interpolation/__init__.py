"""Interpolation subpackage: nearest-neighbor and bilinear zoom."""
from .nearest_neighbor import nearest_neighbor_resize
from .bilinear import bilinear_resize, bilinear_sample
from .zoom import apply_zoom
