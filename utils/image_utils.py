# STATUS: IMPLEMENTED
"""
Shared image utility functions used across all processing and GUI modules.
Handles dtype normalization, channel validation, numpy-to-Qt conversion, and binarization.
"""

import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        mn, mx = float(image.min()), float(image.max())
        if mx == mn:
            return np.zeros_like(image, dtype=np.uint8)
        if mn >= 0.0 and mx <= 1.0:
            return (image * 255).astype(np.uint8)
        return ((image - mn) / (mx - mn) * 255).astype(np.uint8)
    if image.dtype == np.uint16:
        return (image.astype(np.float64) / 65535.0 * 255).astype(np.uint8)
    if image.dtype == np.uint32:
        return (image.astype(np.float64) / 4294967295.0 * 255).astype(np.uint8)
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        if info.max > 0:
            return (image.astype(np.float64) / info.max * 255).astype(np.uint8)
    mn, mx = float(image.min()), float(image.max())
    if mx == mn:
        return np.zeros_like(image, dtype=np.uint8)
    return ((image.astype(np.float64) - mn) / (mx - mn) * 255).astype(np.uint8)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] >= 3:
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
        gray = image[:, :, :3].astype(np.float64) @ weights
        return gray.astype(image.dtype)
    return image


def validate_grayscale(image: np.ndarray) -> bool:
    if image.ndim == 2:
        return True
    raise ValueError(f"Expected grayscale (2D) image, got shape {image.shape}")


def to_qpixmap(image: np.ndarray) -> QPixmap:
    data = normalize_to_uint8(image)
    if data.ndim == 2:
        h, w = data.shape
        data = np.ascontiguousarray(data)
        raw = bytes(data)
        qimg = QImage(raw, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        if data.shape[2] == 4:
            data = data[:, :, :3]
        data = np.ascontiguousarray(data)
        h, w = data.shape[:2]
        raw = bytes(data)
        qimg = QImage(raw, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def binarize(image: np.ndarray, threshold: int) -> np.ndarray:
    validate_grayscale(image)
    gray = normalize_to_uint8(image)   
    return (gray > threshold).astype(np.uint8) #changed by sohaila
