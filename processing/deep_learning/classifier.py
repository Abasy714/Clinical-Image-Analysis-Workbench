"""Thin wrapper used by gui/ai_panel.py for explicit model loading and ROI prediction."""
import numpy as np


def load_model(path: str | None = None):
    from processing.deep_learning.predictor import load_model as _lm
    return _lm(path)


def predict_roi(image: np.ndarray, roi: tuple, input_size: tuple = (224, 224)):
    x, y, w, h = roi
    roi_crop = image[y:y+h, x:x+w]
    from processing.deep_learning.predictor import predict
    result = predict(image, roi_array=roi_crop)
    return result
