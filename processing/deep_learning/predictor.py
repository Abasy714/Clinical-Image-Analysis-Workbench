from pathlib import Path
import json
import numpy as np

WEIGHTS_DIR = Path(__file__).parent / 'weights'
_CLASSIFIER_PATH = WEIGHTS_DIR / 'brain_tumor_classifier.h5'
_METADATA_PATHS  = [WEIGHTS_DIR / 'metadata.json',
                    WEIGHTS_DIR / 'model_metadata.json']

_model = None
_meta  = None
_last_prediction = None


def is_available() -> bool:
    return _CLASSIFIER_PATH.exists()


def _load_metadata() -> dict:
    global _meta
    if _meta is not None:
        return _meta
    for path in _METADATA_PATHS:
        if path.exists():
            with open(path) as f:
                _meta = json.load(f)
            return _meta
    _meta = {
        'input_size':     [224, 224],
        'class_names':    ['No Tumor', 'Meningioma', 'Glioma', 'Pituitary'],
        'last_conv_layer': 'conv_last',
        'test_accuracy':  0.0,
    }
    return _meta


def load_model(path: str | None = None) -> object:
    global _model
    if _model is not None:
        return _model
    try:
        from tensorflow.keras.models import load_model as _keras_load
        target = path or str(_CLASSIFIER_PATH)
        _model = _keras_load(target)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc
    return _model


def _preprocess(roi_array: np.ndarray, sz: int) -> np.ndarray:
    from PIL import Image
    if roi_array.ndim == 2:
        roi_array = np.stack([roi_array] * 3, axis=-1)
    img = Image.fromarray(roi_array.astype(np.uint8)).convert('RGB').resize((sz, sz))
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, 0)


def predict(image: np.ndarray, roi_array: np.ndarray | None = None) -> dict:
    global _last_prediction

    if not is_available():
        raise RuntimeError("Model weights not found")

    meta   = _load_metadata()
    model  = load_model()
    sz     = int(meta.get('input_size', [224, 224])[0])
    names  = meta.get('class_names', ['No Tumor', 'Meningioma', 'Glioma', 'Pituitary'])
    target = roi_array if roi_array is not None else image
    batch  = _preprocess(target, sz)
    scores = model.predict(batch, verbose=0)[0]
    idx    = int(np.argmax(scores))

    _last_prediction = {
        'label':       names[idx] if idx < len(names) else str(idx),
        'confidence':  float(scores[idx]),
        'all_scores':  dict(zip(names, scores.tolist())),
        'class_index': idx,
        'image':       target,
    }
    return _last_prediction


def predict_worker(image: np.ndarray, roi_array: np.ndarray | None = None) -> np.ndarray:
    result = predict(image, roi_array)
    from utils.image_utils import normalize_to_uint8
    target = result.get('image', image)
    return normalize_to_uint8(target)


def get_last_prediction() -> dict | None:
    return _last_prediction
