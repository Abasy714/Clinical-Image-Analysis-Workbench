from pathlib import Path
import numpy as np

WEIGHTS_DIR = Path(__file__).parent / 'weights'
_GRADCAM_PATH = WEIGHTS_DIR / 'gradcam_model.h5'

_grad_model = None


def _load_grad_model():
    global _grad_model
    if _grad_model is not None:
        return _grad_model
    try:
        from tensorflow.keras.models import load_model
        _grad_model = load_model(str(_GRADCAM_PATH))
    except Exception as exc:
        raise RuntimeError(f"Failed to load GradCAM model: {exc}") from exc
    return _grad_model


def _load_metadata() -> dict:
    from processing.deep_learning.predictor import _load_metadata as _pm
    return _pm()


def overlay_heatmap(original_rgb: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    from PIL import Image as PILImage
    h, w = original_rgb.shape[:2]
    if original_rgb.ndim == 2:
        original_rgb = np.stack([original_rgb] * 3, axis=-1)
    hm_u8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    hm_rs = np.array(
        PILImage.fromarray(hm_u8).resize((w, h), PILImage.BILINEAR)
    ).astype(np.float32) / 255.0
    try:
        import matplotlib.pyplot as plt
        colored = (plt.get_cmap('jet')(hm_rs)[:, :, :3] * 255).astype(np.uint8)
    except ImportError:
        heat = (hm_rs * 255).astype(np.uint8)
        colored = np.stack([heat, np.zeros_like(heat), np.zeros_like(heat)], axis=-1)
    return np.clip(
        (1 - alpha) * original_rgb.astype(np.float32)
        + alpha * colored.astype(np.float32), 0, 255
    ).astype(np.uint8)


def compute_gradcam(image: np.ndarray, roi: tuple | None = None,
                    class_idx: int | None = None, alpha: float = 0.4) -> np.ndarray:
    import tensorflow as tf
    from PIL import Image as PILImage
    from utils.image_utils import normalize_to_uint8

    grad_model = _load_grad_model()
    meta = _load_metadata()
    sz   = int(meta.get('input_size', [224, 224])[0])

    if roi is not None:
        x, y, w, h = roi
        target = image[y:y+h, x:x+w]
    else:
        target = image

    if target.ndim == 2:
        target = np.stack([target] * 3, axis=-1)
    orig = np.array(
        PILImage.fromarray(target.astype(np.uint8)).convert('RGB').resize((sz, sz))
    )
    norm  = orig.astype(np.float32) / 255.0
    batch = tf.cast(np.expand_dims(norm, 0), tf.float32)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(batch, training=False)
        tape.watch(conv_out)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        score = preds[:, class_idx]

    grads   = tape.gradient(score, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = (heatmap / (tf.reduce_max(heatmap) + 1e-8)).numpy()

    result = overlay_heatmap(orig, heatmap, alpha)
    return normalize_to_uint8(result)
