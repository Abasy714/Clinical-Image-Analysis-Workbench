from pathlib import Path
import numpy as np

WEIGHTS_DIR = Path(__file__).parent / 'weights'


def _load_metadata() -> dict:
    from processing.deep_learning.predictor import _load_metadata as _pm
    return _pm()


def overlay_heatmap(original_rgb: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    from processing.deep_learning.grad_cam import overlay_heatmap as _ov
    return _ov(original_rgb, heatmap, alpha)


def compute_scorecam(image: np.ndarray, roi: tuple | None = None,
                     alpha: float = 0.4) -> np.ndarray:
    import tensorflow as tf
    from PIL import Image as PILImage
    from utils.image_utils import normalize_to_uint8
    from processing.deep_learning.predictor import load_model

    meta  = _load_metadata()
    sz    = int(meta.get('input_size', [224, 224])[0])
    model = load_model()

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
    batch = np.expand_dims(norm, 0)

    last_conv = meta.get('last_conv_layer', None)
    if last_conv is None:
        heatmap = np.ones((sz, sz), dtype=np.float32)
        return normalize_to_uint8(overlay_heatmap(orig, heatmap, alpha))

    try:
        conv_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output, model.output],
        )
    except Exception:
        heatmap = np.ones((sz, sz), dtype=np.float32)
        return normalize_to_uint8(overlay_heatmap(orig, heatmap, alpha))

    activations, preds = conv_model.predict(batch, verbose=0)
    class_idx  = int(np.argmax(preds[0]))
    n_channels = activations.shape[-1]

    scores = np.zeros(n_channels, dtype=np.float32)
    for ch in range(n_channels):
        act_ch = activations[0, :, :, ch]
        mn, mx = act_ch.min(), act_ch.max()
        norm_ch = (act_ch - mn) / (mx - mn + 1e-8)
        norm_rs = np.array(
            PILImage.fromarray((norm_ch * 255).astype(np.uint8)).resize(
                (sz, sz), PILImage.BILINEAR)
        ).astype(np.float32) / 255.0
        masked = batch * norm_rs[np.newaxis, :, :, np.newaxis]
        scores[ch] = model.predict(masked, verbose=0)[0][class_idx]

    weights = np.maximum(scores, 0)
    heatmap = np.zeros((activations.shape[1], activations.shape[2]), dtype=np.float32)
    for ch in range(n_channels):
        heatmap += weights[ch] * activations[0, :, :, ch]

    heatmap = np.maximum(heatmap, 0)
    mx = heatmap.max()
    if mx > 0:
        heatmap /= mx

    result = overlay_heatmap(orig, heatmap, alpha)
    return normalize_to_uint8(result)
