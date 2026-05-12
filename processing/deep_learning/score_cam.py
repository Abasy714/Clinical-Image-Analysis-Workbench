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
    from PIL import Image as PILImage
    from utils.image_utils import normalize_to_uint8
    from processing.deep_learning.grad_cam import _load_grad_model

    meta = _load_metadata()
    _sz  = meta.get('input_size', [224, 224])
    sz   = int(_sz[0] if isinstance(_sz, (list, tuple)) else _sz)

    if roi is not None:
        x, y, w, h = roi
        target = image[y:y + h, x:x + w]
    else:
        target = image

    if target.ndim == 2:
        target = np.stack([target] * 3, axis=-1)
    orig = np.array(
        PILImage.fromarray(target.astype(np.uint8)).convert('RGB').resize((sz, sz))
    )
    norm  = orig.astype(np.float32) / 255.0
    batch = np.expand_dims(norm, 0)

    grad_model = _load_grad_model()

    # Initial forward pass — get conv activations and class prediction
    activations, preds = grad_model(batch, training=False)
    activations = activations.numpy()   # (1, h, w, C)
    class_idx   = int(np.argmax(preds[0]))
    n_channels  = activations.shape[-1]

    # Score each channel: mask input by normalised channel map, record class score
    scores = np.zeros(n_channels, dtype=np.float32)
    for ch in range(n_channels):
        act_ch = activations[0, :, :, ch]
        mn, mx = act_ch.min(), act_ch.max()
        norm_ch = (act_ch - mn) / (mx - mn + 1e-8)
        mask = np.array(
            PILImage.fromarray((norm_ch * 255).astype(np.uint8))
            .resize((sz, sz), PILImage.BILINEAR)
        ).astype(np.float32) / 255.0
        masked = batch * mask[np.newaxis, :, :, np.newaxis]
        _, score_out = grad_model(masked, training=False)
        scores[ch] = float(score_out[0, class_idx])

    weights = np.maximum(scores, 0)
    heatmap = (activations[0] * weights[np.newaxis, np.newaxis, :]).sum(axis=-1)
    heatmap = np.maximum(heatmap, 0)
    hmax    = heatmap.max()
    if hmax > 0:
        heatmap /= hmax

    return normalize_to_uint8(overlay_heatmap(orig, heatmap, alpha))
