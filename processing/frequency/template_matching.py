"""
Template matching utilities.

The GUI uses normalized cross-correlation so the score is stable across local
brightness changes. The older Fourier cross-correlation functions are kept for
compatibility with earlier project code.
"""

import numpy as np
from utils.image_utils import validate_grayscale, normalize_to_uint8, to_grayscale
from utils.error_handler import wrap_errors


@wrap_errors
def fourier_cross_correlate(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Compute Fourier-domain cross-correlation between image and template.

    The template is zero-padded to the image shape, then:
      fft2(image) * conj(fft2(template_padded)) → ifft2 → fftshift → abs

    Returns
    -------
    np.ndarray float64 correlation map, same shape as image.
    Peak location gives the best-match position.
    """
    image = _as_gray_float(image, "image")
    template = _as_gray_float(template, "template")
    H, W = image.shape[:2]
    th, tw = template.shape[:2]
    if th > H or tw > W:
        raise ValueError(
            f"Template size {tw}x{th} is larger than image size {W}x{H}."
        )

    tmpl_padded = np.zeros((H, W), dtype=np.float64)
    tmpl_padded[:th, :tw] = template

    F_image = np.fft.fft2(image)
    F_tmpl  = np.fft.fft2(tmpl_padded)

    cross = F_image * np.conj(F_tmpl)
    corr  = np.abs(np.fft.fftshift(np.fft.ifft2(cross)))

    return corr


@wrap_errors
def find_best_match(correlation_map: np.ndarray) -> tuple:
    """Return (row, col) of the peak in the correlation map."""
    return np.unravel_index(np.argmax(correlation_map), correlation_map.shape)


def _as_gray_float(image: np.ndarray, name: str) -> np.ndarray:
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError(f"No {name} image provided.")
    if image.ndim == 3:
        image = to_grayscale(image)
    validate_grayscale(image)
    gray = normalize_to_uint8(image).astype(np.float64)
    if gray.size == 0:
        raise ValueError(f"The {name} image is empty.")
    return gray


def _window_sum(image: np.ndarray, height: int, width: int) -> np.ndarray:
    padded = np.pad(image, ((1, 0), (1, 0)), mode="constant")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    return (
        integral[height:, width:]
        - integral[:-height, width:]
        - integral[height:, :-width]
        + integral[:-height, :-width]
    )


def normalized_cross_correlation(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Return a valid-position NCC map.

    Output shape is (image_h - template_h + 1, image_w - template_w + 1).
    Each value is in roughly [-1, 1], where 1 is the strongest match.
    """
    image_f = _as_gray_float(image, "source")
    template_f = _as_gray_float(template, "template")

    image_h, image_w = image_f.shape
    tmpl_h, tmpl_w = template_f.shape
    if tmpl_h > image_h or tmpl_w > image_w:
        raise ValueError(
            f"Template size {tmpl_w}x{tmpl_h} is larger than source image "
            f"{image_w}x{image_h}."
        )
    if tmpl_h < 2 or tmpl_w < 2:
        raise ValueError("Template ROI must be at least 2x2 pixels.")

    template_zero_mean = template_f - template_f.mean()
    template_energy = float(np.sum(template_zero_mean ** 2))
    if template_energy <= 1e-10:
        raise ValueError("Template has almost no contrast; choose a more detailed ROI.")

    fft_shape = (image_h + tmpl_h - 1, image_w + tmpl_w - 1)
    kernel = np.flipud(np.fliplr(template_zero_mean))
    numerator_full = np.fft.ifft2(
        np.fft.fft2(image_f, fft_shape) * np.fft.fft2(kernel, fft_shape)
    ).real
    numerator = numerator_full[tmpl_h - 1:image_h, tmpl_w - 1:image_w]

    area = float(tmpl_h * tmpl_w)
    patch_sum = _window_sum(image_f, tmpl_h, tmpl_w)
    patch_sq_sum = _window_sum(image_f ** 2, tmpl_h, tmpl_w)
    patch_energy = patch_sq_sum - (patch_sum ** 2) / area
    patch_energy = np.maximum(patch_energy, 0.0)

    denom = np.sqrt(patch_energy * template_energy)
    ncc = np.zeros_like(numerator, dtype=np.float64)
    valid = denom > 1e-10
    ncc[valid] = numerator[valid] / denom[valid]
    return np.clip(ncc, -1.0, 1.0)


def _rect_iou(a: tuple[int, int, int, int],
              b: tuple[int, int, int, int]) -> float:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    a2r, a2c = ar + ah, ac + aw
    b2r, b2c = br + bh, bc + bw
    inter_h = max(0, min(a2r, b2r) - max(ar, br))
    inter_w = max(0, min(a2c, b2c) - max(ac, bc))
    inter = inter_h * inter_w
    if inter == 0:
        return 0.0
    union = ah * aw + bh * bw - inter
    return inter / max(union, 1)


def find_template_matches(
    score_map: np.ndarray,
    template_shape: tuple[int, int],
    threshold: float = 0.8,
    max_matches: int = 10,
    iou_limit: float = 0.25,
) -> list[dict]:
    """Find non-overlapping template matches from an NCC score map."""
    if score_map is None or score_map.size == 0:
        raise ValueError("Empty template matching score map.")
    tmpl_h, tmpl_w = template_shape[:2]
    threshold = float(np.clip(threshold, -1.0, 1.0))
    max_matches = max(1, int(max_matches))

    candidate_positions = np.argwhere(score_map >= threshold)
    if candidate_positions.size == 0:
        best = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
        candidate_positions = np.array([best])

    candidate_scores = score_map[candidate_positions[:, 0], candidate_positions[:, 1]]
    order = np.argsort(candidate_scores)[::-1]

    selected: list[dict] = []
    selected_rects: list[tuple[int, int, int, int]] = []
    for idx in order:
        row = int(candidate_positions[idx, 0])
        col = int(candidate_positions[idx, 1])
        score = float(candidate_scores[idx])
        rect = (row, col, tmpl_h, tmpl_w)
        if any(_rect_iou(rect, existing) > iou_limit for existing in selected_rects):
            continue
        selected_rects.append(rect)
        selected.append({
            "row": row,
            "col": col,
            "height": int(tmpl_h),
            "width": int(tmpl_w),
            "score": score,
        })
        if len(selected) >= max_matches:
            break
    return selected


def draw_matches(image: np.ndarray, matches: list[dict]) -> np.ndarray:
    """Draw high-contrast rectangles around matches on a grayscale copy."""
    result = normalize_to_uint8(to_grayscale(image) if image.ndim == 3 else image).copy()
    height, width = result.shape[:2]
    for match in matches:
        row = int(match["row"])
        col = int(match["col"])
        h = int(match["height"])
        w = int(match["width"])
        r1 = max(0, min(row, height - 1))
        c1 = max(0, min(col, width - 1))
        r2 = max(r1 + 1, min(row + h, height))
        c2 = max(c1 + 1, min(col + w, width))

        result[r1:r2, c1] = 0
        result[r1:r2, c2 - 1] = 0
        result[r1, c1:c2] = 0
        result[r2 - 1, c1:c2] = 0

        inner_r1 = min(r1 + 1, r2 - 1)
        inner_c1 = min(c1 + 1, c2 - 1)
        inner_r2 = max(inner_r1 + 1, r2 - 1)
        inner_c2 = max(inner_c1 + 1, c2 - 1)
        result[inner_r1:inner_r2, inner_c1] = 255
        result[inner_r1:inner_r2, inner_c2 - 1] = 255
        result[inner_r1, inner_c1:inner_c2] = 255
        result[inner_r2 - 1, inner_c1:inner_c2] = 255
    return result


def match_template(
    image: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.8,
    max_matches: int = 10,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """
    Match a template against a source image.

    Returns (annotated_image, matches, score_map). If no score reaches the
    threshold, the best match is still returned so the GUI can show feedback.
    """
    image_f = _as_gray_float(image, "source")
    template_f = _as_gray_float(template, "template")
    score_map = normalized_cross_correlation(image_f, template_f)
    matches = find_template_matches(
        score_map,
        template_f.shape,
        threshold=threshold,
        max_matches=max_matches,
    )
    annotated = draw_matches(image_f, matches)
    return annotated, matches, score_map
