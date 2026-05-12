import numpy as np
from utils.image_utils import normalize_to_uint8

_COLORS = [
    (231,  76,  60),
    ( 46, 204, 113),
    ( 52, 152, 219),
    (243, 156,  18),
    (155,  89, 182),
    ( 26, 188, 156),
    (230, 126,  34),
    (233,  30,  99),
]


def _label_components(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """BFS connected-component labeling on a 2D bool/0-1 array."""
    H, W = binary.shape
    labels = np.zeros((H, W), dtype=np.int32)
    current_label = 0
    visited = binary == 0

    for r in range(H):
        for c in range(W):
            if visited[r, c]:
                continue
            current_label += 1
            queue = [(r, c)]
            visited[r, c] = True
            labels[r, c] = current_label
            while queue:
                cr, cc = queue.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                        visited[nr, nc] = True
                        labels[nr, nc] = current_label
                        queue.append((nr, nc))

    return labels, current_label


def analyze_blobs(image: np.ndarray) -> tuple[np.ndarray, list]:
    if image.ndim == 3:
        gray = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1]
                + 0.114 * image[:, :, 2]).astype(np.uint8)
    else:
        gray = normalize_to_uint8(image)

    binary = (gray > 127).astype(np.uint8)
    labels, n_labels = _label_components(binary)

    H, W = gray.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)

    stats = []
    for blob_id in range(1, n_labels + 1):
        mask = labels == blob_id
        pixels = np.argwhere(mask)
        if len(pixels) == 0:
            continue

        area = int(len(pixels))
        centroid = (float(pixels[:, 0].mean()), float(pixels[:, 1].mean()))

        min_r = int(pixels[:, 0].min())
        min_c = int(pixels[:, 1].min())
        max_r = int(pixels[:, 0].max())
        max_c = int(pixels[:, 1].max())
        bbox = (min_r, min_c, max_r, max_c)

        eroded = mask.copy()
        eroded[1:, :]  &= mask[:-1, :]
        eroded[:-1, :] &= mask[1:, :]
        eroded[:, 1:]  &= mask[:, :-1]
        eroded[:, :-1] &= mask[:, 1:]
        boundary = mask & ~eroded
        perimeter = int(boundary.sum())
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)

        color = _COLORS[(blob_id - 1) % len(_COLORS)]
        overlay[mask] = color

        stats.append({
            'id':          blob_id,
            'area':        area,
            'centroid':    centroid,
            'bbox':        bbox,
            'perimeter':   perimeter,
            'circularity': float(circularity),
        })

    return overlay, stats
