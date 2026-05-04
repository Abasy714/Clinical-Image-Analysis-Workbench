"""
Handles exporting the current processed image to disk in JPEG or BMP format.
Converts numpy arrays back to image files using Pillow.
"""

import os
import numpy as np
from PIL import Image

from utils.image_utils import normalize_to_uint8
from utils.error_handler import wrap_errors


@wrap_errors
def save_image(image: np.ndarray, filepath: str) -> bool:
    if image is None:
        raise ValueError("No image to save.")
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    data = normalize_to_uint8(image)    #if the image is not already in uint8 format, it will be normalized to fit within the 0-255 range, which is required for saving as standard image formats like JPEG or BMP
    if data.ndim == 2:      #converts to PIL Image object in grayscale ('L' mode)
        pil_img = Image.fromarray(data, mode='L')
    else:                   #converts to rgb PIL Image object ('RGB' mode) for 3-channel images
        pil_img = Image.fromarray(data, mode='RGB')
    ext = os.path.splitext(filepath)[1].lower()
    save_kwargs = {'quality': 95} if ext in ('.jpg', '.jpeg') else {}  #for JPEG, we set a high quality level to minimize compression artifacts; for BMP, no additional parameters are needed
    pil_img.save(filepath, **save_kwargs)
    return True                     