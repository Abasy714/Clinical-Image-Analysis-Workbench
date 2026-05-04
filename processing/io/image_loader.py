"""
Handles loading of DICOM, JPEG, and BMP image files into numpy arrays.
Also extracts and returns relevant metadata for display in the metadata panel.
"""

"""
Handles loading of DICOM, JPEG, and BMP image files into numpy arrays.
Also extracts and returns relevant metadata for display in the metadata panel.
"""

import os
import numpy as np
from PIL import Image

from utils.image_utils import to_grayscale, normalize_to_uint8
from utils.error_handler import wrap_errors

SUPPORTED_EXTENSIONS = ['.dcm', '.jpg', '.jpeg', '.bmp']


@wrap_errors
def load_image(filepath: str) -> tuple:
    ext = os.path.splitext(filepath)[1].lower() #extracts the file extension and converts to lowercase for case-insensitive comparison
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {ext}")
    if ext == '.dcm':
        return _load_dicom(filepath)
    return _load_standard(filepath)


def _load_dicom(filepath: str) -> tuple:
    import pydicom
    ds = pydicom.dcmread(filepath)       #reads the binary DICOM file into a dataset object ds
    array = ds.pixel_array.astype(np.float64)
    array = normalize_to_uint8(array)    #pixel data is extracted as a numpy array, converted to float64 for processing, and normalized to uint8
    if array.ndim == 3:
        array = to_grayscale(array)      #handles the case of an RGB DICOM
    metadata = _extract_dicom_metadata(ds, filepath)    #extracs metadata from dicom dataser and converts to dictionary
    return array, metadata


def _load_standard(filepath: str) -> tuple:
    pil_img = Image.open(filepath)
    array = np.array(pil_img)
    if array.ndim == 3:
        array = to_grayscale(array)
    array = normalize_to_uint8(array)   #bit depth normalization for standard images (JPEG/BMP)
    metadata = {
        'filename':          os.path.basename(filepath),
        'width':             array.shape[1],
        'height':            array.shape[0],
        'BitsStored':        8,
        #dicom specific fields set to empty strings for non-DICOM images
        'Modality':          '',        
        'PatientName':       '',
        'PatientAge':        '',
        'BodyPartExamined':  '',
        'KVP':               '',
    }
    return array, metadata


def _extract_dicom_metadata(ds, filepath: str) -> dict:
    #helper to get metadata fields from DICOM dataset with safe defaults if smth is missing
    def safe(tag, default=''):
        val = getattr(ds, tag, default)
        return str(val).strip() if val is not None else default     

    rows    = getattr(ds, 'Rows',    0)
    cols    = getattr(ds, 'Columns', 0)
    bits    = getattr(ds, 'BitsStored', 8)

    return {
        'filename':         os.path.basename(filepath),
        'width':            cols,
        'height':           rows,
        'BitsStored':       bits,
        'Modality':         safe('Modality'),
        'PatientName':      safe('PatientName'),
        'PatientAge':       safe('PatientAge'),
        'BodyPartExamined': safe('BodyPartExamined'),
        'KVP':              safe('KVP'),
    }