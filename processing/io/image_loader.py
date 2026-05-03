"""
Handles loading of DICOM, JPEG, and BMP image files into numpy arrays.
Also extracts and returns relevant metadata for display in the metadata panel.
"""

# numpy — image array representation
# Pillow (PIL.Image) — loading JPEG and BMP files
# pydicom — loading DICOM files and extracting DICOM-specific tags
# os, pathlib — file path validation and extension detection
# utils.error_handler — wrap_errors: catch corrupted or unsupported file errors gracefully

# CONSTANTS
# SUPPORTED_EXTENSIONS = ['.dcm', '.jpg', '.jpeg', '.bmp']
# DICOM_TAGS = ['Modality', 'PatientName', 'PatientAge', 'BodyPartExamined']

# FUNCTIONS
# def load_image(filepath: str) -> tuple[np.ndarray, dict]:
#   — detect file type by extension -> dispatch to _load_dicom or _load_standard
#   — return (image_array, metadata_dict)
# def _load_dicom(filepath) -> tuple[np.ndarray, dict]:
#   — use pydicom to read file, extract pixel_array and DICOM tags
# def _load_standard(filepath) -> tuple[np.ndarray, dict]:
#   — use Pillow to open JPEG/BMP, convert to grayscale numpy array, extract basic metadata
# def _extract_metadata(ds) -> dict:
#   — pull DICOM tags safely using getattr with defaults

# --- UTIL USAGE GUIDE ---
# from utils.image_utils import to_grayscale, normalize_to_uint8
# from utils.error_handler import wrap_errors
#
# to_grayscale(array)                 # call on RGB arrays after loading JPEG/BMP — always return 2D
# normalize_to_uint8(array)           # call on DICOM pixel arrays — they may be 12-bit or 16-bit
# @wrap_errors                        # decorate load_image — file may be corrupted or unsupported