# Clinical Image Analysis Workbench

A desktop medical image processing workbench built with PyQt6. Implements spatial filtering, frequency-domain noise removal, local histogram equalization, and morphological operations — all from scratch.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyQt6](https://img.shields.io/badge/PyQt6-GUI-green?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DL-orange?style=flat-square&logo=tensorflow)

---

## Overview

The Clinical Image Analysis Workbench is an academic desktop application designed for exploring and applying classical and modern image processing techniques on medical images (DICOM, JPEG, BMP). Every algorithm in the spatial and morphological domains is implemented from scratch — no OpenCV or scikit-image wrappers are used. The project is structured as a pipeline-based workbench where operations are composed sequentially and can be undone or reset at any point.

---

## Features

### Phase 1 — Core Image Processing

- **Image I/O**: Load DICOM (`.dcm`), JPEG, and BMP files; extract and display DICOM metadata tags (Modality, PatientName, PatientAge, BodyPartExamined)
- **Zoom & Interpolation**: Interactive zoom using nearest-neighbor and bilinear interpolation, both implemented from scratch
- **Spatial Filtering**:
  - Average (box) filter with configurable kernel size
  - Gaussian filter with user-defined sigma, kernel built from scratch
  - Sobel and Prewitt edge detection — horizontal, vertical, and combined magnitude
  - Non-linear median filter from scratch
- **Local Histogram Equalization**: Block-based equalization with configurable block size; ROI histogram display
- **Geometric Transforms**: Image rotation and shearing via inverse mapping with bilinear interpolation
- **Pipeline Management**: Sequential op stack with per-step undo and full reset

### Phase 2 — Frequency Domain & Morphology

- **FFT Spectrum Viewer**: Log-scaled magnitude spectrum display using `numpy.fft2` + `fftshift`
- **Interactive Notch Filtering**: Click on noise spikes in the spectrum to place ideal, Butterworth, or Gaussian notch filters; conjugate pair placed automatically
- **Fourier Template Matching**: Cross-correlation via FFT for template localization
- **Noise Injection**: Gaussian and uniform synthetic noise injection from scratch
- **ROI Statistics**: Mean, variance, and local histogram for any drawn rectangular ROI
- **Morphological Operations**:
  - Binary erosion and dilation from scratch
  - Opening and closing (compound ops)
  - Boundary extraction via eroded subtraction
  - Square and cross structuring element generators

### Bonus / Placeholders

- **Segmentation** (`processing/segmentation/`): Reserved for future segmentation model integration
- **Computer Vision** (`processing/computer_vision/`): Reserved for future CV model integration
- **Deep Learning** (`processing/deep_learning/`): Reserved for future DL inference pipelines (TensorFlow/Keras); model weights stored in `processing/deep_learning/weights/`

---

## Project Structure

```
ClinicalWorkbench/
├── main.py                          # App entry point
├── requirements.txt
├── report/                          # Report output directory
│
├── gui/                             # PyQt6 panels and widgets
│   ├── main_window.py               # Tabbed window, pipeline state manager
│   ├── image_viewer.py              # Display widget, zoom, ROI drawing
│   ├── metadata_panel.py            # DICOM/image metadata display
│   ├── filter_panel.py              # Spatial filter controls
│   ├── histogram_panel.py           # Local histogram equalization
│   ├── pipeline_panel.py            # Undo/reset, op stack UI
│   ├── fourier_panel.py             # Spectrum + notch filter UI
│   ├── noise_panel.py               # Noise injection + ROI stats
│   └── morphology_panel.py          # Threshold, SE selector, op buttons
│
├── processing/
│   ├── io/                          # Image loading and saving
│   ├── interpolation/               # Nearest-neighbor, bilinear, zoom dispatcher
│   ├── spatial/                     # Convolution engine, smoothing, edge, median
│   ├── histogram/                   # Local equalization, histogram utils
│   ├── geometric/                   # Rotation, shearing
│   ├── frequency/                   # FFT spectrum, notch filter, template matching
│   ├── noise/                       # Noise injection, ROI statistics
│   ├── morphology/                  # SE generators, erosion/dilation, opening/closing, boundary
│   ├── segmentation/                # Placeholder
│   ├── computer_vision/             # Placeholder
│   └── deep_learning/               # Placeholder (weights/ gitignored)
│
└── utils/
    ├── pipeline_state.py            # Op stack, undo/reset logic
    ├── image_utils.py               # dtype normalization, Qt conversion, binarize
    └── error_handler.py             # Graceful crash handling, error dialogs
```

---

## Installation

**Prerequisites:** Python 3.11+

```bash
git clone https://github.com/<your-org>/clinical-image-analysis-workbench.git
cd clinical-image-analysis-workbench
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

1. Use **File → Open** to load a DICOM, JPEG, or BMP image.
2. Select a processing tab (Filters, Histogram, Frequency, Noise, Morphology).
3. Configure parameters and click **Apply**.
4. Use the **Pipeline** tab to undo the last step or reset to the original.
5. Use **File → Save** to export the current processed image.

Draw a rectangular ROI directly on the image viewer to restrict histogram and statistics operations to that region.

---

## Team

| Name | Role |
|------|------|
| Mohamed Jameel Alabasy | Lead Developer |

---

## License

This project is licensed under the [MIT License](LICENSE).