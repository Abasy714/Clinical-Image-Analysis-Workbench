# Sample Images

Place your test images here before running the app.

## Folder structure
- dicom/   — DICOM files (.dcm) — loaded via pydicom
- jpeg/    — JPEG medical images (.jpg, .jpeg) — loaded via Pillow
- bmp/     — BMP medical images (.bmp) — loaded via Pillow

## Supported formats
Exactly as specified in the project statement: DICOM, JPEG, BMP.

## Getting more DICOM samples
Run:
    python -c "import pydicom.data; [print(f) for f in pydicom.data.get_testfiles_name()]"
to list all built-in DICOM test files shipped with pydicom.

## Notes
- All images are converted to grayscale on load
- DICOM metadata (Modality, PatientName, Age, BodyPartExamined) is extracted automatically
- JPEG and BMP metadata shows Width, Height, Bit depth only
