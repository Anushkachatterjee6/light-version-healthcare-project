# Brain Tumor Detection Using MRI Images

## Lightweight presentation copy

This repository intentionally excludes the raw `BraTS2020/` MRI dataset, all raw medical-image volumes, and large trained-model checkpoints. Supply an authorized BraTS-style dataset separately at `BraTS2020/` before running MRI-backed tutorials or checkpoint-dependent upload analysis. Small evaluation figures and JSON metadata are included for presentation and reproducibility context; they do not replace the dataset or checkpoint.

An integrated local teaching workspace: **MRI Dataset → Cleaning → Enhancement → Segmentation → Feature Extraction → CNN Classification → Clinical Text Analysis → MRI Feature Evaluation**.

Run locally from PowerShell:

```powershell
py -3 -m pip install -r requirements.txt
.\run_server.bat
```

- Dashboard: http://127.0.0.1:8000/
- Main application workspace: http://127.0.0.1:8000/app
- Image Analysis Studio (SIFT/RANSAC and CNN): http://127.0.0.1:8000/image-analysis
- API documentation: http://127.0.0.1:8000/docs

T1–T6 use local BraTS-style NIfTI MRI files. T7–T9 and T13–T15 derive case-level MRI features at request time: modality availability, FLAIR nonzero-voxel intensity statistics, volume depth, and supplied segmentation tumour burden. Missing modalities and masks are reported as missing; they are never fabricated. T10–T12 use clearly labeled synthetic brain-tumour report templates unless an authorized text CSV is uploaded.

The compact CNN trains a fresh patient-separated 32×32 classifier demo on local labeled slices. It returns actual training/validation measurements for that run, held-out predictions, image previews, confusion matrix and classification report. It is not a pretrained diagnostic system and its scores are not calibrated clinical confidence. The supplied local dataset has no verified tumour-negative cohort; T15 therefore evaluates segmentation-derived tumour-burden regression, not tumour/no-tumour classification, and no invalid ROC/AUC is shown.

Run verification:

```powershell
$env:LOKY_MAX_CPU_COUNT='4'
py -3 -m unittest discover -s tests -v
node tests/frontend_integration.cjs
node tests/studio_integration.cjs
```

Browser automation is not bundled. Open the served dashboard (not the HTML file), run T1, T7, T10, T13 and T15, then open Image Analysis Studio to run SIFT/RANSAC and CNN. See [SUBMISSION_REPORT.md](SUBMISSION_REPORT.md) and [DATASETS.md](DATASETS.md) for scope and limitations.
