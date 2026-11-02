# CNN dataset audit

Audit performed from the local `BraTS2020/` files on 2 November 2026.

- Independent case folders: **8**.
- FLAIR available: **8/8**.
- Segmentation available: **6/8**; the two unsegmented cases are excluded from supervised labels and are never treated as negative slices.
- Labeled volumes: six FLAIR/SEG pairs, each **240 × 240 × 155**.
- Observed segmentation labels: **0, 1, 2, 3**. The binary classifier target uses `mask > 0`: 0 is background; every nonzero supplied reference label is tumour-positive for this slice-level target.
- Existing preprocessing: NaN/Inf cleaning; 160×160 center crop for large slices; per-slice min–max normalization; 128×128 resize; one FLAIR channel.
- Existing notebook and `cnn.py` use patient-level train/validation splitting. The new script uses a fixed 4/1/1 patient train/validation/test split because only six labeled patients exist.

The script records exact slice/class counts after its real split. Six labeled cases is too small for clinically validated independent performance claims.

## Completed persistent training run

- Fixed seed: `42`; requested epochs: 12; early stopping selected the best validation-loss checkpoint.
- Patient split: training `BraTS20_Training_457`, `449`, `218`, `206`; validation `306`; test `318`. There is no case overlap.
- Sampled split counts (`max_slices_per_case=40`): train 132 slices (56 negative, 76 positive), validation 35, test 35 (12 negative, 23 positive).
- All eligible labelled slices before the training cap: 798 (355 negative, 443 positive).
- Training-only class weights: negative 1.1786, positive 0.8684.
- Held-out one-case test accuracy: 0.6286; ROC-AUC: 0.9384; confusion matrix `[[11,1],[12,11]]`.
- This one-patient test estimate is not a clinically validated performance claim.
- Saved artifacts: `models/brain_tumor_cnn.keras`, `models/model_metadata.json`, `models/training_history.json`, `models/evaluation_report.json`, `models/training_accuracy.png`, and `models/training_loss.png`.
- Reload verification: model input `(None, 128, 128, 1)` and a preprocessed slice produced a `(1, 1)` sigmoid output.
