"""T5 CNN helpers: BraTS slice dataset, model, training, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .io import default_data_root, discover_cases, load_nifti
from .preprocess import clean_image, minmax_normalize


@dataclass
class SliceRecord:
    case_id: str
    slice_idx: int
    image: np.ndarray  # (H, W) float32 normalized
    mask: np.ndarray  # (H, W) float32
    label: int


def crop_center(img: np.ndarray, cropx: int, cropy: int) -> np.ndarray:
    """Center-crop a 2-D array (same idea as the original BraTS notebook)."""
    y, x = img.shape[:2]
    startx = max(x // 2 - (cropx // 2), 0)
    starty = max(y // 2 - (cropy // 2), 0)
    return img[starty : starty + cropy, startx : startx + cropx]


def resize_slice(img: np.ndarray, size=(128, 128)) -> np.ndarray:
    return cv2.resize(img.astype(np.float32), size, interpolation=cv2.INTER_AREA)


def _prepare_slice(mri_slice: np.ndarray, mask_slice: np.ndarray, target_size=(128, 128)):
    mri_slice = clean_image(mri_slice)
    mask_slice = clean_image(mask_slice)

    # Optional center crop when slices are large (BraTS 240-ish).
    h, w = mri_slice.shape[:2]
    if h >= 160 and w >= 160:
        mri_slice = crop_center(mri_slice, 160, 160)
        mask_slice = crop_center(mask_slice, 160, 160)

    mri_slice = minmax_normalize(mri_slice)
    mri_slice = resize_slice(mri_slice, target_size)
    mask_slice = cv2.resize(mask_slice.astype(np.float32), target_size, interpolation=cv2.INTER_NEAREST)
    # Keep mask labels discrete after resize.
    mask_bin = (mask_slice > 0).astype(np.float32)
    label = int(np.any(mask_bin > 0))
    return mri_slice.astype(np.float32), mask_bin, label


def preprocess_inference_slice(mri_slice: np.ndarray, target_size=(128, 128)) -> np.ndarray:
    """Apply the exact MRI-input branch of training preprocessing without a mask."""
    image, _, _ = _prepare_slice(mri_slice, np.zeros_like(mri_slice), target_size)
    return image[..., None].astype(np.float32)


def discover_labeled_cases(data_root: str | Path | None = None) -> list[dict]:
    """Return cases that have both FLAIR and segmentation."""
    cases = discover_cases(data_root or default_data_root())
    labeled = []
    for case in cases:
        mods = case.get("modalities", {})
        if "flair" in mods and "seg" in mods:
            labeled.append(case)
    return labeled


def build_case_slice_dataset(
    case: dict,
    *,
    target_size=(128, 128),
    max_slices: int | None = 40,
    min_brain_fraction: float = 0.02,
) -> list[SliceRecord]:
    """
    Build 2-D FLAIR slices + binary tumor labels for one patient.

    Labels come ONLY from the segmentation mask (tumor pixel present/absent).
    """
    flair_path = case["modalities"]["flair"]
    seg_path = case["modalities"]["seg"]
    flair, _ = load_nifti(flair_path)
    seg, _ = load_nifti(seg_path)

    if flair.shape != seg.shape:
        # Allow broadcastable depth mismatch only if spatial dims match.
        if flair.shape[:2] != seg.shape[:2] or flair.shape[2] != seg.shape[2]:
            raise ValueError(
                f"FLAIR/SEG shape mismatch for {case['case_id']}: {flair.shape} vs {seg.shape}"
            )

    depth = flair.shape[2]
    indices = list(range(depth))
    if max_slices is not None and depth > max_slices:
        # Take evenly spaced slices across the volume (keeps both tumor/non-tumor).
        indices = np.linspace(0, depth - 1, max_slices).astype(int).tolist()

    records: list[SliceRecord] = []
    for z in indices:
        mri_slice = flair[:, :, z]
        mask_slice = seg[:, :, z]
        if float(np.mean(mri_slice > 0)) < min_brain_fraction:
            continue
        img, mask, label = _prepare_slice(mri_slice, mask_slice, target_size)
        records.append(
            SliceRecord(
                case_id=case["case_id"],
                slice_idx=int(z),
                image=img,
                mask=mask,
                label=label,
            )
        )
    return records


def patient_level_split(
    case_ids: list[str],
    val_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split patients (not slices) into train/validation."""
    rng = np.random.default_rng(seed)
    ids = list(case_ids)
    rng.shuffle(ids)
    if len(ids) < 2:
        raise ValueError("Need at least 2 labeled patients for a patient-level split.")
    n_val = max(1, int(round(len(ids) * val_ratio)))
    n_val = min(n_val, len(ids) - 1)
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return train_ids, val_ids


def records_to_arrays(records: list[SliceRecord]) -> tuple[np.ndarray, np.ndarray, list[SliceRecord]]:
    if not records:
        return (
            np.zeros((0, 128, 128, 1), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            [],
        )
    X = np.stack([r.image[..., None] for r in records], axis=0).astype(np.float32)
    y = np.asarray([r.label for r in records], dtype=np.float32)
    return X, y, records


def build_cnn_datasets(
    data_root: str | Path | None = None,
    *,
    target_size=(128, 128),
    max_slices_per_case: int = 40,
    val_ratio: float = 0.3,
    seed: int = 42,
) -> dict[str, Any]:
    """Create patient-level train/val arrays from labeled BraTS-style cases."""
    labeled = discover_labeled_cases(data_root)
    if len(labeled) < 2:
        raise FileNotFoundError(
            "Need at least 2 cases with both FLAIR and SEG under BraTS2020/. "
            "Run scripts/download_labeled_brats_sample.py first."
        )

    by_case: dict[str, list[SliceRecord]] = {}
    for case in labeled:
        recs = build_case_slice_dataset(
            case,
            target_size=target_size,
            max_slices=max_slices_per_case,
        )
        if recs:
            by_case[case["case_id"]] = recs

    train_ids, val_ids = patient_level_split(list(by_case.keys()), val_ratio=val_ratio, seed=seed)
    train_records = [r for cid in train_ids for r in by_case[cid]]
    val_records = [r for cid in val_ids for r in by_case[cid]]

    X_train, y_train, train_records = records_to_arrays(train_records)
    X_val, y_val, val_records = records_to_arrays(val_records)

    return {
        "labeled_cases": list(by_case.keys()),
        "train_case_ids": train_ids,
        "val_case_ids": val_ids,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "train_records": train_records,
        "val_records": val_records,
    }


def make_tumor_cnn(input_shape=(128, 128, 1)):
    """Build the T5 CNN architecture (TensorFlow/Keras)."""
    try:
        from tensorflow.keras import layers, models
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TensorFlow is required for the CNN section. Install with:\n"
            "  py -3 -m pip install tensorflow"
        ) from exc

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_tumor_cnn(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    epochs: int = 8,
    batch_size: int = 16,
):
    try:
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TensorFlow is required. Install with: py -3 -m pip install tensorflow"
        ) from exc

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def evaluate_tumor_cnn(model, X_val, y_val) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    probs = model.predict(X_val, verbose=0).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    y_true = y_val.astype(int)

    return {
        "y_true": y_true,
        "y_pred": preds,
        "y_prob": probs,
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1]),
        "classification_report": classification_report(
            y_true,
            preds,
            labels=[0, 1],
            target_names=["No Tumor", "Tumor Present"],
            zero_division=0,
        ),
    }
