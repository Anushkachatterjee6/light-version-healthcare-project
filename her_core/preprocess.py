"""MRI preprocessing helpers reused from the HER T2 tutorial flow."""

from __future__ import annotations

import numpy as np


def clean_image(img):
    """Replace NaN/Inf with zeros and cast to float32."""
    img = np.asarray(img, dtype=np.float32)
    return np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)


def zscore_standardize(img, mask=None):
    """Z-score standardize non-background voxels."""
    img = clean_image(img)
    if mask is None:
        mask = img != 0

    values = img[mask]
    if values.size == 0:
        return np.zeros_like(img)

    mean = values.mean()
    std = values.std()
    if std < 1e-8:
        return np.zeros_like(img)

    standardized = np.zeros_like(img)
    standardized[mask] = (img[mask] - mean) / std
    return standardized


def minmax_normalize(img):
    """Scale intensities to [0, 1]."""
    img = clean_image(img)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-8:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def slice_to_uint8(slice_img):
    """Convert an MRI slice to grayscale uint8 for OpenCV SIFT."""
    norm = minmax_normalize(slice_img)
    return (norm * 255.0).astype(np.uint8)
