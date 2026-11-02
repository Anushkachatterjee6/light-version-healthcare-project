"""Shared HER biomedical MRI utilities used by the notebook, API, and UI."""

from .io import discover_cases, get_slice, list_modalities, load_nifti, load_case_modality
from .preprocess import clean_image, minmax_normalize, slice_to_uint8, zscore_standardize
from .sift_ransac import analyze_mri_sift_ransac
from .cnn import (
    build_cnn_datasets,
    evaluate_tumor_cnn,
    make_tumor_cnn,
    train_tumor_cnn,
)

__all__ = [
    "discover_cases",
    "get_slice",
    "list_modalities",
    "load_nifti",
    "load_case_modality",
    "clean_image",
    "minmax_normalize",
    "slice_to_uint8",
    "zscore_standardize",
    "analyze_mri_sift_ransac",
    "build_cnn_datasets",
    "make_tumor_cnn",
    "train_tumor_cnn",
    "evaluate_tumor_cnn",
]
