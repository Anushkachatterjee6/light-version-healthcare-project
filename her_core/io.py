"""NIfTI discovery and loading for BraTS-style HER MRI folders."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

MODALITY_KEYS = ("flair", "t1", "t1ce", "t2", "seg")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_data_root() -> Path:
    return project_root() / "BraTS2020"


def load_nifti(path):
    """Load a NIfTI volume. Returns (array, nibabel image)."""
    if nib is None:
        raise ImportError("Install nibabel first: pip install nibabel")
    nii = nib.load(str(path))
    return np.asarray(nii.get_fdata(), dtype=np.float32), nii


def infer_modality(path: Path) -> str | None:
    name = path.name.lower()
    for key in MODALITY_KEYS:
        if f"_{key}." in name or name.endswith(f"_{key}.nii") or name.endswith(f"_{key}.nii.gz"):
            return key
        if key in name.replace("-", "").replace("_", ""):
            # fallback for alternate naming such as Flair-brain.nii.gz
            if key == "flair" and "flair" in name:
                return "flair"
            if key == "t2" and ("_t2" in name or name.startswith("t2") or "-t2" in name):
                return "t2"
            if key == "t1ce" and ("t1ce" in name or "t1gd" in name):
                return "t1ce"
            if key == "t1" and ("_t1" in name or name.startswith("t1")) and "t1ce" not in name and "t1gd" not in name:
                return "t1"
            if key == "seg" and "seg" in name:
                return "seg"
    return None


def discover_cases(data_root: str | Path | None = None) -> list[dict]:
    """
    Discover patient/case folders under a BraTS-style root.

    Returns a list of dicts:
      { "case_id": str, "path": Path, "modalities": {modality: Path} }
    """
    root = Path(data_root) if data_root else default_data_root()
    if not root.exists():
        return []

    cases: dict[str, dict] = {}

    nii_files = sorted(list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz")))
    for f in nii_files:
        modality = infer_modality(f)
        # Prefer parent folder as case id when it looks like a BraTS case.
        parent = f.parent
        if parent == root:
            case_id = f.stem.replace(".nii", "")
        else:
            case_id = parent.name

        entry = cases.setdefault(
            case_id,
            {"case_id": case_id, "path": parent, "modalities": {}},
        )
        if modality:
            entry["modalities"][modality] = f
        else:
            # Keep unclassified volumes available under a generic key.
            entry["modalities"].setdefault("volume", f)

    return [cases[k] for k in sorted(cases.keys())]


def list_modalities(case: dict) -> list[str]:
    return sorted(case.get("modalities", {}).keys())


def load_case_modality(case: dict, modality: str):
    """Load one modality volume for a discovered case."""
    path = case["modalities"].get(modality)
    if path is None:
        raise FileNotFoundError(f"Modality '{modality}' not found for case {case['case_id']}")
    volume, nii = load_nifti(path)
    return volume, nii, path


def get_slice(volume: np.ndarray, slice_idx: int, axis: int = 2) -> np.ndarray:
    """Extract a 2-D slice from a 3-D volume along the given axis."""
    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {volume.shape}")
    n = volume.shape[axis]
    idx = int(np.clip(slice_idx, 0, n - 1))
    return np.take(volume, idx, axis=axis)
