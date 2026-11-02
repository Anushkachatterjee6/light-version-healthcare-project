"""Download a few labeled BraTS-style cases from MSD Task01 using HTTP Range requests.

Only tar headers are scanned for most members; file payloads are downloaded
only for the selected image/label pairs.
"""

from __future__ import annotations

import io
import math
import re
import struct
import tarfile
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT / "BraTS2020"
URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
N_CASES = 6


def get_range(start: int, end: int) -> bytes:
    """Inclusive byte range download."""
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


def parse_tar_header(block: bytes) -> tuple[str, int, str] | None:
    if len(block) < 512 or block == b"\0" * 512:
        return None
    # Use tarfile's header parser via a tiny in-memory tar stream is awkward;
    # parse POSIX ustar fields directly.
    name = block[0:100].split(b"\0", 1)[0].decode("utf-8", "ignore")
    prefix = block[345:500].split(b"\0", 1)[0].decode("utf-8", "ignore")
    if prefix:
        name = f"{prefix}/{name}"
    size_str = block[124:136].split(b"\0", 1)[0].decode("ascii", "ignore").strip()
    if not size_str:
        return None
    try:
        size = int(size_str, 8)
    except ValueError:
        return None
    typeflag = chr(block[156]) if block[156] else "0"
    return name.replace("\\", "/"), size, typeflag


def case_id_from_name(name: str) -> str | None:
    m = re.search(r"BRATS_(\d+)", name)
    if not m:
        return None
    return f"BraTS20_Training_{int(m.group(1)):03d}"


def padded(size: int) -> int:
    return int(math.ceil(size / 512) * 512)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Indexing tar via Range requests...", flush=True)

    offset = 0
    selected: list[str] = []
    image_members: dict[str, tuple[int, int]] = {}  # case -> (data_offset, size)
    label_members: dict[str, tuple[int, int]] = {}

    empty = 0
    scanned = 0
    while empty < 2:
        header = get_range(offset, offset + 511)
        if header == b"\0" * 512 or not header:
            empty += 1
            offset += 512
            continue
        empty = 0
        parsed = parse_tar_header(header)
        if parsed is None:
            offset += 512
            continue
        name, size, typeflag = parsed
        data_offset = offset + 512
        scanned += 1
        if scanned % 50 == 0:
            print(f"  scanned {scanned} members @ {offset/1e9:.2f} GB", flush=True)

        cid = case_id_from_name(name)
        if cid and typeflag in ("0", "\0", ""):
            if "/imagesTr/" in name and name.endswith(".nii.gz"):
                if len(selected) < N_CASES and cid not in selected:
                    selected.append(cid)
                    image_members[cid] = (data_offset, size)
                    print(f"select image {cid} size={size/1e6:.1f}MB", flush=True)
                elif cid in selected and cid not in image_members:
                    image_members[cid] = (data_offset, size)
            elif "/labelsTr/" in name and name.endswith(".nii.gz"):
                if cid in selected:
                    label_members[cid] = (data_offset, size)
                    print(f"found label {cid} size={size/1e6:.1f}MB", flush=True)

        offset = data_offset + padded(size)

        if len(selected) >= N_CASES and len(label_members) >= N_CASES:
            if all(c in image_members and c in label_members for c in selected):
                print("Index complete for selected cases.", flush=True)
                break

    complete = [c for c in selected if c in image_members and c in label_members]
    print("Complete cases:", complete, flush=True)
    if len(complete) < 2:
        raise RuntimeError("Need at least 2 labeled cases for patient-level split.")

    import nibabel as nib
    import numpy as np

    for cid in complete:
        case_dir = OUT_ROOT / cid
        case_dir.mkdir(parents=True, exist_ok=True)
        img_off, img_size = image_members[cid]
        seg_off, seg_size = label_members[cid]
        print(f"Downloading {cid} image/label...", flush=True)
        img_bytes = get_range(img_off, img_off + img_size - 1)
        seg_bytes = get_range(seg_off, seg_off + seg_size - 1)

        multimodal = case_dir / f"{cid}_multimodal.nii.gz"
        seg_path = case_dir / f"{cid}_seg.nii.gz"
        multimodal.write_bytes(img_bytes)
        seg_path.write_bytes(seg_bytes)

        img = nib.load(str(multimodal))
        data = np.asanyarray(img.dataobj)
        if data.ndim != 4:
            raise RuntimeError(f"{cid} unexpected shape {data.shape}")

        for i, mod in enumerate(["flair", "t1", "t1ce", "t2"]):
            if i >= data.shape[-1]:
                break
            vol = np.asarray(data[..., i], dtype=np.float32)
            out = case_dir / f"{cid}_{mod}.nii.gz"
            nib.save(nib.Nifti1Image(vol, img.affine), str(out))
            print(f"  wrote {out.name} {vol.shape}", flush=True)

        multimodal.unlink(missing_ok=True)
        print(f"Saved {cid}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
