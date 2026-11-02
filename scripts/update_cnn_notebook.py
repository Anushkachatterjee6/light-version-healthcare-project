"""Replace the short T5 CNN notebook section with the full faculty-demo CNN flow."""

from __future__ import annotations

import json
from pathlib import Path

NB_PATHS = [
    Path(__file__).resolve().parents[1] / "HER_Biomedical_Image_Tutorials_T1_T6.ipynb",
]


def src(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [line + "\n" for line in lines]


NEW_CELLS = [
    (
        "markdown",
        """
# T5 — CNN-Based Biomedical Image Analysis

This section builds a practical **2-D CNN** on top of the existing BraTS-style MRI pipeline.

- **Input:** FLAIR MRI slices (not the segmentation mask)
- **Label:** binary tumor presence from the segmentation mask (`1` if any tumor pixel exists on that slice)
- **Split:** patient-level train/validation to avoid slice leakage
- **SIFT + RANSAC cells above are unchanged**
""",
    ),
    (
        "markdown",
        """
## 1. CNN Dataset Preparation

Load labeled cases that contain both **FLAIR** and **SEG**, normalize intensities, resize to `128×128`, and create binary slice labels from the mask.
""",
    ),
    (
        "code",
        """
# TensorFlow/Keras is required for the CNN section.
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    print("TensorFlow version:", tf.__version__)
except ImportError:
    tf = None
    print("TensorFlow is not installed.")
    print("Install it with: py -3 -m pip install tensorflow")

try:
    import seaborn as sns
except ImportError:
    sns = None
    print("Optional: py -3 -m pip install seaborn")

from her_core.cnn import (
    build_cnn_datasets,
    evaluate_tumor_cnn,
    make_tumor_cnn,
    train_tumor_cnn,
)
from her_core.io import default_data_root, discover_cases

print("Dataset root:", default_data_root())
print("Discovered cases:", [c["case_id"] for c in discover_cases()])
""",
    ),
    (
        "code",
        """
# Build a compact 2-D CNN dataset from FLAIR + SEG volumes.
# max_slices_per_case keeps the experiment practical on CPU/GPU.

cnn_data = None
try:
    cnn_data = build_cnn_datasets(
        data_root=DATA_ROOT if "DATA_ROOT" in globals() else None,
        target_size=(128, 128),
        max_slices_per_case=36,
        val_ratio=0.34,
        seed=42,
    )
except Exception as e:
    print("CNN dataset preparation failed:", e)
    print("Ensure BraTS2020 cases include both *_flair.nii.gz and *_seg.nii.gz")
    print("You can run: py -3 scripts/download_labeled_brats_sample.py")

if cnn_data is not None:
    print("Labeled patients:", cnn_data["labeled_cases"])
    print("Train patients:", cnn_data["train_case_ids"])
    print("Val patients:", cnn_data["val_case_ids"])
    print("X_train:", cnn_data["X_train"].shape, "y_train counts:", np.bincount(cnn_data["y_train"].astype(int)))
    print("X_val:", cnn_data["X_val"].shape, "y_val counts:", np.bincount(cnn_data["y_val"].astype(int)))

    # Quick visual check: one MRI slice and its mask-derived label
    demo = cnn_data["train_records"][0]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(demo.image.T, cmap="gray", origin="lower")
    axes[0].set_title(f"FLAIR slice {demo.slice_idx}")
    axes[1].imshow(demo.mask.T, cmap="magma", origin="lower")
    axes[1].set_title(f"SEG mask | label={demo.label}")
    for ax in axes:
        ax.axis("off")
    plt.suptitle("CNN sample (MRI input + mask label)")
    plt.tight_layout()
    plt.show()
""",
    ),
    (
        "markdown",
        """
## 2. Patient-Level Train/Validation Split

Patients are assigned entirely to train or validation. Slices from the same patient never appear in both sets.
""",
    ),
    (
        "code",
        """
if cnn_data is not None:
    print("Patient-level split")
    print("  train_idx / train patients:", cnn_data["train_case_ids"])
    print("  val_idx / val patients:", cnn_data["val_case_ids"])
    overlap = set(cnn_data["train_case_ids"]) & set(cnn_data["val_case_ids"])
    print("  patient overlap between train and val:", overlap if overlap else "None (good)")
""",
    ),
    (
        "markdown",
        """
## 3. CNN Architecture

Input shape: `128×128×1`

`Conv(32) → Pool → Conv(64) → Pool → Conv(128) → Pool → Flatten → Dense(128) → Dropout(0.5) → Dense(1, sigmoid)`
""",
    ),
    (
        "code",
        """
cnn_model = None
if tf is None:
    print("Skip architecture: install TensorFlow first.")
elif cnn_data is None:
    print("Skip architecture: CNN dataset unavailable.")
else:
    cnn_model = make_tumor_cnn(input_shape=cnn_data["X_train"].shape[1:])
    cnn_model.summary()
""",
    ),
    (
        "markdown",
        """
## 4. Model Training

Train for a small number of epochs with validation monitoring and EarlyStopping.
""",
    ),
    (
        "code",
        """
history = None
if cnn_model is None:
    print("Skipping training.")
else:
    history = train_tumor_cnn(
        cnn_model,
        cnn_data["X_train"],
        cnn_data["y_train"],
        cnn_data["X_val"],
        cnn_data["y_val"],
        epochs=8,
        batch_size=16,
    )
    print("Training finished.")
""",
    ),
    (
        "markdown",
        """
## 5. Training Performance
""",
    ),
    (
        "code",
        """
if history is not None:
    hist = history.history
    epochs_axis = range(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs_axis, hist["accuracy"], marker="o", label="Train accuracy")
    plt.plot(epochs_axis, hist.get("val_accuracy", []), marker="o", label="Validation accuracy")
    plt.title("CNN Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs_axis, hist["loss"], marker="o", label="Train loss")
    plt.plot(epochs_axis, hist.get("val_loss", []), marker="o", label="Validation loss")
    plt.title("CNN Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("No training history available.")
""",
    ),
    (
        "markdown",
        """
## 6. CNN Evaluation
""",
    ),
    (
        "code",
        """
cnn_eval = None
if cnn_model is None:
    print("Skipping evaluation.")
else:
    cnn_eval = evaluate_tumor_cnn(cnn_model, cnn_data["X_val"], cnn_data["y_val"])
    print("Validation accuracy: {:.3f}".format(cnn_eval["accuracy"]))
    print("Precision: {:.3f}".format(cnn_eval["precision"]))
    print("Recall: {:.3f}".format(cnn_eval["recall"]))
    print("F1-score: {:.3f}".format(cnn_eval["f1"]))
    print("\\nClassification report:\\n")
    print(cnn_eval["classification_report"])
""",
    ),
    (
        "markdown",
        """
## 7. Confusion Matrix
""",
    ),
    (
        "code",
        """
if cnn_eval is not None:
    cm = cnn_eval["confusion_matrix"]
    plt.figure(figsize=(5, 4))
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Tumor", "Tumor Present"],
            yticklabels=["No Tumor", "Tumor Present"],
        )
    else:
        plt.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")
        plt.xticks([0, 1], ["No Tumor", "Tumor Present"])
        plt.yticks([0, 1], ["No Tumor", "Tumor Present"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("CNN Confusion Matrix (Validation)")
    plt.tight_layout()
    plt.show()
""",
    ),
    (
        "markdown",
        """
## 8. MRI Prediction Demonstration

Predict on real validation FLAIR slices and compare against the ground-truth segmentation mask.
""",
    ),
    (
        "code",
        """
if cnn_model is None or cnn_data is None or not cnn_data["val_records"]:
    print("Skipping MRI prediction demo.")
else:
    val_records = cnn_data["val_records"]
    probs = cnn_eval["y_prob"]
    preds = cnn_eval["y_pred"]

    # Prefer a true-positive tumor slice for a clearer demo; otherwise first val slice.
    demo_idx = 0
    for i, rec in enumerate(val_records):
        if rec.label == 1:
            demo_idx = i
            break

    rec = val_records[demo_idx]
    prob = float(probs[demo_idx])
    pred = int(preds[demo_idx])
    pred_label = "Tumor Present" if pred == 1 else "No Tumor"
    true_label = "Tumor Present" if rec.label == 1 else "No Tumor"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rec.image.T, cmap="gray", origin="lower")
    axes[0].set_title("Original FLAIR MRI")
    axes[1].imshow(rec.mask.T, cmap="magma", origin="lower")
    axes[1].set_title(f"Ground Truth ({true_label})")
    axes[2].imshow(rec.image.T, cmap="gray", origin="lower")
    axes[2].set_title(f"CNN Prediction\\n{pred_label}")
    for ax in axes:
        ax.axis("off")
    plt.suptitle("CNN-Based Brain Tumor MRI Analysis", fontsize=14)
    plt.tight_layout()
    plt.show()

    print("CNN Prediction:", pred_label)
    print("Tumor Probability: {:.2f}".format(prob))
    print("Ground truth:", true_label)
    print("Case / slice:", rec.case_id, rec.slice_idx)
""",
    ),
    (
        "code",
        """
# Optional: several validation slices with predicted tumor probability
if cnn_model is not None and cnn_data is not None and len(cnn_data["val_records"]) > 0:
    n_show = min(6, len(cnn_data["val_records"]))
    # Spread across validation set
    idxs = np.linspace(0, len(cnn_data["val_records"]) - 1, n_show).astype(int)
    fig, axes = plt.subplots(n_show, 3, figsize=(10, 3.2 * n_show))
    if n_show == 1:
        axes = np.array([axes])

    for row, i in enumerate(idxs):
        rec = cnn_data["val_records"][i]
        prob = float(cnn_eval["y_prob"][i])
        pred = "Tumor" if cnn_eval["y_pred"][i] == 1 else "No Tumor"
        gt = "Tumor" if rec.label == 1 else "No Tumor"
        axes[row, 0].imshow(rec.image.T, cmap="gray", origin="lower")
        axes[row, 0].set_title(f"FLAIR | {rec.case_id} z={rec.slice_idx}")
        axes[row, 1].imshow(rec.mask.T, cmap="magma", origin="lower")
        axes[row, 1].set_title(f"GT: {gt}")
        axes[row, 2].imshow(rec.image.T, cmap="gray", origin="lower")
        axes[row, 2].set_title(f"Pred: {pred} ({prob:.2f})")
        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle("Validation slices — MRI / Ground Truth / CNN Probability", fontsize=13)
    plt.tight_layout()
    plt.show()

    print("\\nConcise metrics")
    print(cnn_model.summary())
    print(
        "Val Acc={:.3f} | Precision={:.3f} | Recall={:.3f} | F1={:.3f}".format(
            cnn_eval["accuracy"],
            cnn_eval["precision"],
            cnn_eval["recall"],
            cnn_eval["f1"],
        )
    )
""",
    ),
]


def cell_text(cell) -> str:
    return "".join(cell.get("source", []))


def is_old_cnn_section(cell) -> bool:
    text = cell_text(cell)
    markers = [
        "## 5.3 CNN-based image analysis",
        "def make_cnn(",
        "def create_slice_dataset(",
        "# TensorFlow/Keras is imported only for the CNN section.",
        "cnn_model = make_cnn",
        "X, y = create_slice_dataset",
    ]
    return any(m in text for m in markers)


def main() -> None:
    for path in NB_PATHS:
        if not path.exists():
            print("missing", path)
            continue
        nb = json.loads(path.read_text(encoding="utf-8"))
        start = None
        end = None
        for i, cell in enumerate(nb["cells"]):
            if is_old_cnn_section(cell) and start is None:
                start = i
            # Stop before T6 visualization section
            if start is not None and i > start:
                text = cell_text(cell)
                if text.startswith("# T6") or text.startswith("## T6") or "# T6 —" in text[:40]:
                    end = i
                    break
        if start is None:
            # Insert before T6 if present, else append
            insert_at = len(nb["cells"])
            for i, cell in enumerate(nb["cells"]):
                if "# T6" in cell_text(cell)[:60]:
                    insert_at = i
                    break
            new_cells = [{"cell_type": t, "metadata": {}, "source": src(s)} for t, s in NEW_CELLS]
            for c in new_cells:
                if c["cell_type"] == "code":
                    c["execution_count"] = None
                    c["outputs"] = []
            nb["cells"] = nb["cells"][:insert_at] + new_cells + nb["cells"][insert_at:]
            print(path, "inserted CNN section at", insert_at)
        else:
            if end is None:
                end = start + 1
                while end < len(nb["cells"]) and is_old_cnn_section(nb["cells"][end]):
                    end += 1
                # also consume consecutive code cells belonging to old CNN block
                while end < len(nb["cells"]):
                    text = cell_text(nb["cells"][end])
                    if text.startswith("# T6") or text.startswith("## T6") or text.startswith("# T6 —"):
                        break
                    if any(
                        k in text
                        for k in (
                            "make_cnn",
                            "create_slice_dataset",
                            "TensorFlow/Keras is imported only for the CNN",
                            "cnn_model.fit",
                            "CNN dataset shape",
                        )
                    ):
                        end += 1
                        continue
                    break
            new_cells = []
            for t, s in NEW_CELLS:
                cell = {"cell_type": t, "metadata": {}, "source": src(s)}
                if t == "code":
                    cell["execution_count"] = None
                    cell["outputs"] = []
                new_cells.append(cell)
            nb["cells"] = nb["cells"][:start] + new_cells + nb["cells"][end:]
            print(path, f"replaced cells {start}:{end} with {len(new_cells)} CNN cells")

        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
