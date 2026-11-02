#!/usr/bin/env python3
"""Test CNN model building and quick training."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import numpy as np
from her_core.cnn import build_cnn_datasets, make_tumor_cnn, train_tumor_cnn, evaluate_tumor_cnn

print("=" * 70)
print("CNN MODEL BUILD AND TRAINING TEST")
print("=" * 70)

# Build dataset
print("\n1. Loading BraTS data...")
cnn_data = build_cnn_datasets(Path("BraTS2020"), max_slices_per_case=36, val_ratio=0.34, seed=42)
X_train, y_train = cnn_data['X_train'], cnn_data['y_train']
X_val, y_val = cnn_data['X_val'], cnn_data['y_val']
print(f"   ✓ X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"   ✓ X_val: {X_val.shape}, y_val: {y_val.shape}")

# Build model
print("\n2. Building CNN model...")
try:
    model = make_tumor_cnn(input_shape=(128, 128, 1))
    print(f"   ✓ Model created successfully!")
    params = model.count_params()
    print(f"   Total parameters: {params:,}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Quick training (1 epoch just to verify it works)
print("\n3. Quick training test (1 epoch)...")
try:
    history = train_tumor_cnn(
        model, X_train, y_train, X_val, y_val,
        epochs=1, batch_size=16
    )
    print(f"   ✓ Training completed!")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Evaluate
print("\n4. Model evaluation...")
try:
    eval_result = evaluate_tumor_cnn(model, X_val, y_val)
    print(f"   ✓ Validation Accuracy: {eval_result['accuracy']:.4f}")
    print(f"   Precision: {eval_result['precision']:.4f}")
    print(f"   Recall: {eval_result['recall']:.4f}")
    print(f"   F1-score: {eval_result['f1']:.4f}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ All tests passed! CNN implementation is ready for notebook execution.")
print("=" * 70)
