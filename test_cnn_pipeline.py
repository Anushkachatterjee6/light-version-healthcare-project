#!/usr/bin/env python3
"""Quick test of CNN pipeline with actual BraTS data."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from her_core.io import discover_cases
from her_core.cnn import discover_labeled_cases, build_cnn_datasets

print("=" * 70)
print("CNN IMPLEMENTATION VERIFICATION")
print("=" * 70)

root = Path("BraTS2020")
print(f"\n1. Data root exists: {root.exists()}")
print(f"   Path: {root.resolve()}")

# Discover all cases
all_cases = discover_cases(root)
print(f"\n2. Total cases found: {len(all_cases)}")
for case in all_cases[:3]:
    mods = sorted(case['modalities'].keys())
    print(f"   - {case['case_id']}: {mods}")

# Check labeled cases (with FLAIR + SEG)
labeled = discover_labeled_cases(root)
print(f"\n3. Labeled cases (FLAIR+SEG): {len(labeled)}")
for case in labeled:
    print(f"   - {case['case_id']}")

if len(labeled) >= 2:
    print(f"\n4. Building CNN datasets...")
    try:
        cnn_data = build_cnn_datasets(
            root, 
            target_size=(128, 128), 
            max_slices_per_case=36, 
            val_ratio=0.34, 
            seed=42
        )
        print(f"   ✓ Dataset created successfully!")
        print(f"   Train patients: {cnn_data['train_case_ids']}")
        print(f"   Val patients: {cnn_data['val_case_ids']}")
        print(f"   X_train shape: {cnn_data['X_train'].shape}")
        print(f"   X_val shape: {cnn_data['X_val'].shape}")
        
        y_train_dist = np.bincount(cnn_data['y_train'].astype(int))
        y_val_dist = np.bincount(cnn_data['y_val'].astype(int))
        print(f"   y_train (No Tumor/Tumor): {y_train_dist}")
        print(f"   y_val (No Tumor/Tumor): {y_val_dist}")
        
        print("\n✓ CNN dataset pipeline is working correctly!")
        print("\nSample slice info:")
        if cnn_data['train_records']:
            sample = cnn_data['train_records'][0]
            print(f"   Case: {sample.case_id}, Slice: {sample.slice_idx}, Label: {sample.label}")
            print(f"   Image shape: {sample.image.shape}, Mask shape: {sample.mask.shape}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"\n   ✗ Not enough labeled cases (need >= 2)")

print("\n" + "=" * 70)
