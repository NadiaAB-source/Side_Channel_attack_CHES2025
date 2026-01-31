# -*- coding: utf-8 -*-
"""
Custom Dataset for CHES 2025 (Fast + Challenge-Compatible)
- Loads preprocessed PoI + QS + PCA traces (runs preprocessing only if cache is missing)
- Always returns PyTorch tensors (no NoneType issues)
- Adds channel dimension for Conv1d (1, L)
- Keeps same interface so it works with main_pytorch.py and trainer.py
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.preprocessing import run_hybrid_preprocessing  # PoI + QS + PCA pipeline


class Custom_Dataset(Dataset):
    def __init__(self, root='./', dataset="CHES_2025", leakage="HW", transform=None):
        cache_dir = os.path.join(root, "cache_hybrid")
        x_train_file = os.path.join(cache_dir, "X_train_fp32.npy")
        x_attack_file = os.path.join(cache_dir, "X_attack_fp32.npy")
        y_train_file = os.path.join(cache_dir, "y_train_int64.npy")
        plains_file = os.path.join(cache_dir, "plains_attack_u8.npy")

        # Preprocess only if no cache exists
        if not (os.path.exists(x_train_file) and os.path.exists(x_attack_file)):
            print("[INFO] No cached data found. Running hybrid PoI + QS + PCA pipeline...")
            run_hybrid_preprocessing(root)

        # Load preprocessed arrays (memory-mapped for speed)
        self.X_profiling = np.load(x_train_file, mmap_mode='r')
        self.X_attack = np.load(x_attack_file, mmap_mode='r')
        self.Y_profiling = np.load(y_train_file, mmap_mode='r')
        self.plt_attack = np.load(plains_file, mmap_mode='r')

        # Dummy plaintexts for profiling (not used by challenge scoring)
        self.plt_profiling = np.zeros_like(self.Y_profiling, dtype=np.uint8)

        # Attack labels (placeholders, unused by challenge)
        self.Y_attack = np.zeros(len(self.X_attack), dtype=np.int64)

        # AES key byte (fixed for challenge)
        self.correct_key = 0

        # Store transform for DataLoader
        self.transform = transform

        # Default data pointer (profiling set for training)
        self.X, self.Y = self.X_profiling, self.Y_profiling

    def split_attack_set_validation_test(self, val_ratio=0.1):
        """Splits attack set into validation and test sets for hyperparameter tuning."""
        self.X_attack_test, self.X_attack_val, self.Y_attack_test, self.Y_attack_val = train_test_split(
            self.X_attack, self.Y_attack, test_size=val_ratio, random_state=0
        )

    def choose_phase(self, phase):
        """Switches which data subset (train/validation/test) is served by __getitem__."""
        if phase == 'train':
            self.X, self.Y = self.X_profiling, self.Y_profiling
        elif phase == 'validation':
            self.X, self.Y = self.X_attack_val, self.Y_attack_val
        elif phase == 'test':
            self.X, self.Y = self.X_attack_test, self.Y_attack_test

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Copy to ensure write-safe NumPy array for PyTorch
        trace = np.array(self.X[idx], copy=True)
        label = int(self.Y[idx])

        # Convert to tensors (always, no NoneType issue)
        trace_tensor = torch.from_numpy(trace).unsqueeze(0).float()  # (1, L) for Conv1d
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Apply any extra transform (if provided)
        if self.transform:
            trace_tensor, label_tensor = self.transform({'trace': trace_tensor, 'sensitive': label_tensor})

        return trace_tensor, label_tensor


class ToTensor_trace(object):
    """Pass-through transform (kept for compatibility, can be extended for augmentations)."""
    def __call__(self, sample):
        # Already tensors from dataset, so just unpack
        return sample['trace'], sample['sensitive']
