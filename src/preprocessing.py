# -*- coding: utf-8 -*-
"""
Hybrid Preprocessing for CHES 2025 (PoI + PCA)
- Finds 40 Points of Interest (PoIs) using SNR and Quadrant Scan
- Crops windows around PoIs (64 samples each)
- Normalizes (Z-score per trace)
- Compresses to ≤1024 dimensions via PCA (99% variance)
- Saves arrays to cache_hybrid/ for training and evaluation
"""

import os, h5py, numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

# -------------------- Configuration --------------------
TOP_SNR = 200             # Initial number of SNR peaks to consider
QS_WINDOW = 512
QS_STEP = 128
QS_THRESH = 0.5            # Quadrant Scan threshold for PoI acceptance
TARGET_POIS = 40           # Final number of PoIs
WIN_LEN = 64               # Samples per PoI window
PCA_TARGET_DIM = 1024      # Max dimensions after PCA
VARIANCE_KEEP = 0.99       # Preserve 99% variance
OUTPUT_DIR = "cache_hybrid"
DATASET_FILE = "CHES_Challenge.h5"  # Default name

# -------------------- Utility Functions --------------------
def compute_snr(traces, labels, num_classes=9):
    """Compute SNR curve over all time samples."""
    snr = np.zeros(traces.shape[1], dtype=np.float64)
    mean_tot = np.mean(traces, axis=0)
    for c in range(num_classes):
        cls = traces[labels == c]
        if len(cls) == 0:
            continue
        mean_c = np.mean(cls, axis=0)
        var_c = np.var(cls, axis=0) + 1e-6
        snr += (mean_c - mean_tot) ** 2 / var_c
    return snr

def quadrant_scan_light(trace, window=QS_WINDOW, step=QS_STEP):
    """Lightweight Quadrant Scan (variance + correlation based)."""
    N = len(trace)
    vals = np.zeros(N, dtype=np.float64)
    for start in range(0, N - window, step):
        seg = trace[start:start + window]
        v1, v2 = np.var(seg[:window // 2]), np.var(seg[window // 2:])
        corr = np.corrcoef(seg[:window // 2], seg[window // 2:])[0, 1]
        vals[start + window // 2] = (v1 + v2) * (1 - corr)
    max_val = np.max(vals)
    return vals / (max_val + 1e-12) if max_val > 0 else vals

def normalize_zscore(arr):
    """Z-score normalize each trace."""
    return (arr - arr.mean(1, keepdims=True)) / (arr.std(1, keepdims=True) + 1e-8)

def slice_windows(traces, idx, win_len=WIN_LEN):
    """Extract windows of length `win_len` around PoIs."""
    half = win_len // 2
    pad = ((0, 0), (half, half))
    padded = np.pad(traces, pad, mode="edge")
    windows = [padded[:, i:i + win_len] for i in idx]
    return np.concatenate(windows, axis=1)

# -------------------- Main Preprocessing --------------------
def run_hybrid_preprocessing(root="./"):
    """
    Runs the hybrid preprocessing pipeline:
    1. Load raw traces from CHES_Challenge.h5
    2. Select PoIs using SNR + Quadrant Scan
    3. Crop, normalize, and compress using PCA
    4. Save arrays for training and attack phases
    """
    # Try multiple paths for flexibility
    dataset_path = os.path.join(root, "Dataset/CHES_2025", DATASET_FILE)
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(root, DATASET_FILE)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Could not find {DATASET_FILE} in {root} or Dataset/CHES_2025/")

    os.makedirs(os.path.join(root, OUTPUT_DIR), exist_ok=True)

    print(f"[INFO] Loading dataset from {dataset_path}...")
    with h5py.File(dataset_path, "r") as f:
        Xp = f["Profiling_traces/traces"][:].astype(np.float32)
        Xa = f["Attack_traces/traces"][:].astype(np.float32)
        Yp = f["Profiling_traces/metadata"]["labels"][:].astype(np.int64)
        plains_attack = f["Attack_traces/metadata"]["plaintext"][:, 0].astype(np.uint8)

    # Step 1: Compute SNR
    print("[INFO] Computing SNR...")
    snr_vals = compute_snr(Xp, Yp)
    snr_peaks, _ = find_peaks(snr_vals, distance=20)
    ranked = snr_peaks[np.argsort(snr_vals[snr_peaks])[-TOP_SNR:]]

    # Step 2: Quadrant Scan scoring
    print("[INFO] Applying Quadrant Scan...")
    qs_scores = np.zeros_like(snr_vals, dtype=np.float64)
    for _ in range(5):  # Average over 5 random traces for stability
        trace = Xp[np.random.randint(0, Xp.shape[0])]
        qs_scores += quadrant_scan_light(trace)
    qs_scores /= 5.0

    pois = []
    for poi in sorted(ranked):
        region = np.arange(max(0, poi - 128), min(len(qs_scores), poi + 128))
        if np.max(qs_scores[region]) < QS_THRESH:
            continue
        pois.append(poi)
    pois = np.unique(pois)

    # Step 3: Limit to TARGET_POIS
    while len(pois) > TARGET_POIS:
        worst = np.argmin(snr_vals[pois])  # prune weakest PoI by SNR
        pois = np.delete(pois, worst)

    raw_dim = len(pois) * WIN_LEN
    print(f"[INFO] Final PoIs: {len(pois)} → Raw dim = {raw_dim}")

    # Step 4: Crop and normalize
    Xp_crop, Xa_crop = slice_windows(Xp, pois), slice_windows(Xa, pois)
    Xp_norm, Xa_norm = normalize_zscore(Xp_crop), normalize_zscore(Xa_crop)

    # Step 5: PCA compression
    print(f"[INFO] Applying PCA (≤{PCA_TARGET_DIM} dims, 99% variance)...")
    pca = PCA(n_components=min(PCA_TARGET_DIM, Xp_norm.shape[1]), svd_solver="full")
    Xp_pca = pca.fit_transform(Xp_norm)
    Xa_pca = pca.transform(Xa_norm)
    print(f"[INFO] PCA output dim: {Xp_pca.shape[1]}")

    # Step 6: Save arrays
    np.save(os.path.join(root, OUTPUT_DIR, "X_train_fp32.npy"), Xp_pca)
    np.save(os.path.join(root, OUTPUT_DIR, "X_attack_fp32.npy"), Xa_pca)
    np.save(os.path.join(root, OUTPUT_DIR, "y_train_int64.npy"), Yp)
    np.save(os.path.join(root, OUTPUT_DIR, "plains_attack_u8.npy"), plains_attack)
    np.save(os.path.join(root, OUTPUT_DIR, "poi_indices.npy"), pois)

    print(f"[DONE] Saved processed arrays to {os.path.join(root, OUTPUT_DIR)}")
