# -*- coding: utf-8 -*-
"""
CHES 2025 Metrics & Training Helpers
- Includes jitter, label smoothing, SNR weighting
- Guessing Entropy (GE), NTGE, and Challenge Score
- Uses your provided code structure (unchanged logic)
"""

import numpy as np, math, gc, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# === Curriculum Jitter & Label Sharpening ===
def get_jitter(epoch, max_epoch, base=32, max_shift=256):
    return min(base + int((epoch/max_epoch)*max_shift), max_shift)

def get_smoothing(epoch, max_epoch, start=0.05, end=0.0):
    return max(end, start*(1 - epoch/max_epoch))

# === SNR/PoI Weighting ===
def compute_snr_weights(traces, labels, num_classes=9):
    snr = np.zeros(traces.shape[1])
    for c in range(num_classes):
        subset = traces[labels==c]
        if len(subset)>0:
            mu_c = subset.mean(0)
            mu_t = traces.mean(0)
            var_c = subset.var(0)+1e-6
            snr += (mu_c - mu_t)**2/var_c
    return snr/snr.max()

# === Guessing Entropy & Challenge Metrics ===
def gge_fast(probs, plains, U=1000, key_byte0=0):
    """Estimate Guessing Entropy quickly using random subsets."""
    Q = probs.shape[0]
    ranks = []
    for _ in range(Q // U):
        idx = np.random.choice(Q, U, replace=False)
        logp = np.log(probs[idx] + 1e-9).sum(0)
        rank = (-logp).argsort().argsort()[key_byte0]
        ranks.append(rank)
    return int(np.mean(ranks))

def ge_curve(probs, key_byte0=0):
    """Compute full GE curve and NTGE (first index where GE=0)."""
    Q = probs.shape[0]
    logp_cum = np.zeros(256)
    ge_vals = []
    for i in range(Q):
        logp_cum += np.log(probs[i] + 1e-9)
        ranks = (-logp_cum).argsort().argsort()
        ge_vals.append(ranks[key_byte0])
    ge_vals = np.array(ge_vals)
    idx = np.where(ge_vals == 0)[0]
    ntge = idx[0] if len(idx) > 0 else np.inf
    return ge_vals, ntge

def challenge_score(ge_vals, ntge, N_attack=100_000, c=100_000):
    """
    Final CHES Challenge Score:
    - If GE=0 is achieved: return NTGE
    - Else: return GE@100k + N_attack + c
    """
    return ntge if ntge != np.inf else ge_vals[-1] + N_attack + c
