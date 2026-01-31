# -*- coding: utf-8 -*-
"""
CHES 2025 Conv-Transformer (ConvTF) Model
- Challenge-Optimized: Residual Convs + Relative Positional Bias
- Handles PCA-reduced traces (~1024 dims, 40 PoIs)
- 256-class AES key byte classifier
- Fixed for PyTorch >=2.4 (no reentrant warnings)
"""

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

# ---------- Relative Positional Bias ----------
class RelPosBias(nn.Module):
    def __init__(self, heads, max_rel):
        super().__init__()
        self.max_rel = max_rel
        self.bias = nn.Parameter(torch.zeros(2 * max_rel + 1, heads))
        nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, q_len, k_len):
        # Compute clipped relative positions
        idx = (torch.arange(q_len, device=self.bias.device)[:, None]
               - torch.arange(k_len, device=self.bias.device)[None, :])
        idx = idx.clamp(-self.max_rel, self.max_rel) + self.max_rel
        return self.bias[idx]


# ---------- Multi-Head Attention with Relative Bias ----------
class RelMHAttn(nn.Module):
    def __init__(self, d_model, n_heads, max_rel):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.rpb = RelPosBias(n_heads, max_rel)

    def forward(self, x):
        T = x.size(1)
        bias = self.rpb(T, T).permute(2, 0, 1).mean(0)  # Shape (T, T)
        out, _ = self.mha(x, x, x, attn_mask=bias, need_weights=False)
        return out


# ---------- Residual Convolutional Block ----------
class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm1d(c),
            nn.SiLU(),
            nn.Conv1d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm1d(c)
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.net(x) + x)


# ---------- Conv-Transformer Model ----------
class ConvTF(nn.Module):
    def __init__(self, L, num_classes=256, n_heads=4, n_layers=3, max_rel=256):
        super().__init__()
        # Convolutional Stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, 16, 5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.SiLU()
        )

        # Residual Convolutional Body
        self.body = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            nn.Conv1d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            ResidualBlock(32)
        )

        # Positional Encoding
        self.pos = nn.Parameter(torch.randn(1, math.ceil(L / 4), 32) * 0.02)

        # Transformer Layers
        self.tf = nn.ModuleList([
            nn.ModuleList([
                RelMHAttn(32, n_heads, max_rel),
                nn.LayerNorm(32),
                nn.Sequential(
                    nn.Linear(32, 64),
                    nn.SiLU(),
                    nn.Linear(64, 32)
                )
            ]) for _ in range(n_layers)
        ])

        # Final Classifier Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # Input: (B, 1, L)
        x = self.stem(x)
        x = self.body(x)
        T = x.size(2)

        # Add positional encoding
        x = x.permute(0, 2, 1) + self.pos[:, :T, :]

        # Transformer Blocks (checkpointed with explicit use_reentrant=False)
        for attn, ln, mlp in self.tf:
            x = ln(x + cp.checkpoint(attn, x, use_reentrant=False))
            x = ln(x + cp.checkpoint(mlp, x, use_reentrant=False))

        # Back to (B, C, T) and classify
        x = x.permute(0, 2, 1)
        return self.head(x)
