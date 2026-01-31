# -*- coding: utf-8 -*-
"""
CHES 2025 Trainer (ConvTF Optimized)
- Trains ConvTF with train/val phases
- Monitors Guessing Entropy (GE) each epoch
- Stops early when GE=0 (challenge goal)
- Saves only the best checkpoint (lowest GE) for evaluation
"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from src.net import ConvTF

# Utility: fast Guessing Entropy computation
def gge_fast(probs, plains, U=1000, key_byte0=0):
    """Estimate Guessing Entropy quickly over subsets of attack traces."""
    ranks = []
    for _ in range(probs.shape[0] // U):
        idx = np.random.choice(probs.shape[0], U, replace=False)
        logp = np.log(probs[idx] + 1e-9).sum(0)
        ranks.append((-logp).argsort().argsort()[key_byte0])
    return int(np.mean(ranks))


def trainer(config, num_epochs, num_sample_pts, dataloaders, dataset_sizes,
            model_type, classes, device,
            X_attack=None, plains_attack=None, save_file="best_convtf.pth"):
    """
    Trains ConvTF model with GE monitoring and early stopping.

    Parameters
    ----------
    config : dict
        Hyperparameters (batch_size, lr, heads, layers).
    num_epochs : int
        Max training epochs.
    num_sample_pts : int
        Feature length (L) after preprocessing.
    dataloaders : dict
        {'train': DataLoader, 'val': DataLoader}.
    dataset_sizes : dict
        {'train': int, 'val': int}.
    classes : int
        Output classes (AES = 256 or 9 for HW).
    device : torch.device
        CPU or GPU.
    X_attack, plains_attack : np.ndarray, optional
        Attack traces and plaintexts for monitoring GE.
    save_file : str
        File path to save the best checkpoint.
    """

    # Build ConvTF
    model = ConvTF(num_sample_pts,
                   num_classes=classes,
                   n_heads=int(config.get("heads", 4)),
                   n_layers=int(config.get("layers", 3))).to(device)

    # Optimizer
    lr = float(config.get("lr", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) \
        if config.get("optimizer", "Adam") == "Adam" \
        else torch.optim.RMSprop(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    best_ge, best_epoch = 999, -1

    print(f"[INFO] Training ConvTF for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects = 0.0, 0

            for traces, labels in dataloaders[phase]:
                traces, labels = traces.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(traces)
                    _, preds = torch.max(outputs, dim=1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * traces.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Guessing Entropy check
        if X_attack is not None and plains_attack is not None:
            model.eval()
            pa = []
            with torch.no_grad():
                for i in range(0, 1000, int(config.get("batch_size", 128))):
                    bx = torch.from_numpy(X_attack[i:i+int(config.get("batch_size", 128))]) \
                              .unsqueeze(1).float().to(device)
                    pa.append(F.softmax(model(bx), dim=1).cpu().numpy())
            pa = np.concatenate(pa, axis=0)

            ge_now = gge_fast(pa, plains_attack[:1000], 1000)
            print(f"[GE MONITOR] Epoch {epoch+1}: GE={ge_now} (best {best_ge})")

            # Save only if GE improves
            if ge_now < best_ge:
                best_ge, best_epoch = ge_now, epoch + 1
                torch.save(model.state_dict(), save_file)
                print(f"[INFO] Saved checkpoint (GE={best_ge}) → {save_file}")

            # Stop early if GE reaches 0 (NTGE finite)
            if ge_now == 0:
                print(f"[EARLY STOP] GE=0 reached at epoch {epoch+1}.")
                break

    print(f"\n[INFO] Training completed. Best GE={best_ge} (epoch {best_epoch}).")
    print(f"[INFO] Best model saved to {save_file}")
    return model
