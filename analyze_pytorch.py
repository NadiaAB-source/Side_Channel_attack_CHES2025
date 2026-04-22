# -*- coding: utf-8 -*-
"""
CHES 2025 Analyze Script (ConvTF Challenge Version)
"""

import os
import random
import numpy as np
import torch

from src.dataloader_v0 import ToTensor_trace, Custom_Dataset
from src.net import ConvTF
from src.utils import evaluate, AES_Sbox

torch.backends.cudnn.enabled = False

if __name__ == "__main__":

    dataset = "CHES_2025"
    leakage = "ID"   # ✅ MUST MATCH TRAINING

    nb_traces_attacks = 1700
    total_nb_traces_attacks = 2000

    # ---------------- Reproducibility ----------------
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nb_attacks = 100

    # ---------------- Dataset ----------------
    dataloadertest = Custom_Dataset(
        root='./../', dataset=dataset, leakage="ID",
        transform=ToTensor_trace()
    )

    # ---------------- FIX KEY ----------------
    dataloadertest.correct_key = 127   # ✅ YOUR REAL KEY

    # ---------------- Leakage function ----------------
    def leakage_fn(att_plt, k):
        if isinstance(att_plt, np.ndarray):
            att_plt = att_plt[0]
        return AES_Sbox[k ^ int(att_plt)]

    classes = 256

    # ---------------- Prepare test set ----------------
    dataloadertest.split_attack_set_validation_test()
    dataloadertest.choose_phase("test")

    correct_key = dataloadertest.correct_key
    X_attack = dataloadertest.X_attack
    plt_attack = dataloadertest.plt_attack

    if len(plt_attack.shape) > 1:
        plt_attack = plt_attack[:, 0]

    num_sample_pts = X_attack.shape[-1]

    # ---------------- Load model ----------------
    model_type = "convtf"
    root = "./Result000/"   # adjust if needed
    save_root = os.path.join(root, f"{dataset}_{model_type}_{leakage}")
    model_root = os.path.join(save_root, "models")

    config = np.load(
        os.path.join(model_root, "model_configuration_0.npy"),
        allow_pickle=True
    ).item()

    model = ConvTF(
        num_sample_pts,
        num_classes=classes,
        n_heads=int(config.get("heads", 4)),
        n_layers=int(config.get("layers", 3))
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(model_root, "model_0.pth")))
    model.eval()

    # ---------------- Evaluate ----------------
    GE, NTGE = evaluate(
        device, model,
        X_attack, plt_attack, correct_key,
        leakage_fn=leakage_fn,
        nb_attacks=nb_attacks,
        total_nb_traces_attacks=total_nb_traces_attacks,
        nb_traces_attacks=nb_traces_attacks
    )

    print(f"[EVAL] Final NTGE={NTGE}, GE@100k={GE[-1]}")