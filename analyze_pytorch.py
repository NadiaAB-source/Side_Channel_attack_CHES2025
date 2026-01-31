# -*- coding: utf-8 -*-
"""
CHES 2025 Analyze Script (ConvTF Challenge Version)
- Loads ConvTF trained model (config + weights)
- Runs official evaluate() to compute GE and NTGE
"""

import os
import random
import numpy as np
import torch

from src.dataloader import ToTensor_trace, Custom_Dataset
from src.net import ConvTF  # Challenge model
from src.utils import evaluate, AES_Sbox, calculate_HW

if __name__ == "__main__":
    dataset = "CHES_2025"
    leakage = "HW"  # Choose 'HW' or 'ID'
    nb_traces_attacks = 1700
    total_nb_traces_attacks = 2000

    # Reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nb_attacks = 100

    # ---------------- Dataset ----------------
    # Always load with "ID" leakage as per challenge, we convert manually for HW if needed
    dataloadertest = Custom_Dataset(
        root='./../', dataset=dataset, leakage="ID",
        transform=ToTensor_trace()
    )

    # Define leakage model
    if leakage == 'ID':
        def leakage_fn(att_plt, k): return AES_Sbox[k ^ int(att_plt)]
        classes = 256
    else:  # HW
        def leakage_fn(att_plt, k):
            hw = [bin(x).count("1") for x in range(256)]
            return hw[AES_Sbox[k ^ int(att_plt)]]
        classes = 9
        dataloadertest.Y_attack = calculate_HW(dataloadertest.Y_attack)

    # Prepare attack/test set
    dataloadertest.split_attack_set_validation_test()
    dataloadertest.choose_phase("test")
    correct_key = dataloadertest.correct_key
    X_attack = dataloadertest.X_attack
    plt_attack = dataloadertest.plt_attack
    num_sample_pts = X_attack.shape[-1]

    # ---------------- Load Model ----------------
    model_type = "convtf"
    root = "./Result/"
    save_root = os.path.join(root, f"{dataset}_{model_type}_{leakage}")
    model_root = os.path.join(save_root, "models")

    # Load config (heads/layers)
    config = np.load(os.path.join(model_root, "model_configuration_0.npy"), allow_pickle=True).item()

    # Build and load ConvTF
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
        device, model, X_attack, plt_attack, correct_key,
        leakage_fn=leakage_fn,
        nb_attacks=nb_attacks,
        total_nb_traces_attacks=total_nb_traces_attacks,
        nb_traces_attacks=nb_traces_attacks
    )

    print(f"[EVAL] Final NTGE={NTGE}, GE@100k={GE[-1]}")
