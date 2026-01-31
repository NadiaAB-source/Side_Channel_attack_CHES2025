# -*- coding: utf-8 -*-
"""
CHES 2025 Main Script (ConvTF Training + Evaluation)
- Optimized for speed and challenge compliance
- Reuses a single dataset instance for all phases (no deepcopy overhead)
- Monitors Guessing Entropy (GE) each epoch
"""

from src.dataloader import ToTensor_trace, Custom_Dataset
from src.net import ConvTF
from src.trainer import trainer
from src.utils import evaluate, AES_Sbox, calculate_HW

import os
import random
import numpy as np
import torch


if __name__ == "__main__":
    # ---------------- Config ----------------
    dataset = "CHES_2025"
    model_type = "convtf"
    leakage = "HW"  # "HW" or "ID"
    train_models = True
    num_epochs = 80
    total_num_models = 2  # Train multiple seeds/configs for robustness
    nb_traces_attacks = 1700
    total_nb_traces_attacks = 2000

    # ---------------- Directories ----------------
    root = "./Result/"
    save_root = os.path.join(root, f"{dataset}_{model_type}_{leakage}")
    model_root = os.path.join(save_root, "models")
    os.makedirs(model_root, exist_ok=True)
    print(f"root: {root}")
    print(f"save_path: {save_root}")

    # ---------------- Reproducibility ----------------
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------------- Leakage Model ----------------
    nb_attacks = 100
    if leakage == 'ID':
        def leakage_fn(att_plt, k): return AES_Sbox[k ^ int(att_plt)]
        classes = 256
    else:  # HW
        def leakage_fn(att_plt, k): return bin(int(AES_Sbox[k ^ int(att_plt)])).count("1")
        classes = 9

    # ---------------- Dataset ----------------
    dataloader = Custom_Dataset(
        root='./../', dataset=dataset, leakage=leakage,
        transform=ToTensor_trace()
    )

    # Convert labels for HW leakage if needed
    if leakage == "HW":
        dataloader.Y_profiling = np.array(calculate_HW(dataloader.Y_profiling))
        dataloader.Y_attack = np.array(calculate_HW(dataloader.Y_attack))

    # Split profiling/attack sets into train/val/test
    dataloader.split_attack_set_validation_test()

    # ---------------- Prepare DataLoaders ----------------
    # Training loader (profiling set)
    dataloader.choose_phase("train")
    train_loader = torch.utils.data.DataLoader(
        dataloader, batch_size=256, shuffle=True
    )

    # Validation loader (attack validation split)
    dataloader.choose_phase("validation")
    val_loader = torch.utils.data.DataLoader(
        dataloader, batch_size=256, shuffle=False
    )

    # Return to train phase for next loop
    dataloader.choose_phase("train")

    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": len(train_loader.dataset), "val": len(val_loader.dataset)}

    # Attack set for final evaluation
    correct_key = dataloader.correct_key
    X_attack = dataloader.X_attack
    plt_attack = dataloader.plt_attack
    num_sample_pts = X_attack.shape[-1]

    # ---------------- Train Multiple Configurations ----------------
    for num_models in range(total_num_models):
        config_path = os.path.join(model_root, f"model_configuration_{num_models}.npy")
        model_path = os.path.join(model_root, f"model_{num_models}.pth")

        if train_models:
            # Random search hyperparameters for ConvTF
            config = {
                "batch_size": int(np.random.choice([128, 256])),
                "lr": float(np.random.uniform(1e-4, 2e-3)),
                "heads": int(np.random.choice([2, 4, 8])),
                "layers": int(np.random.choice([2, 3, 4]))
            }
            np.save(config_path, config)

            # Rebuild loaders with updated batch size
            train_loader = torch.utils.data.DataLoader(
                dataloader, batch_size=config["batch_size"], shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                dataloader, batch_size=config["batch_size"], shuffle=False
            )
            dataloaders = {"train": train_loader, "val": val_loader}
            dataset_sizes = {"train": len(train_loader.dataset), "val": len(val_loader.dataset)}

            # Train the model (trainer monitors GE and saves best checkpoint)
            model = trainer(config, num_epochs, num_sample_pts, dataloaders, dataset_sizes,
                            model_type, classes, device,
                            X_attack=X_attack, plains_attack=plt_attack,
                            save_file=model_path)
        else:
            # Reload previously saved model
            config = np.load(config_path, allow_pickle=True).item()
            model = ConvTF(num_sample_pts, num_classes=classes,
                           n_heads=config["heads"], n_layers=config["layers"]).to(device)
            model.load_state_dict(torch.load(model_path))

        # Final evaluation (full attack set)
        GE, NTGE = evaluate(device, model, X_attack, plt_attack, correct_key,
                            leakage_fn=leakage_fn,
                            nb_attacks=nb_attacks,
                            total_nb_traces_attacks=total_nb_traces_attacks,
                            nb_traces_attacks=nb_traces_attacks)

        np.save(os.path.join(model_root, f"result_{num_models}.npy"), {"GE": GE, "NTGE": NTGE})
        print(f"[MODEL {num_models}] NTGE={NTGE}, GE@100k={GE[-1]}")
