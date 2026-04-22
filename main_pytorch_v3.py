# -*- coding: utf-8 -*-
"""
CHES 2025 Main Script (ConvTF Training + Evaluation)
- Optimized for speed and challenge compliance
- Reuses a single dataset instance for all phases (no deepcopy overhead)
- Monitors Guessing Entropy (GE) each epoch
"""

from src.dataloader_v3 import ToTensor_trace, Custom_Dataset
from src.net import ConvTF
from src.trainer import trainer
from src.utils import evaluate, AES_Sbox, calculate_HW
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import torch
import csv

if __name__ == "__main__":
    # ---------------- Config ----------------
    dataset = "CHES_2025"
    model_type = "convtf"
    leakage = "HW"  # "HW" or "ID"
    train_models = True
    num_epochs = 10
    total_num_models = 2  # Train multiple seeds/configs for robustness
    nb_traces_attacks = 5000
    total_nb_traces_attacks = 10000

        # ---------------- Directories ----------------
    root = "./Result33/"
    save_root = os.path.join(root, f"{dataset}_{model_type}_{leakage}")
    model_root = os.path.join(save_root, "models")

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)
    print(f"root: {root}")
    print(f"save_path: {save_root}")
    log_file = os.path.join(save_root, "training_log.csv")
    with open(log_file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["model_id","epoch","train_loss","val_loss","train_acc","val_acc","GE"])

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
        def leakage_fn(att_plt, k):
            # make sure plaintext is scalar
            if isinstance(att_plt, np.ndarray):
                att_plt = att_plt[0]
            return AES_Sbox[k ^ int(att_plt)]
        classes = 256

    else:  # HW
        def leakage_fn(att_plt, k):
            # make sure plaintext is scalar
            if isinstance(att_plt, np.ndarray):
                att_plt = att_plt[0]
            return bin(AES_Sbox[k ^ int(att_plt)]).count("1")
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
                "lr": float(np.random.uniform(0.002, 0.004)),
                "heads": int(np.random.choice([4, 8])),
                "layers": int(np.random.choice([3, 4]))
            }
            np.save(config_path, config)
            print("CONFIG:", config)

            # Rebuild loaders with updated batch size
            train_loader = torch.utils.data.DataLoader(
                dataloader, batch_size=config["batch_size"], shuffle=True, num_workers=2
            )
            val_loader = torch.utils.data.DataLoader(
                dataloader, batch_size=config["batch_size"], shuffle=False, num_workers=2
            )
            dataloaders = {"train": train_loader, "val": val_loader}
            dataset_sizes = {"train": len(train_loader.dataset), "val": len(val_loader.dataset)}
            print("Profiling traces:", len(dataloader.X_profiling))
            print("Attack traces:", len(dataloader.X_attack))
            print("Trace length:", dataloader.X_profiling.shape[1])
            # Train the model (trainer monitors GE and saves best checkpoint)
            model = trainer(config, num_epochs, num_sample_pts, dataloaders, dataset_sizes,
                model_type, classes, device,
                X_attack=X_attack, plains_attack=plt_attack,
                save_file=model_path,
                model_id=num_models,
                log_file=log_file)
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
        np.save(os.path.join(model_root, f"GE_curve_{num_models}.npy"), GE)
        

        plt.figure()
        plt.plot(range(len(GE)), GE, label="ConvTF Hybrid")
        plt.legend()
        plt.xlabel("Number of attack traces")
        plt.ylabel("Guessing Entropy")
        plt.title("GE Curve")
        plt.grid(True)
        plt.savefig(os.path.join(model_root, f"GE_curve_{num_models}.png"))
        plt.close()

        np.save(os.path.join(model_root, f"result_{num_models}.npy"), {"GE": GE, "NTGE": NTGE})
        print(f"[MODEL {num_models}] NTGE={NTGE}, GE@100k={GE[-1]}")
