from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.train_tft_pinn import train_tft
from src.utils.config import load_config, ensure_dirs, dump_config_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TFT or TFT+PINN model from config.")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML.")
    parser.add_argument("--mode", type=str, default="tft_pinn", choices=["tft", "tft_pinn"], help="Training mode.")
    return parser.parse_args()


def write_preds(path: Path, t_ref, y_true, y_pred, target_cols, horizons):
    rows = []
    for i, ts in enumerate(t_ref):
        for h_idx, h in enumerate(horizons):
            row = {"timestamp": str(ts), "horizon": int(h)}
            for t_idx, col in enumerate(target_cols):
                row[f"y_true_{col}"] = float(y_true[i, h_idx, t_idx])
                row[f"y_pred_{col}"] = float(y_pred[i, h_idx, t_idx])
            rows.append(row)
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    dump_config_snapshot(cfg, cfg["reports_dir"] / "tables" / "config_used.yaml")

    if args.mode == "tft":
        lambda_phys = 0.0
        model_name = "tft"
    else:
        lambda_phys = float(cfg.get("loss", {}).get("lambda_phys", 0.1))
        model_name = "tft_pinn"

    result = train_tft(cfg, lambda_phys=lambda_phys, model_name=model_name)

    model_path = cfg["models_dir"] / f"{model_name}.pt"
    torch.save(result["model"].state_dict(), model_path)

    pred_path = cfg["reports_dir"] / "preds" / f"{model_name}.csv"
    write_preds(pred_path, result["t_ref"], result["y_true"], result["y_pred"], result["target_cols"], result["horizons"])

    # Loss curve
    if model_name == "tft_pinn":
        fig_name = "loss_curve.png"
    else:
        fig_name = f"{model_name}_loss_curve.png"
    fig_path = cfg["reports_dir"] / "figures" / fig_name
    plt.figure(figsize=(6, 4))
    plt.plot(result["history"]["train"], label="train")
    plt.plot(result["history"]["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
