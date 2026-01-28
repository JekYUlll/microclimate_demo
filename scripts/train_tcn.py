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

from src.train.train_tcn import train_tcn
from src.utils.config import load_config, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TCN baseline from config.")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML.")
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
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    result = train_tcn(cfg)

    model_path = cfg["models_dir"] / "tcn.pt"
    torch.save(result["model"].state_dict(), model_path)

    pred_path = cfg["reports_dir"] / "preds" / "tcn.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    write_preds(pred_path, result["t_ref"], result["y_true"], result["y_pred"], result["target_cols"], result["horizons"])

    fig_path = cfg["reports_dir"] / "figures" / "tcn_loss_curve.png"
    plt.figure(figsize=(6, 4))
    plt.plot(result["history"]["train"], label="train")
    plt.plot(result["history"]["val"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("tcn loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
