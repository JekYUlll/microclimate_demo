from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make paper figures from reports outputs.")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    reports = cfg["reports_dir"]
    figures_dir = reports / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_overall = reports / "tables" / "metrics_overall.csv"
    metrics_by_h = reports / "tables" / "metrics_by_horizon.csv"
    metrics_extremes = reports / "tables" / "metrics_extremes.csv"

    if metrics_overall.exists():
        df = pd.read_csv(metrics_overall)
        plt.figure(figsize=(8, 4))
        plt.bar(df["model"], df["rmse"], label="RMSE")
        plt.bar(df["model"], df["mae"], label="MAE", alpha=0.7)
        plt.ylabel("Error")
        plt.title("Model comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "metrics_bar.png")
        plt.close()

    if metrics_by_h.exists():
        df = pd.read_csv(metrics_by_h)
        plt.figure(figsize=(8, 4))
        for model in df["model"].unique():
            sub = df[df["model"] == model]
            sub = sub.groupby("horizon")["rmse"].mean().reset_index()
            plt.plot(sub["horizon"], sub["rmse"], label=model)
        plt.xlabel("Horizon")
        plt.ylabel("RMSE")
        plt.title("RMSE by horizon")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figures_dir / "rmse_by_horizon.png")
        plt.close()

    # Prediction timeline
    preds_dir = reports / "preds"
    if metrics_overall.exists() and preds_dir.exists():
        metrics_df = pd.read_csv(metrics_overall)
        baseline = None
        for name in metrics_df["model"].tolist():
            if name not in {"tft", "tft_pinn"}:
                baseline = name
                break

        target_cols = cfg.get("columns", {}).get("targets", [])
        target = target_cols[0] if target_cols else None
        horizon = 1

        def load_series(model_name: str):
            path = preds_dir / f"{model_name}.csv"
            if not path.exists():
                return None
            df = pd.read_csv(path)
            df = df[df["horizon"] == horizon].copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            return df

        if target:
            plt.figure(figsize=(10, 4))
            if baseline:
                df_b = load_series(baseline)
                if df_b is not None:
                    plt.plot(df_b["timestamp"].head(cfg.get("plot", {}).get("timeline_points", 500)),
                             df_b[f"y_pred_{target}"].head(cfg.get("plot", {}).get("timeline_points", 500)),
                             label=f"{baseline} pred")

            df_tft = load_series("tft")
            if df_tft is not None:
                plt.plot(df_tft["timestamp"].head(cfg.get("plot", {}).get("timeline_points", 500)),
                         df_tft[f"y_pred_{target}"].head(cfg.get("plot", {}).get("timeline_points", 500)),
                         label="tft pred")

            df_pinn = load_series("tft_pinn")
            if df_pinn is not None:
                plt.plot(df_pinn["timestamp"].head(cfg.get("plot", {}).get("timeline_points", 500)),
                         df_pinn[f"y_pred_{target}"].head(cfg.get("plot", {}).get("timeline_points", 500)),
                         label="tft_pinn pred")

            if df_pinn is not None:
                plt.plot(df_pinn["timestamp"].head(cfg.get("plot", {}).get("timeline_points", 500)),
                         df_pinn[f"y_true_{target}"].head(cfg.get("plot", {}).get("timeline_points", 500)),
                         label="true", alpha=0.6)

            ax = plt.gca()
            locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            plt.ylabel(target)
            plt.title("Prediction timeline")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(figures_dir / "prediction_timeline.png")
            plt.close()

    if metrics_extremes.exists():
        df = pd.read_csv(metrics_extremes)
        for target in df["target"].unique():
            sub = df[df["target"] == target]
            plt.figure(figsize=(8, 4))
            for slice_name in ["bottom", "top"]:
                slice_df = sub[sub["slice"] == slice_name]
                plt.bar(slice_df["model"], slice_df["rmse"], alpha=0.7, label=slice_name)
            plt.title(f"Extremes RMSE - {target}")
            plt.ylabel("RMSE")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / f"extremes_{target}.png")
            plt.close()

    print(f"Saved figures to {figures_dir}")


if __name__ == "__main__":
    main()
