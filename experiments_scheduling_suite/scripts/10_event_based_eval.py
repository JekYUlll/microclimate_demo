from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_scheduling_suite.src.posthoc.event_eval import compute_event_metrics, compute_imputation_error
from experiments_scheduling_suite.src.plots.style import apply_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="事件段 vs 平稳段评估 / 插补误差评估。")
    parser.add_argument("--reports_dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "reports" / "_aggregate")
    parser.add_argument("--event_def", type=str, default="threshold_exceedance", choices=["threshold_exceedance", "q90_flux", "q90_wind"])
    parser.add_argument("--mode", type=str, default="event_eval", choices=["event_eval", "imputation_error"])
    parser.add_argument("--target_col", type=str, default=None)
    return parser.parse_args()


def _plot_event_bars(df: pd.DataFrame, out_path: Path, horizon: int) -> None:
    apply_style()
    sub = df[df["horizon"] == horizon]
    if sub.empty:
        return
    agg = sub.groupby(["segment"])["rmse"].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    # 保证 event / non_event 都有位置
    labels = ["event", "non_event"]
    values = [agg.get("event", 0.0), agg.get("non_event", 0.0)]
    ax.bar(labels, values, color=["tab:red", "tab:blue"])
    ax.set_title(f"Event vs Non-event RMSE (H={horizon})")
    ax.set_ylabel("RMSE")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_tables = args.out_dir / "tables"
    out_figs = args.out_dir / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    if args.mode == "event_eval":
        rows = []
        for run_dir in args.reports_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name == "_aggregate":
                continue
            df = compute_event_metrics(run_dir, event_def=args.event_def)
            if not df.empty:
                rows.append(df)
        if rows:
            event_df = pd.concat(rows, ignore_index=True)
            event_df.to_csv(out_tables / "event_metrics_long.csv", index=False)
            for h in sorted(event_df["horizon"].unique()):
                _plot_event_bars(event_df, out_figs / f"event_vs_nonevent_H{h}.png", horizon=h)
            print(f"Saved event metrics -> {out_tables / 'event_metrics_long.csv'}")
        else:
            print("No event metrics computed.")

    elif args.mode == "imputation_error":
        rows = []
        for run_dir in args.reports_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name == "_aggregate":
                continue
            df = compute_imputation_error(run_dir, target_col=args.target_col)
            if not df.empty:
                rows.append(df)
        if rows:
            imp_df = pd.concat(rows, ignore_index=True)
            imp_df.to_csv(out_tables / "imputation_error_long.csv", index=False)
            print(f"Saved imputation error -> {out_tables / 'imputation_error_long.csv'}")
        else:
            print("No imputation error computed.")


if __name__ == "__main__":
    main()
