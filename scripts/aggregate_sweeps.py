from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate sweep experiments for sampling policies.")
    parser.add_argument("--reports-root", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--exp-ids", type=str, default=None, help="Comma-separated exp ids. If omitted, scan reports root.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory for aggregated tables/figures.")
    parser.add_argument("--temp-col", type=str, default=None, help="Override temperature target column.")
    parser.add_argument("--k-list", type=str, default="1,3", help="Budgets to plot for horizon charts.")
    return parser.parse_args()


def _load_config(path: Path) -> Dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _infer_temp_col(cfg: Dict, override: Optional[str], targets: List[str]) -> Optional[str]:
    if override:
        return override
    sampling = cfg.get("sampling", {})
    if sampling.get("temp_col"):
        return sampling.get("temp_col")
    for col in targets:
        if "temp" in str(col).lower():
            return col
    for col in targets:
        return col
    return None


def _budget_sort_val(value: object) -> float:
    if value is None:
        return float("inf")
    if isinstance(value, str) and value.lower() == "all":
        return 1e9
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1e9


def _exp_ids_from_reports(root: Path) -> List[str]:
    ids = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        tables = path / "tables"
        if (tables / "metrics_overall.csv").exists():
            ids.append(path.name)
    return sorted(ids)


def main() -> None:
    args = parse_args()
    reports_root = args.reports_root
    out_dir = args.out or (reports_root / "aggregate")
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    if args.exp_ids:
        exp_ids = [e.strip() for e in args.exp_ids.split(",") if e.strip()]
    else:
        exp_ids = _exp_ids_from_reports(reports_root)
    if not exp_ids:
        raise SystemExit("No experiments found to aggregate.")

    rows: List[Dict[str, object]] = []

    for exp_id in exp_ids:
        tables_dir = reports_root / exp_id / "tables"
        metrics_overall_path = tables_dir / "metrics_overall.csv"
        metrics_by_h_path = tables_dir / "metrics_by_horizon.csv"
        metrics_ext_path = tables_dir / "metrics_extremes.csv"
        cfg_path = tables_dir / "config_used.yaml"

        if not metrics_overall_path.exists():
            continue

        cfg = _load_config(cfg_path)
        sampling = cfg.get("sampling", {})
        policy = sampling.get("strategy", "none")
        budget_k = sampling.get("budget_k", "all")

        targets = cfg.get("columns", {}).get("targets", [])
        temp_col = _infer_temp_col(cfg, args.temp_col, targets)

        overall_df = pd.read_csv(metrics_overall_path)
        by_h_df = pd.read_csv(metrics_by_h_path) if metrics_by_h_path.exists() else pd.DataFrame()
        ext_df = pd.read_csv(metrics_ext_path) if metrics_ext_path.exists() else pd.DataFrame()

        for _, row in overall_df.iterrows():
            model = row.get("model")
            station_id = row.get("station_id", cfg.get("station_id_main"))

            rmse_temp_mean = None
            rmse_h1 = None
            rmse_h2 = None
            rmse_h8 = None
            if not by_h_df.empty and temp_col:
                sub = by_h_df[(by_h_df["model"] == model) & (by_h_df["target"] == temp_col)]
                if not sub.empty:
                    rmse_temp_mean = float(sub["rmse"].mean())
                    rmse_map = {int(h): float(rmse) for h, rmse in zip(sub["horizon"], sub["rmse"])}
                    rmse_h1 = rmse_map.get(1)
                    rmse_h2 = rmse_map.get(2)
                    rmse_h8 = rmse_map.get(8)

            extreme_rmse = None
            if not ext_df.empty and temp_col:
                ext_sub = ext_df[(ext_df["model"] == model) & (ext_df["target"] == temp_col) & (ext_df["slice"] == "bottom")]
                if not ext_sub.empty:
                    extreme_rmse = float(ext_sub["rmse"].mean())

            rows.append({
                "exp_id": exp_id,
                "station_id": station_id,
                "model": model,
                "policy": policy,
                "budget_k": budget_k,
                "rmse_overall": float(row.get("rmse")),
                "rmse_temp_mean": rmse_temp_mean,
                "rmse_h1": rmse_h1,
                "rmse_h2": rmse_h2,
                "rmse_h8": rmse_h8,
                "extreme_rmse_bottom": extreme_rmse,
            })

    perf_df = pd.DataFrame(rows)
    if perf_df.empty:
        raise SystemExit("No metrics rows found to aggregate.")

    perf_df.to_csv(out_dir / "perf_vs_budget.csv", index=False)

    # Prepare plotting columns
    perf_df["budget_k_label"] = perf_df["budget_k"].astype(str)
    perf_df["budget_k_num"] = perf_df["budget_k"].apply(_budget_sort_val)

    # RMSE vs budget
    for model in sorted(perf_df["model"].dropna().unique()):
        fig, ax = plt.subplots(figsize=(6, 4))
        model_df = perf_df[perf_df["model"] == model]
        for policy in sorted(model_df["policy"].dropna().unique()):
            sub = model_df[model_df["policy"] == policy]
            agg = sub.groupby("budget_k_num")["rmse_temp_mean"].mean().reset_index()
            labels = sub.drop_duplicates("budget_k_num").set_index("budget_k_num")["budget_k_label"]
            agg = agg.sort_values("budget_k_num")
            ax.plot(agg["budget_k_num"], agg["rmse_temp_mean"], marker="o", label=policy)
            ax.set_xticks(agg["budget_k_num"].tolist())
            ax.set_xticklabels([labels.get(x, str(x)) for x in agg["budget_k_num"].tolist()])
        ax.set_xlabel("Budget k")
        ax.set_ylabel("Temp RMSE (mean over horizons)")
        ax.set_title(f"RMSE vs Budget ({model})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figs_dir / f"rmse_vs_budget_{model}.png")
        plt.close(fig)

    # RMSE by horizon for selected k values
    k_list = [k.strip() for k in args.k_list.split(",") if k.strip()]
    for model in sorted(perf_df["model"].dropna().unique()):
        for k in k_list:
            sub = perf_df[(perf_df["model"] == model) & (perf_df["budget_k_label"] == k)]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            policies = sorted(sub["policy"].dropna().unique())
            x = np.arange(len(policies))
            width = 0.25
            rmse_h1 = sub.groupby("policy")["rmse_h1"].mean().reindex(policies)
            rmse_h2 = sub.groupby("policy")["rmse_h2"].mean().reindex(policies)
            rmse_h8 = sub.groupby("policy")["rmse_h8"].mean().reindex(policies)
            ax.bar(x - width, rmse_h1, width, label="H=1")
            ax.bar(x, rmse_h2, width, label="H=2")
            ax.bar(x + width, rmse_h8, width, label="H=8")
            ax.set_xticks(x)
            ax.set_xticklabels(policies, rotation=20)
            ax.set_ylabel("Temp RMSE")
            ax.set_title(f"RMSE by Horizon ({model}, k={k})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figs_dir / f"rmse_by_horizon_{model}_k{k}.png")
            plt.close(fig)

    # Extremes vs budget (bottom slice)
    if perf_df["extreme_rmse_bottom"].notna().any():
        for model in sorted(perf_df["model"].dropna().unique()):
            fig, ax = plt.subplots(figsize=(6, 4))
            model_df = perf_df[perf_df["model"] == model]
            for policy in sorted(model_df["policy"].dropna().unique()):
                sub = model_df[model_df["policy"] == policy]
                agg = sub.groupby("budget_k_num")["extreme_rmse_bottom"].mean().reset_index()
                labels = sub.drop_duplicates("budget_k_num").set_index("budget_k_num")["budget_k_label"]
                agg = agg.sort_values("budget_k_num")
                ax.plot(agg["budget_k_num"], agg["extreme_rmse_bottom"], marker="o", label=policy)
                ax.set_xticks(agg["budget_k_num"].tolist())
                ax.set_xticklabels([labels.get(x, str(x)) for x in agg["budget_k_num"].tolist()])
            ax.set_xlabel("Budget k")
            ax.set_ylabel("Extreme RMSE (bottom)")
            ax.set_title(f"Extremes vs Budget ({model})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figs_dir / f"extremes_vs_budget_{model}.png")
            plt.close(fig)

    print(f"Wrote aggregated table to {out_dir / 'perf_vs_budget.csv'}")


if __name__ == "__main__":
    main()
