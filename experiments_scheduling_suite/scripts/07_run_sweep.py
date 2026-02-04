from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_scheduling_suite.src.utils.io import build_run_id, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行完整 sweep。")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--dataset", type=Path, default=Path("configs/datasets/synthetic.yaml"))
    parser.add_argument("--quick", action="store_true", help="快速调试模式")
    return parser.parse_args()


def _run(cmd: List[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(PROJECT_ROOT / args.config)
    dataset_cfg = load_yaml(PROJECT_ROOT / args.dataset).get("dataset", {})

    if args.quick:
        missingness_list = ["mcar", "round_robin"]
        imputation_list = ["none_maskaware", "linear"]
        model_list = ["lstm", "informer", "naive"]
    else:
        missingness_list = ["mcar", "block", "duty_cycle", "round_robin", "info_priority"]
        imputation_list = ["none_maskaware", "linear", "spline", "kalman"]
        model_list = ["lstm", "transformer", "informer", "tcn", "xgboost", "naive", "mlp"]

    sweep_id = f"sweep_{dataset_cfg.get('name', 'dataset')}"
    sweep_reports = PROJECT_ROOT / "reports" / sweep_id
    (sweep_reports / "tables").mkdir(parents=True, exist_ok=True)
    leaderboard_rows = []

    for miss in missingness_list:
        miss_path = PROJECT_ROOT / "configs" / "missingness" / f"{miss}.yaml"
        miss_cfg = load_yaml(miss_path).get("missingness", {})
        for imp in imputation_list:
            imp_path = PROJECT_ROOT / "configs" / "imputation" / f"{imp}.yaml"
            imp_cfg = load_yaml(imp_path).get("imputation", {})

            run_id = build_run_id(dataset_cfg, base_cfg, miss_cfg, imp_cfg)

            _run([
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "01_prepare_dataset.py"),
                "--config",
                str(PROJECT_ROOT / args.config),
                "--dataset",
                str(PROJECT_ROOT / args.dataset),
                "--missingness",
                str(miss_path),
                "--imputation",
                str(imp_path),
                "--run-id",
                run_id,
            ])

            _run([sys.executable, str(PROJECT_ROOT / "scripts" / "02_visualize_pretrain.py"), "--run-id", run_id])

            _run([
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "03_train_models.py"),
                "--run-id",
                run_id,
                "--models",
                ",".join(model_list),
            ])

            _run([sys.executable, str(PROJECT_ROOT / "scripts" / "04_evaluate.py"), "--run-id", run_id])
            _run([sys.executable, str(PROJECT_ROOT / "scripts" / "05_plot_predictions.py"), "--run-id", run_id])
            _run([sys.executable, str(PROJECT_ROOT / "scripts" / "06_plot_summary.py"), "--run-id", run_id])

            # 收集 metrics 用于 leaderboard
            metrics_path = PROJECT_ROOT / "reports" / run_id / "tables" / "metrics_overall.csv"
            if metrics_path.exists():
                import pandas as pd

                df = pd.read_csv(metrics_path)
                df["run_id"] = run_id
                df["missingness"] = miss
                df["imputation"] = imp
                leaderboard_rows.append(df)

    if leaderboard_rows:
        import pandas as pd
        from experiments_scheduling_suite.src.eval.aggregate import leaderboard

        all_metrics = pd.concat(leaderboard_rows, ignore_index=True)
        horizons = base_cfg.get("run", {}).get("horizons", [1, 2, 3])
        lb = leaderboard(all_metrics, horizons=horizons)
        lb.to_csv(sweep_reports / "tables" / "leaderboard.csv", index=False)
        print(f"Saved sweep leaderboard -> {sweep_reports / 'tables' / 'leaderboard.csv'}")


if __name__ == "__main__":
    main()
