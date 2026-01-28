from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, ensure_dirs, dump_config_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full AntAWS pipeline from config.")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    dump_config_snapshot(cfg, cfg["reports_dir"] / "tables" / "config_used.yaml")
    summary_path = cfg["reports_dir"] / "tables" / "exp_summary.csv"
    summary = {
        "exp_id": cfg.get("exp_id"),
        "station_id_main": cfg.get("station_id_main"),
        "freq": cfg.get("freq"),
        "resample_freq": cfg.get("resample_freq"),
        "targets": ",".join(cfg.get("columns", {}).get("targets", [])),
        "horizons": ",".join(str(h) for h in cfg.get("horizons", [])),
        "window_size": cfg.get("window_size"),
        "split_mode": cfg.get("split", {}).get("mode"),
        "split_train": cfg.get("split", {}).get("train"),
        "split_val": cfg.get("split", {}).get("val"),
        "split_test": cfg.get("split", {}).get("test"),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    # 1) Prepare station
    run([sys.executable, "scripts/prepare_antaws_station.py", "--config", str(args.config)])

    # 2) Missingness profile
    station_id = cfg.get("station_id_main")
    station_path = cfg["processed_dir"] / f"{station_id}.csv"
    run([sys.executable, "scripts/profile_missingness.py", "--data", str(station_path), "--config", str(args.config)])

    # 3) Station groups
    run([sys.executable, "scripts/assign_station_groups.py", "--config", str(args.config)])

    # 4) Build dataset
    run([sys.executable, "scripts/build_dataset.py", "--config", str(args.config)])

    # 5) Baselines
    run([sys.executable, "scripts/run_baselines.py", "--config", str(args.config)])

    # 6) TFT+PINN
    run([sys.executable, "scripts/train_tft_pinn.py", "--config", str(args.config), "--mode", "tft_pinn"])

    # 7) Missing strategy compare
    run([sys.executable, "scripts/compare_missing_strategies.py", "--config", str(args.config)])

    # 8) Evaluate
    run([sys.executable, "scripts/evaluate_models.py", "--config", str(args.config)])

    # 9) Figures
    run([sys.executable, "scripts/make_figures.py", "--config", str(args.config)])

    # 10) Environment snapshot
    env_path = cfg["reports_dir"] / "tables" / "pip_freeze.txt"
    with env_path.open("w") as f:
        subprocess.check_call([sys.executable, "-m", "pip", "freeze"], stdout=f)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
