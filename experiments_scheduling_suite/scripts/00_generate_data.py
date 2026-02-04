from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_scheduling_suite.src.data.generator.synthetic_windblown import generate, save_csv
from experiments_scheduling_suite.src.utils.io import load_yaml, deep_merge, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成/准备原始数据。")
    parser.add_argument("--config", type=Path, required=True, help="base.yaml")
    parser.add_argument("--dataset", type=Path, required=True, help="dataset config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.config)
    dataset_cfg = load_yaml(args.dataset).get("dataset", {})
    cfg = deep_merge(base_cfg, {"dataset": dataset_cfg})

    paths = cfg.get("paths", {})
    ensure_dirs({
        "raw_dir": PROJECT_ROOT / paths.get("raw_dir", "data/raw"),
        "generated_dir": PROJECT_ROOT / paths.get("generated_dir", "data/generated"),
    })

    mode = dataset_cfg.get("mode", "synthetic")
    if mode == "synthetic":
        rows = int(dataset_cfg.get("rows", 2000))
        freq_seconds = int(dataset_cfg.get("freq_seconds", 1))
        df = generate(rows=rows, freq_seconds=freq_seconds, seed=int(cfg.get("run", {}).get("seed", 42)))
        out_name = dataset_cfg.get("output_csv", "synthetic.csv")
        out_path = PROJECT_ROOT / paths.get("generated_dir", "data/generated") / out_name
        save_csv(df, out_path)
        print(f"Generated synthetic data -> {out_path}")
    else:
        input_csv = Path(dataset_cfg.get("input_csv", ""))
        if not input_csv.exists():
            raise SystemExit(f"Missing raw data: {input_csv}")
        print(f"Using real data: {input_csv}")


if __name__ == "__main__":
    main()
