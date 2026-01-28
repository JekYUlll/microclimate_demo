from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.antaws_loader import load_antaws
from src.data.resample import resample_df
from src.utils.config import load_config, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a single AntAWS station CSV.")
    parser.add_argument("--input", type=Path, required=False, help="Raw AntAWS station file.")
    parser.add_argument("--output", type=Path, required=False, help="Output processed CSV.")
    parser.add_argument("--freq", type=str, default=None, help="Optional resample frequency (e.g. 3H or 1H).")
    parser.add_argument("--encoding", type=str, default=None, help="Optional encoding override.")
    parser.add_argument("--config", type=Path, default=None, help="Config file for exp_id/report paths.")
    parser.add_argument("--station-id", type=str, default=None, help="Station id (if using config for input/output).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = None
    if args.config:
        cfg = load_config(args.config)
        ensure_dirs(cfg)

    if args.input is None and cfg is None:
        raise SystemExit("--input or --config must be provided")

    input_path = args.input
    output_path = args.output
    if cfg is not None:
        station_id = args.station_id or cfg.get("station_id_main")
        if station_id is None:
            raise SystemExit("station_id_main missing in config; pass --station-id")
        if input_path is None:
            freq = args.freq or cfg.get("freq", "3H")
            input_path = cfg["paths"]["raw_root"] / f"{station_id}_{str(freq).lower()}.csv"
        if output_path is None:
            output_path = cfg["processed_dir"] / f"{station_id}.csv"
        if args.freq is None and cfg.get("resample_freq"):
            args.freq = cfg.get("resample_freq")

    if output_path is None:
        raise SystemExit("--output is required when --config is not provided")

    df, log = load_antaws(input_path, encoding=args.encoding)
    if args.freq:
        df = resample_df(df, args.freq)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if cfg is not None:
        log_path = cfg["reports_dir"] / "tables" / "prepare_log.csv"
        log_df = pd.DataFrame([{
            "station_id": args.station_id or cfg.get("station_id_main", ""),
            "input": str(input_path),
            "output": str(output_path),
            **log,
        }])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(log_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
