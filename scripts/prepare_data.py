from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings
from src.data import load_all_stations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw meteorological Excel files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the processed CSV file.",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default=None,
        help="Override the resampling frequency defined in the config (e.g. 1H, 6H, 1D).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()

    if args.freq:
        settings.data.freq = args.freq

    df = load_all_stations(settings.paths.raw_data_dir, settings.data)

    processed_dir = settings.paths.processed_data_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output or (processed_dir / "meteorology.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows to {output_path}")


if __name__ == "__main__":
    main()
