from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_scheduling_suite.src.posthoc.plotting.summary_panels import make_overview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成后期分析总览图（可选）。")
    parser.add_argument("--figures_dir", type=Path, default=PROJECT_ROOT / "reports" / "_aggregate" / "figures")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "_aggregate" / "figures" / "posthoc_overview.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures = sorted(args.figures_dir.glob("*.png"))
    make_overview(figures[:6], args.out)
    print(f"Saved overview -> {args.out}")


if __name__ == "__main__":
    main()
