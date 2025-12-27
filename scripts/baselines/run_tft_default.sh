#!/usr/bin/env bash
set -euo pipefail

# Default runner for TFT baseline on Taishan CSV.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${DATA_PATH:-$ROOT/data/AntAWS/3_hourly/Taishan_3h.csv}"
LOG="${LOG_PATH:-$ROOT/log/tft_baseline.log}"
ENCODING="${ENCODING:-latin1}"

python "$ROOT/scripts/baselines/run_tft.py" \
  --data "$DATA" \
  --target-col "Temperature(Ąć)" \
  --encoding "$ENCODING" \
  --horizon 6 \
  --input-window 24 \
  --train-ratio 0.8 \
  --stride 1 \
  --max-windows 200 \
  --epochs 5 \
  --hidden-size 32 \
  --num-heads 4 \
  --dropout 0.1 \
  --devices 1 \
  --log-path "$LOG" \
  --plot "$ROOT/plots/tft_baseline.png" \
  "$@"
