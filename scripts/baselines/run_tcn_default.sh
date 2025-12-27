#!/usr/bin/env bash
set -euo pipefail

# Default runner for TCN baseline on Taishan CSV.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${DATA_PATH:-$ROOT/data/AntAWS/3_hourly/Taishan_3h.csv}"
LOG="${LOG_PATH:-$ROOT/log/tcn_baseline.log}"
ENCODING="${ENCODING:-latin1}"

python "$ROOT/scripts/baselines/run_tcn.py" \
  --data "$DATA" \
  --target-col "Temperature(Ąć)" \
  --encoding "$ENCODING" \
  --horizon 6 \
  --input-window 24 \
  --train-ratio 0.8 \
  --stride 1 \
  --max-windows 200 \
  --epochs 5 \
  --kernel-size 3 \
  --num-filters 16 \
  --dilation-base 2 \
  --dropout 0.1 \
  --devices 1 \
  --log-path "$LOG" \
  --plot "$ROOT/plots/tcn_baseline.png" \
  "$@"
