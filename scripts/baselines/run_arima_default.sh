#!/usr/bin/env bash
set -euo pipefail

# Default runner for ARIMA baseline on Taishan CSV.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${DATA_PATH:-$ROOT/data/AntAWS/3_hourly/Taishan_3h.csv}"
LOG="${LOG_PATH:-$ROOT/log/arima_baseline.log}"
ENCODING="${ENCODING:-latin1}"

python "$ROOT/scripts/baselines/run_arima.py" \
  --data "$DATA" \
  --target-col "Temperature(Ąć)" \
  --encoding "$ENCODING" \
  --horizon 6 \
  --train-ratio 0.8 \
  --stride 1 \
  --max-windows 200 \
  --arima 2,1,2 \
  --log-path "$LOG" \
  --plot "$ROOT/plots/arima_baseline.png" \
  "$@"
