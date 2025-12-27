#!/usr/bin/env bash
set -euo pipefail

# Default runner for naive baselines (mean + seasonal) on Taishan CSV.
# Adjust paths/args below as needed.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${DATA_PATH:-$ROOT/data/AntAWS/3_hourly/Taishan_3h.csv}"
LOG="${LOG_PATH:-$ROOT/log/naive_baseline.log}"
ENCODING="${ENCODING:-latin1}"

python "$ROOT/scripts/baselines/run_naive.py" \
  --data "$DATA" \
  --target-col "Temperature(Ąć)" \
  --encoding "$ENCODING" \
  --horizon 6 \
  --train-ratio 0.8 \
  --stride 1 \
  --season-length 8 \
  --max-windows 200 \
  --log-path "$LOG" \
  --plot "$ROOT/plots/naive_baseline.png" \
  "$@"
