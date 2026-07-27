#!/usr/bin/env bash
set -euo pipefail

# Server-side runner. Execute this only on the GPU server.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRAMEWORK_ROOT="${FRAMEWORK_ROOT:-/home/zhangzhuyu/_code/microclimate_demo/rl_sensor_scheduling_framework}"
PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
OUT_ROOT="${OUT_ROOT:-reports/v31_no_warmup_dqn_split_diagnostic}"
BUDGETS="${BUDGETS:-1.65 1.70}"
SEEDS="${SEEDS:-41 42 43}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-60000}"
GPU_IDS="${GPU_IDS:-4 5}"
MAX_JOBS="${MAX_JOBS:-2}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$FRAMEWORK_ROOT/$OUT_ROOT/logs" "$FRAMEWORK_ROOT/$OUT_ROOT/done"

read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [ "${#GPU_ARRAY[@]}" -eq 0 ]; then
  GPU_ARRAY=("")
fi

budget_tag() {
  "$PY" - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):.2f}".replace(".", "p"))
PY
}

job_idx=0
for budget in $BUDGETS; do
  tag="$(budget_tag "$budget")"
  for seed in $SEEDS; do
    label="budget${tag}_seed${seed}"
    metrics="$FRAMEWORK_ROOT/$OUT_ROOT/raw/$label/v2_dqn_split_metrics.csv"
    done_path="$FRAMEWORK_ROOT/$OUT_ROOT/done/$label.done"
    log_path="$FRAMEWORK_ROOT/$OUT_ROOT/logs/$label.log"
    if [ -f "$metrics" ] && [ -f "$done_path" ]; then
      echo "[skip] $label"
      continue
    fi

    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
      wait -n
    done

    gpu="${GPU_ARRAY[$((job_idx % ${#GPU_ARRAY[@]}))]}"
    job_idx=$((job_idx + 1))
    echo "[run] $label gpu=${gpu:-none} log=$log_path"
    (
      cd "$PROJECT_ROOT"
      if [ -n "$gpu" ]; then
        export CUDA_VISIBLE_DEVICES="$gpu"
      fi
      "$PY" scripts/run_no_warmup_dqn_split.py \
        --framework-root "$FRAMEWORK_ROOT" \
        --source-root reports/v31_split_protocol_no_warmup \
        --out-root "$OUT_ROOT" \
        --budget "$budget" \
        --seed "$seed" \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --device auto \
        --oracle-inference-device cpu
      if [ -f "$metrics" ]; then
        printf 'label=%s\nmetrics=%s\nlog=%s\n' "$label" "$metrics" "$log_path" > "$done_path"
      else
        echo "[missing] $metrics" >&2
        exit 1
      fi
    ) > "$log_path" 2>&1 &
  done
done

wait
echo "[done] no-warmup DQN split pilot"
