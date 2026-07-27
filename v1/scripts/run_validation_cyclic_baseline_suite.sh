#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  # Non-interactive SSH shells on the experiment server may not source conda.
  # Try the known installs without failing if one is absent.
  source /opt/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1 || \
    source "${HOME}/miniconda3/etc/profile.d/conda.sh" >/dev/null 2>&1 || \
    source "${HOME}/.conda/etc/profile.d/conda.sh" >/dev/null 2>&1 || true
fi

OUT_ROOT="${OUT_ROOT:-v1/artifacts/validation_cyclic_dwell_v6_transport_20260604}"
INPUT_ROOT="${INPUT_ROOT:-rl_sensor_scheduling_framework/reports/energy_account_split_protocol_gate_semimarkov}"
BUDGET_TAG="${BUDGET_TAG:-budget1p20}"
SENSOR_CFG="${SENSOR_CFG:-v1/configs/sensors/windblown_sensors_physical_event_v6_complex_static_break.yaml}"
SELECTION="${SELECTION:-event_transport_rich}"
BUDGET="${BUDGET:-1.36}"
STARTUP_PEAK_BUDGET="${STARTUP_PEAK_BUDGET:-1.75}"
ENERGY_CAPACITY="${ENERGY_CAPACITY:-70}"
INITIAL_ENERGY="${INITIAL_ENERGY:-70}"
HARVEST_PER_STEP="${HARVEST_PER_STEP:-0.8}"
RESERVE_ENERGY="${RESERVE_ENERGY:-20}"
TASK_ERROR_WEIGHT="${TASK_ERROR_WEIGHT:-0.3}"
TRAIN_STEPS="${TRAIN_STEPS:-128}"
TRAIN_ROLLOUTS="${TRAIN_ROLLOUTS:-4}"
STATIC_SELECTION_STEPS="${STATIC_SELECTION_STEPS:-256}"
STATIC_SELECTION_ROLLOUTS="${STATIC_SELECTION_ROLLOUTS:-4}"
EVAL_STEPS="${EVAL_STEPS:-256}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-4}"
VALIDATION_CYCLIC_TOP_K="${VALIDATION_CYCLIC_TOP_K:-4}"
VALIDATION_CYCLIC_DWELL_GRID="${VALIDATION_CYCLIC_DWELL_GRID:-2 4 8 16}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(41 42 44)
fi

mkdir -p "${OUT_ROOT}"

for seed in "${SEEDS[@]}"; do
  seed_dir="${INPUT_ROOT}/${BUDGET_TAG}_seed${seed}"
  out_dir="${OUT_ROOT}/seed${seed}"
  if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${out_dir}/gate_summary.json" ]; then
    echo "=== seed ${seed} skip existing $(date -Is) ==="
    continue
  fi
  mkdir -p "${out_dir}"
  echo "=== seed ${seed} start $(date -Is) ==="
  conda run -n darts python v1/scripts/run_protocol_gate.py \
    --truth-csv "${seed_dir}/truth_energy_split.csv" \
    --sensor-cfg "${SENSOR_CFG}" \
    --oracle-path "${seed_dir}/v2_tcn_oracle.pt" \
    --oracle-type tcn \
    --oracle-device cpu \
    --out-dir "${out_dir}" \
    --seed "${seed}" \
    --freq-s 10800 \
    --split-ratios 0.30 0.45 0.125 0.125 \
    --selection "${SELECTION}" \
    --selection-stride 64 \
    --lookback 20 \
    --horizon 8 \
    --train-steps "${TRAIN_STEPS}" \
    --train-rollouts "${TRAIN_ROLLOUTS}" \
    --static-selection-steps "${STATIC_SELECTION_STEPS}" \
    --static-selection-rollouts "${STATIC_SELECTION_ROLLOUTS}" \
    --eval-steps "${EVAL_STEPS}" \
    --eval-rollouts "${EVAL_ROLLOUTS}" \
    --max-active 4 \
    --budget "${BUDGET}" \
    --startup-peak-budget "${STARTUP_PEAK_BUDGET}" \
    --energy-account \
    --energy-capacity "${ENERGY_CAPACITY}" \
    --initial-energy "${INITIAL_ENERGY}" \
    --harvest-per-step "${HARVEST_PER_STEP}" \
    --reserve-energy "${RESERVE_ENERGY}" \
    --lambda-warmup-abort 0.08 \
    --planning-horizon 3 \
    --beam-width 4 \
    --max-branch 8 \
    --teacher-lambda-warmup-abort 0.16 \
    --candidate-prior-weight 0.5 \
    --candidate-prefilter-top-k 24 \
    --teacher-anchor-source validation_best \
    --anchor-improvement-margin 0.002 \
    --bc-epochs 1 \
    --bc-hidden-dim 32 \
    --bc-device cpu \
    --bc-batch-size 128 \
    --no-include-rule-baselines \
    --objective-mode task_composite \
    --task-error-weight "${TASK_ERROR_WEIGHT}" \
    --task-error-columns snow_mass_flux_kg_m2_s snow_particle_mean_diameter_mm snow_particle_mean_velocity_ms \
    --task-error-scales 1e-4 0.2 5.0 \
    --anchor-regret-guard \
    --bc-preserve-warming \
    --bc-action-support-top-k 6 \
    --no-include-bc-policy \
    --no-include-knn-policy \
    --no-include-mask-bc-policy \
    --deployable-selection validation \
    --deployable-selection-criterion static_margin_guard \
    --deployable-selection-min-mean-margin 0.001 \
    --deployable-selection-min-start-margin -0.01 \
    --deployable-selection-max-negative-starts 0 \
    --deployable-selection-require-guard-pass \
    --no-include-residual-bc-policy \
    --no-include-value-residual-policy \
    --no-include-rollout-value-policy \
    --no-include-ensemble-value-policy \
    --no-include-advantage-residual-policy \
    --no-learned-event-forecast \
    --no-include-cost-policy \
    --no-include-event-threshold-policy \
    --no-include-event-support-cycle-policy \
    --no-include-option-planner-policy \
    --no-include-teacher-rate-policy \
    --no-include-contextual-duty-policy \
    --no-include-sequence-mask-policy \
    --no-include-recurrent-value-policy \
    --no-include-recurrent-advantage-policy \
    --no-include-teacher-cycle-policy \
    --include-validation-cyclic-policy \
    --validation-cyclic-top-k "${VALIDATION_CYCLIC_TOP_K}" \
    --validation-cyclic-dwell-grid ${VALIDATION_CYCLIC_DWELL_GRID} \
    --validation-cyclic-preserve-warming
  echo "=== seed ${seed} done $(date -Is) ==="
done
