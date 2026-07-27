#!/usr/bin/env bash
set -euo pipefail

# Launch no-warmup PD-PPO补跑 on the server. This script starts remote tmux
# jobs only; it does not run experiments locally.

REMOTE_HOST="${REMOTE_HOST:-remote-gpu}"
FRAMEWORK_ROOT="${FRAMEWORK_ROOT:-/home/zhangzhuyu/_code/microclimate_demo/rl_sensor_scheduling_framework}"
PY="${PY:-/home/zhangzhuyu/.conda/envs/darts/bin/python}"
PROFILE="${PROFILE:-hard_duty_reduced}"
SESSION="${SESSION:-pdppo_no_warmup_paper_${PROFILE}_$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-5}"

case "$PROFILE" in
  base_completion)
    OUT_DIR="${OUT_DIR:-reports/v31_split_protocol_no_warmup_paper_base}"
    CMD="$PY scripts/59_v31_split_protocol_grid.py \
      --out-dir $OUT_DIR \
      --sensor-cfg configs/sensors/windblown_sensors_balanced_no_warmup.yaml \
      --budgets ${BUDGETS:-1.65 1.70 1.75} \
      --seeds ${SEEDS:-41 42 43 44 45 46 47 48 49 50} \
      --workers ${WORKERS:-1} \
      --gpu-ids $GPU_IDS \
      --lambda-warmup-abort 0.0"
    ;;
  hard_duty_reduced)
    OUT_DIR="${OUT_DIR:-reports/v31_split_protocol_no_warmup_hguard_reduced}"
    CMD="$PY scripts/59_v31_split_protocol_grid.py \
      --out-dir $OUT_DIR \
      --sensor-cfg configs/sensors/windblown_sensors_balanced_no_warmup.yaml \
      --budgets ${BUDGETS:-1.70} \
      --seeds ${SEEDS:-43} \
      --workers ${WORKERS:-1} \
      --gpu-ids $GPU_IDS \
      --total-timesteps ${TOTAL_TIMESTEPS:-40000} \
      --lambda-warmup-abort 0.0 \
      --lambda-duty-balance ${LAMBDA_DUTY_BALANCE:-0.8} \
      --duty-balance-low ${DUTY_BALANCE_LOW:-0.12} \
      --duty-balance-high ${DUTY_BALANCE_HIGH:-0.85} \
      --duty-score-feedback ${DUTY_SCORE_FEEDBACK:-2.5} \
      --duty-hard-guard \
      --duty-hard-low ${DUTY_HARD_LOW:-0.12} \
      --duty-hard-high ${DUTY_HARD_HIGH:-0.85} \
      --duty-hard-score ${DUTY_HARD_SCORE:-12} \
      --awbc-coef ${AWBC_COEF:-0.02} \
      --prior-kl-coef ${PRIOR_KL_COEF:-0.05} \
      --candidate-prior-scale ${CANDIDATE_PRIOR_SCALE:-0.5} \
      --ent-coef ${ENT_COEF:-0.003} \
      --eval-duty-constrained-baselines \
      --baseline-duty-hard-low ${BASELINE_DUTY_HARD_LOW:-0.12} \
      --baseline-duty-hard-high ${BASELINE_DUTY_HARD_HIGH:-0.85} \
      --baseline-duty-hard-score ${BASELINE_DUTY_HARD_SCORE:-12} \
      --baseline-duty-score-feedback ${BASELINE_DUTY_SCORE_FEEDBACK:-2.5} \
      --bonferroni-family ${BONFERRONI_FAMILY:-3} \
      --skip-rollout-evaluation \
      --skip-collect"
    ;;
  *)
    echo "unknown PROFILE=$PROFILE" >&2
    exit 2
    ;;
esac

REMOTE_CMD="cd $FRAMEWORK_ROOT && \
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && \
  tmux has-session -t $SESSION 2>/dev/null || \
  tmux new-session -d -s $SESSION \"$CMD 2>&1 | tee ${OUT_DIR}_driver_${SESSION}.log\" && \
  tmux ls"

ssh "$REMOTE_HOST" "$REMOTE_CMD"
echo "launched_or_existing session=$SESSION profile=$PROFILE out_dir=$OUT_DIR"
