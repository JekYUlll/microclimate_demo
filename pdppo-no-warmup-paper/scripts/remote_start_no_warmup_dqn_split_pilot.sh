#!/usr/bin/env bash
set -euo pipefail

# Local launcher. It syncs this isolated paper track to the server and starts a
# tmux session there. It does not run experiments locally.

REMOTE_HOST="${REMOTE_HOST:-remote-gpu}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-/home/zhangzhuyu/_code/microclimate_demo}"
LOCAL_PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="${SESSION:-pdppo_no_warmup_dqn_split_pilot_$(date +%Y%m%d_%H%M%S)}"
GPU_IDS_VALUE="${GPU_IDS:-1 5}"
MAX_JOBS_VALUE="${MAX_JOBS:-2}"
TOTAL_TIMESTEPS_VALUE="${TOTAL_TIMESTEPS:-60000}"
OUT_ROOT_VALUE="${OUT_ROOT:-reports/v31_no_warmup_dqn_split_diagnostic}"

rsync -av --delete \
  --exclude='results/' \
  --exclude='paper/*.pdf' \
  --exclude='paper/*.aux' \
  --exclude='paper/*.bbl' \
  --exclude='paper/*.blg' \
  --exclude='paper/*.log' \
  "$LOCAL_PROJECT_ROOT/pdppo-no-warmup-paper/" \
  "$REMOTE_HOST:$REMOTE_PROJECT_ROOT/pdppo-no-warmup-paper/"

ssh "$REMOTE_HOST" \
  "cd '$REMOTE_PROJECT_ROOT/pdppo-no-warmup-paper' && \
   chmod +x scripts/server_run_no_warmup_dqn_split_pilot.sh scripts/run_no_warmup_dqn_split.py && \
   cat > .dqn_split_pilot_env <<'EOF'
export GPU_IDS='$GPU_IDS_VALUE'
export MAX_JOBS='$MAX_JOBS_VALUE'
export TOTAL_TIMESTEPS='$TOTAL_TIMESTEPS_VALUE'
export OUT_ROOT='$OUT_ROOT_VALUE'
EOF
   tmux has-session -t '$SESSION' 2>/dev/null || \
   tmux new-session -d -s '$SESSION' 'source .dqn_split_pilot_env && bash scripts/server_run_no_warmup_dqn_split_pilot.sh'; \
   tmux ls"

echo "launched_or_existing session=$SESSION gpu_ids=$GPU_IDS_VALUE max_jobs=$MAX_JOBS_VALUE total_timesteps=$TOTAL_TIMESTEPS_VALUE"
