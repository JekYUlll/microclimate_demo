#!/usr/bin/env bash
set -euo pipefail

# Sync lightweight no-warmup result artifacts from the server. Large rollout
# arrays and checkpoints are intentionally excluded.

REMOTE_HOST="${REMOTE_HOST:-remote-gpu}"
REMOTE_FRAMEWORK="${REMOTE_FRAMEWORK:-/home/zhangzhuyu/_code/microclimate_demo/rl_sensor_scheduling_framework}"
LOCAL_FRAMEWORK="${LOCAL_FRAMEWORK:-../rl_sensor_scheduling_framework}"

for rel in \
  reports/v31_split_protocol_no_warmup \
  reports/v31_split_protocol_no_warmup_hguard_reduced
do
  mkdir -p "$LOCAL_FRAMEWORK/$rel"
  rsync -av --prune-empty-dirs \
    --include='*/' \
    --include='v2_custom_ppo_metrics.csv' \
    --include='v2_ppo_metadata.json' \
    --include='custom_ppo_training_history.json' \
    --include='custom_ppo_training_log.csv' \
    --include='custom_ppo_candidate_prior.csv' \
    --include='validation_static_candidates.csv' \
    --include='split_protocol_manifest.json' \
    --include='*.log' \
    --include='*.done' \
    --exclude='*' \
    "$REMOTE_HOST:$REMOTE_FRAMEWORK/$rel/" \
    "$LOCAL_FRAMEWORK/$rel/"
done
