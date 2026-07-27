#!/bin/bash
# 通过唯一远程 GPU 入口拉取 smoke 结果；不执行关机或电源操作。

set -euo pipefail

SERVER="${REMOTE_GPU_ALIAS:-remote-gpu}"
REMOTE_BASE="~/_code/microclimate_demo/rl_sensor_scheduling_framework"
LOCAL_BASE="/home/horeb/_code/microclimate_demo/rl_sensor_scheduling_framework/reports/runs"

ssh_cmd() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$SERVER" "$@"
}


echo "[$(date)] 脚本启动，等待到 08:30 拉取结果..."

# 等到 08:30
while true; do
    HOUR=$(date +%H)
    MIN=$(date +%M)
    if [ "$HOUR" -gt 8 ] || ([ "$HOUR" -eq 8 ] && [ "$MIN" -ge 30 ]); then
        break
    fi
    sleep 60
done

echo "[$(date)] 开始拉取结果..."

# 检查训练是否完成
TRAINING_DONE=$(ssh_cmd "ls $REMOTE_BASE/reports/runs/fix_smoke_test/training_log.csv 2>/dev/null && echo yes || echo no")

if [ "$TRAINING_DONE" = "yes" ]; then
    echo "[$(date)] 训练已完成，拉取结果..."
else
    echo "[$(date)] 训练可能还未完成，仍然拉取当前结果..."
fi

# 拉取 DQN 结果
scp -r \
    "$SERVER:$REMOTE_BASE/reports/runs/fix_smoke_test" \
    "$LOCAL_BASE/" && echo "[$(date)] fix_smoke_test 拉取成功" || echo "[$(date)] fix_smoke_test 拉取失败"

# 拉取训练日志
scp \
    "$SERVER:/tmp/smoke_dqn.log" \
    "/tmp/smoke_dqn_remote.log" && echo "[$(date)] DQN log 拉取成功" || true

scp \
    "$SERVER:/tmp/smoke_cmdp.log" \
    "/tmp/smoke_cmdp_remote.log" && echo "[$(date)] CMDP log 拉取成功" || true

echo "[$(date)] 脚本结束。"
