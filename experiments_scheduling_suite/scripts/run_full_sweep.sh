#!/usr/bin/env bash
set -euo pipefail

# 进入脚本目录并定位工程根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SUITE_ROOT}/.." && pwd)"

# 保证 Python 可以找到 experiments_scheduling_suite 包
export PYTHONPATH="${REPO_ROOT}"

BASE_CFG="${SUITE_ROOT}/configs/base.yaml"
DATASET_CFG="${SUITE_ROOT}/configs/datasets/synthetic.yaml"

# 1) 生成/准备原始数据（只需要一次）
python "${SUITE_ROOT}/scripts/00_generate_data.py" \
  --config "${BASE_CFG}" \
  --dataset "${DATASET_CFG}"

# 2) 定义全量组合（所有模型 / 所有插补 / 所有调度）
MISSINGNESS_LIST=(mcar block duty_cycle round_robin info_priority)
IMPUTATION_LIST=(none_maskaware linear spline kalman gp)
MODEL_LIST=(lstm transformer informer tcn xgboost naive mlp)

# 3) 逐组合运行：准备数据 -> 预训练可视化 -> 训练 -> 评估 -> 预测图 -> 总结图
for miss in "${MISSINGNESS_LIST[@]}"; do
  MISS_CFG="${SUITE_ROOT}/configs/missingness/${miss}.yaml"
  for imp in "${IMPUTATION_LIST[@]}"; do
    IMP_CFG="${SUITE_ROOT}/configs/imputation/${imp}.yaml"

    # 用 Python 生成与系统一致的 RUN_ID
    RUN_ID=$(python - <<PY
from pathlib import Path
from experiments_scheduling_suite.src.utils.io import load_yaml, build_run_id
base = load_yaml(Path("${BASE_CFG}"))
dataset = load_yaml(Path("${DATASET_CFG}")).get("dataset", {})
missing = load_yaml(Path("${MISS_CFG}")).get("missingness", {})
impute = load_yaml(Path("${IMP_CFG}")).get("imputation", {})
print(build_run_id(dataset, base, missing, impute))
PY
)

    echo "=== RUN_ID: ${RUN_ID} (missingness=${miss}, imputation=${imp}) ==="

    # A) 准备数据（缺失 + 插补 + 窗口化）
    python "${SUITE_ROOT}/scripts/01_prepare_dataset.py" \
      --config "${BASE_CFG}" \
      --dataset "${DATASET_CFG}" \
      --missingness "${MISS_CFG}" \
      --imputation "${IMP_CFG}" \
      --run-id "${RUN_ID}"

    # B) 预训练可视化
    python "${SUITE_ROOT}/scripts/02_visualize_pretrain.py" \
      --run-id "${RUN_ID}"

    # C) 训练所有模型
    python "${SUITE_ROOT}/scripts/03_train_models.py" \
      --run-id "${RUN_ID}" \
      --models "$(IFS=,; echo "${MODEL_LIST[*]}")"

    # D) 评估
    python "${SUITE_ROOT}/scripts/04_evaluate.py" \
      --run-id "${RUN_ID}"

    # E) 预测图
    python "${SUITE_ROOT}/scripts/05_plot_predictions.py" \
      --run-id "${RUN_ID}"

    # F) 总结大图
    python "${SUITE_ROOT}/scripts/06_plot_summary.py" \
      --run-id "${RUN_ID}"
  done
 done

 echo "All experiments completed."
