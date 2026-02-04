独立实验套件（Wind-blown Snow / Sensor Scheduling / Imputation / Multi-Model Forecasting）

使用说明（建议流程）：
1) 生成或准备数据：
   python scripts/00_generate_data.py --config configs/base.yaml --dataset configs/datasets/synthetic.yaml
2) 准备数据集：
   python scripts/01_prepare_dataset.py --config configs/base.yaml --dataset configs/datasets/synthetic.yaml \
     --missingness configs/missingness/mcar.yaml --imputation configs/imputation/linear.yaml
3) 训练模型：
   python scripts/03_train_models.py --run-id <RUN_ID>
4) 评估与画图：
   python scripts/04_evaluate.py --run-id <RUN_ID>
   python scripts/05_plot_predictions.py --run-id <RUN_ID>
5) 扫描完整矩阵：
   python scripts/07_run_sweep.py --quick

所有脚本、模块均仅依赖 experiments_scheduling_suite/src 内代码与标准 pip 依赖。
