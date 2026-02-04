————————————————————————————————————————
TITLE: Post-hoc Cross-Strategy Analysis Task List (Scheduling × Imputation × Model)

SCOPE:

* 现有实验已完成：每个 RUN 文件夹包含 metrics、preds、summary 图等。
* 现在新增一个“后期分析模块”，横向比较：调度策略/缺失机制对结果的影响（跨文件夹汇总）。
* 所有新产物写到：experiments_scheduling_suite/reports/_aggregate/

DO NOT:

* 不重训模型
* 不改旧实验目录结构
* 不依赖旧项目代码（只在 experiments_scheduling_suite 内新增代码）

————————————————————————————————————————
TASK 0. 新增目录与入口脚本（必做）

0.1 新增目录

* experiments_scheduling_suite/src/posthoc/

  * collect.py
  * derive_features.py
  * event_eval.py
  * stats_test.py
  * plotting/

    * heatmaps.py
    * boxplots.py
    * curves.py
    * scatter.py
    * rankings.py
    * summary_panels.py
* experiments_scheduling_suite/scripts/

  * 08_collect_results.py
  * 09_plot_cross_strategy.py
  * 10_event_based_eval.py
  * 11_significance_tests.py
  * 12_make_posthoc_report.py (可选，生成一张“索引图+表”)

0.2 新增输出目录

* experiments_scheduling_suite/reports/_aggregate/

  * tables/
  * figures/
  * logs/

————————————————————————————————————————
TASK 1. 统一结果汇总（必须先完成）

目标：

* 扫描 reports/ 下所有 RUN_ID 文件夹
* 解析 config_used.yaml（或 RUN_ID 命名）得到：dataset、freq、missingness_strategy、missingness_params、imputer、model
* 汇总 metrics（H=1/2/3）和必要的缺失形态统计到长表

输入来源（按优先级）：

* reports/RUN_ID/config_used.yaml（如果存在且字段齐全）
* reports/RUN_ID/tables/metrics_overall.csv 或 metrics_long.csv
* reports/RUN_ID/tables/missingness_stats.csv（如果有）
* reports/RUN_ID/preds/*.csv（用于某些后续分析）

输出：
A) reports/_aggregate/tables/metrics_long.csv

* 每行 = 1 个点：(run_id, dataset, freq, missingness, missingness_params, imputer, model, horizon, metric, value)
* metric 至少包含 RMSE/MAE/MAPE

B) reports/_aggregate/tables/run_manifest.csv

* 每行 = 1 个 run_id 的元信息：run_id、dataset、freq、missingness、params、imputer、timestamp、available_models、etc.

实现要求：

* 写脚本：scripts/08_collect_results.py
* 自动忽略缺文件的 run，并记录到 reports/_aggregate/logs/collect_warnings.txt
* 需要可过滤：--dataset、--missingness、--imputer、--model

————————————————————————————————————————
TASK 2. 衍生“缺失形态特征”表（解释性分析的关键）

目标：

* 对每个 RUN_ID 的“最终训练数据版本”（mask + impute 之前/之后）提取缺失结构特征
* 你若保留了“masked pre-impute CSV”，优先用它；若没有，则至少用 missingness_stats.csv + 从数据重算

建议特征（每个 run_id 一行）：

* overall_missing_rate
* per_feature_missing_rate_topk（可选）
* mean_gap_len, p90_gap_len, p95_gap_len, max_gap_len
* num_gaps
* co_missingness_mean（每个时刻缺失的变量数均值）
* obs_event_rate（观测事件频率；对调度策略很关键）
* effective_k（如果 round-robin/duty-cycle 可算到）

输出：

* reports/_aggregate/tables/missingness_features.csv

实现：

* src/posthoc/derive_features.py
* 脚本可集成到 08_collect_results.py 或单独写 scripts/08b_derive_missingness_features.py（可选）

————————————————————————————————————————
TASK 3. 横向图表 1：调度策略 × 插值方法 热力图（必做）

目的：

* 在固定模型与 horizon 下，横向比较不同调度/缺失策略与插值法组合的效果

输入：

* metrics_long.csv

定义：

* 选择一个 baseline（可配置），推荐：

  * baseline = missingness=MCAR(p=0.2) + imputer=linear
* heatmap 的值用：

  * ΔRMSE% = (RMSE - RMSE_baseline) / RMSE_baseline * 100
  * 或直接 RMSE（给一个开关）

输出（每个模型 3 张：H=1,2,3）：

* reports/_aggregate/figures/heatmap_MODEL_H1.png
* reports/_aggregate/figures/heatmap_MODEL_H2.png
* reports/_aggregate/figures/heatmap_MODEL_H3.png

实现：

* scripts/09_plot_cross_strategy.py --mode heatmap
* src/posthoc/plotting/heatmaps.py

————————————————————————————————————————
TASK 4. 横向图表 2：按调度策略分组的箱线/小提琴图（必做）

目的：

* 不再强调模型差异，而是看“策略整体优劣与稳定性”

两种版本（二选一或都做）：
A) 每个 horizon 一张：x=missingness_strategy(含参数)，y=RMSE

* 点 = 不同 model（可用 jitter）
  B) 固定模型：每个模型一张，x=missingness_strategy，y=RMSE（展示策略影响）

输出：

* reports/_aggregate/figures/boxplot_H1.png / H2 / H3
* 可选：reports/_aggregate/figures/boxplot_MODEL.png

实现：

* scripts/09_plot_cross_strategy.py --mode boxplot
* src/posthoc/plotting/boxplots.py

————————————————————————————————————————
TASK 5. 横向图表 3：相对改进曲线（Improvement curves）（推荐必做）

目的：

* 直接讲“策略从弱到强（缺失强度/预算）时，误差怎么变”
* 特别适合 MCAR(p) / block(b) / dutycycle(duty_ratio) 这些带参数族

输入：

* metrics_long.csv + run_manifest.csv（用于提取参数）

做法：

* 针对每一类策略 family（例如 MCAR），按参数排序：

  * x = p_missing 或 block_len 或 duty_ratio 或 budget_k
  * y = RMSE（或 ΔRMSE%）
  * 多条线：模型（或只选 top-3 模型）

输出：

* reports/_aggregate/figures/sensitivity_mcar_H1.png（H2/H3 同理）
* reports/_aggregate/figures/sensitivity_block_H1.png
* reports/_aggregate/figures/sensitivity_dutycycle_H1.png
* reports/_aggregate/figures/sensitivity_roundrobin_H1.png

实现：

* scripts/09_plot_cross_strategy.py --mode sensitivity
* src/posthoc/plotting/curves.py

————————————————————————————————————————
TASK 6. 解释性图表：缺失形态特征 vs 预测误差（强烈推荐）

目的：

* 回答“为什么某策略更好”：不是仅看 missing rate，而是 gap 结构、co-missingness、事件率等

输入：

* metrics_long.csv（固定 horizon）
* missingness_features.csv

图表集合（至少 3 张散点）：

1. x = max_gap_len, y = RMSE
2. x = p95_gap_len, y = RMSE
3. x = co_missingness_mean, y = RMSE
   可选：
4. x = obs_event_rate, y = RMSE

标记：

* 颜色 = missingness_strategy
* 点形状 = imputer 或 model（可选）
* 可加简单拟合线（线性回归/LOESS）

输出：

* reports/_aggregate/figures/scatter_gap_vs_rmse_H1.png（H2/H3 可选）

实现：

* scripts/09_plot_cross_strategy.py --mode scatter
* src/posthoc/plotting/scatter.py

————————————————————————————————————————
TASK 7. 纯插值评估（仅合成数据时强烈推荐）

前提：

* 你的 synthetic 数据存在“mask 前真值”
* 且你能定位哪些点是被 mask 掉后再 impute 回来的

目标：

* 评估 imputation 本身的误差，而非预测误差
* 用来解释：某插值在某调度下表现差，是因为插值误差巨大

输入：

* 若有保存：masked_pre_impute.csv + imputed.csv + original.csv
* 若没有：需要在未来版本里保存；当前可先跳过或仅做部分

指标：

* impute_RMSE_on_masked_positions
* impute_MAE_on_masked_positions
* 分条件：gap length 分桶、事件段 vs 平稳段（见 Task 8）

输出：

* reports/_aggregate/tables/imputation_error_long.csv
* reports/_aggregate/figures/imputation_error_by_gap.png
* reports/_aggregate/figures/imputation_error_by_strategy.png

实现：

* scripts/10_event_based_eval.py --mode imputation_error
* src/posthoc/event_eval.py（可复用事件切分逻辑）

————————————————————————————————————————
TASK 8. 事件段 vs 平稳段 的策略对比（强烈推荐，贴风吹雪场景）

目标：

* 分场景评估：吹雪发生（事件）时调度策略是否更重要

事件定义（可配置，默认取其一）：

* threshold_exceedance == 1
* 或 snow_mass_flux_kg_m2_s > Q90（test 集分位）
* 或 wind_speed_ms > Ut（若 Ut 可重算/已保存）

流程：

* 对每个 RUN_ID、每个模型，计算 test 上逐点误差
* 分割为 event / non-event 两组
* 统计 RMSE/MAE（每组）并输出

输出表：

* reports/_aggregate/tables/event_metrics_long.csv
  字段：run_id, missingness, imputer, model, horizon, segment(event/non_event), rmse, mae, mape

输出图：

1. 每个 horizon 一张：事件段 RMSE vs 平稳段 RMSE（分组柱状图）
2. “事件段恶化倍数”：(RMSE_event / RMSE_non_event) 按策略画条形图

实现：

* scripts/10_event_based_eval.py --mode event_eval
* src/posthoc/event_eval.py

————————————————————————————————————————
TASK 9. 排名稳定性分析（可选但加分）

目标：

* 看“调度策略是否改变模型排名”
* 输出调度之间的排名一致性

输入：

* metrics_long.csv

做法：

* 固定 imputer + horizon
* 每个 missingness_strategy 下对模型按 RMSE 排序
* 计算策略两两之间的 Spearman rho 或 Kendall tau

输出：

* reports/_aggregate/tables/rank_corr_matrix_H1.csv（H2/H3 可选）
* reports/_aggregate/figures/rank_corr_heatmap_H1.png

实现：

* scripts/09_plot_cross_strategy.py --mode ranking
* src/posthoc/plotting/rankings.py

————————————————————————————————————————
TASK 10. 显著性检验矩阵（可选但论文需要）

目标：

* 对比不同调度策略的误差序列是否显著不同（pairwise）

输入：

* preds/ 中的逐时刻预测（或逐窗口）
* 需要能对齐同一时间点的误差序列

方法：

* 对每个 (model, horizon, imputer) 固定：

  * 选择两种策略 A,B
  * 计算逐点绝对误差序列 |y-yhat|
  * 使用 Wilcoxon signed-rank 或 paired t-test
  * 输出 p-value + effect size（Cliff’s delta 可选）

输出：

* reports/_aggregate/tables/significance_MODEL_H1.csv
* reports/_aggregate/figures/significance_heatmap_MODEL_H1.png

实现：

* scripts/11_significance_tests.py
* src/posthoc/stats_test.py

————————————————————————————————————————
TASK 11. 生成“后期分析总览页”（可选）

目标：

* 把最关键的图表拼成 1 张或 1 个目录索引，方便展示

输出：

* reports/_aggregate/figures/posthoc_overview.png
* reports/_aggregate/tables/posthoc_key_findings.csv（可选）

实现：

* scripts/12_make_posthoc_report.py
* src/posthoc/plotting/summary_panels.py

————————————————————————————————————————
CLI SPEC（建议）

08_collect_results.py

* --reports_dir experiments_scheduling_suite/reports
* --out_dir experiments_scheduling_suite/reports/_aggregate
* --dataset FILTER
* --overwrite

09_plot_cross_strategy.py

* --in metrics_long.csv
* --out figures/
* --horizons 1 2 3
* --models informer transformer lstm tcn xgboost naive mlp
* --mode heatmap|boxplot|sensitivity|scatter|ranking
* --baseline "mcar_p0.2+linear"  (字符串匹配或 config 方式)

10_event_based_eval.py

* --in metrics_long.csv
* --reports_dir reports/
* --event_def threshold_exceedance|q90_flux|q90_wind
* --mode event_eval|imputation_error

11_significance_tests.py

* --reports_dir reports/
* --pairs "mcar_p0.2 vs block_b10" (或自动全对)
* --model informer
* --horizon 1
* --imputer linear

————————————————————————————————————————
DEFINITION OF DONE（完成标准）

* 生成 reports/_aggregate/tables/metrics_long.csv + run_manifest.csv
* 至少生成以下图（H=1/2/3）：

  * heatmap（每个模型 3 张）
  * boxplot（至少 3 张）
  * sensitivity（至少 2 个策略族）
  * scatter（至少 1 张）
* 若你有事件变量：生成 event_metrics_long.csv + 事件段对比图
* 所有脚本一键可跑：从 08 -> 09 -> 10（-> 11 可选）
* 不依赖重训，不修改旧结果文件夹

END
