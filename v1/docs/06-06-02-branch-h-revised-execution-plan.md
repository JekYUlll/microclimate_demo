# Branch H 修订执行计划：Window-Level Mean-Risk Controller Selection

日期：2026-06-06

状态：下一主路线。本文档替代 `06-06-01-v1md` 中 Branch H 的具体实现规格；原文档保留为设计输入，不作为直接执行依据。

## 1. 目标

实现一个无 validation/final 泄漏、训练和部署决策单位一致的两层调度器：

- 外层 `MeanRiskControllerPolicy`：在完整评估窗口开始时，从多个动态 controller 中选择一个风险合格的 controller；否则执行 static anchor。
- 内层动态 controller：复用 task-only proxy-MPC，在窗口内逐步执行因果 forecast-aware scheduling。

外层直接学习候选 controller 相对 static anchor 的完整窗口结果分布，不再优化手工 scalar utility。

## 2. 固定定义

### 2.1 Margin

全项目统一定义：

```text
margin = static_objective - candidate_objective
```

- `margin > 0`：candidate 优于 static。
- `margin = 0`：与 static 等价。
- `margin < 0`：candidate 劣于 static。

任何新 dataset、CSV、model output、selection guard 和测试必须使用该符号。

### 2.2 独立样本单位

一个训练样本是：

```text
(independent_start, anchor, controller, paired_seed_offset) -> scalar window margin
```

一个 256-step rollout 只产生一个 window outcome。不得把同一个 window margin 复制到 256 个 step 并宣称为独立样本。

### 2.3 决策单位

第一版外层策略只在 256-step evaluation window 开始时选择 controller。选择后：

- dynamic controller 在窗口内继续逐步规划和切换；
- 外层不在窗口中途重新选择；
- 训练 label horizon、validation calibration horizon 和部署 horizon 都是 256 steps。

这是为避免再次出现 `16/32`-step calibration 驱动 `256`-step行为的单位错配。只有第一版通过后，才考虑收集匹配标签并增加 macro-boundary reselection。

## 3. 数据协议

### 3.1 Split 边界

保留现有 chronological protocol：

- oracle/pretrain：只训练 learned forecast models；
- RL/train：Branch H 数据生成、模型拟合和内部校准；
- validation：选择是否允许 deployable controller 进入 final；
- final：只做一次锁定评估。

final 结果不得用于特征设计、模型选择、阈值调整或补充训练。

### 3.2 Train 内部再划分

在现有 RL/train 区间内生成两组时间不重叠、window 不重叠的起点：

- `risk_fit_starts`：模型拟合；
- `risk_cal_starts`：one-sided risk calibration 和停止门控。

默认目标：

- `risk_fit_starts >= 32`
- `risk_cal_starts >= 12`
- 任意两个起点间隔至少 `256` steps
- 使用 `event_transport_rich`，但保留 regime coverage，不只取最极端窗口

若单 seed 无法获得足够的不重叠窗口，优先扩大 train start pool，不允许通过重叠窗口伪增样本。

### 3.3 Static anchor 无泄漏设计

禁止使用 validation-selected static 生成模型训练标签。

训练阶段：

1. 仅在 `risk_fit_starts` 上评估 feasible static masks。
2. 选择 top-K train anchor bank，默认 `K=8`。
3. 对每个 anchor 保存 mask、功耗、coverage 和 task-column support 特征。
4. 每个 start 使用 3 个 anchors：train-best anchor 加 2 个从 top-K bank
   平衡轮换的 anchors。不得对 8 anchors 和全部 controllers 做无约束全笛卡尔积。
5. Branch H 在这些 train anchors 上生成 paired outcomes，使模型显式条件化于 anchor。

部署阶段：

1. validation 正常选择 static anchor。
2. 将该 anchor 的同类特征输入已冻结的 risk model。
3. risk model 不再更新。

如果 validation anchor 不在 train top-K bank 中，也只能依靠 anchor feature 泛化，不能补做训练。

## 4. 候选 Controller Family

### 4.1 第一版候选

候选不是孤立 mask，也不是提前固定的 256-step truth-dependent sequence，而是可部署的 controller 配置：

1. static anchor fallback；
2. task-only proxy-MPC controller grid；
3. conservative proxy-MPC variants；
4. block-structured proxy variants。

### 4.2 Proxy controller grid

从现有 proxy-MPC 网格中保留一个预先声明的小网格，不使用 validation 结果筛参数：

- event weight：`0.5/1.5`
- target-rate weight：`0.0/0.5`
- dwell：`1/2`
- depth：`2/3`
- age weight：`0.25/0.75`
- 其他固定：
  - magnitude `1.0`
  - variability `0.5`
  - freshness `0.25`
  - power `0.03`
  - switch `0.03`
  - beam width `4`
  - max branch `8`
  - support top-k `16`

完整笛卡尔积为 32 个 controller。工程 pilot 可先用 16 个覆盖均匀的配置；正式数据收集使用预先锁定的 16 或 32 个配置，不根据 validation 增删。

### 4.3 Block variants

如需扰动，必须在 controller 的 planned prefix 上使用连续 block：

- block length：`2/4/8/16`
- replacement mask：teacher-supported feasible mask
- 每个 variant 最多替换一个连续 block

禁止随机替换 1-3 个孤立 step。孤立替换既不符合 warmup/dwell 机制，也与外层 window-level decision 不一致。

## 5. Paired Window Outcome Collection

### 5.1 Common-random-number replay

对同一 `(start, anchor, controller)`：

- static 和 candidate 使用相同 `seed_offset`；
- 相同 truth slice；
- 相同窗口长度；
- 相同 objective definition；
- 分别重置 policy/env state。

这降低传感器噪声对 paired margin 的方差。

### 5.2 每条记录

新增结构 `WindowRiskRecord`：

```text
seed
split_name
start
anchor_action_idx
anchor_mask
controller_id
controller_config
paired_seed_offset
static_objective
candidate_objective
margin
power_mean
warmup_abort_count
constraint_violation_count
feature_vector
feature_names
```

保存：

- `window_risk_rows.csv`
- `window_risk_dataset.npz`
- `window_risk_feature_schema.json`
- `window_risk_collection_manifest.json`

### 5.3 可用特征

所有 feature 必须在 controller 执行前可获得。

允许：

- 当前状态摘要、SOC、previous mask；
- fc4/SPC freshness；
- learned event probabilities；
- learned flux/diameter/velocity forecasts；
- time-of-day；
- anchor mask/power/coverage；
- controller hyperparameters；
- 基于 learned forecast 生成的 depth-2/3 planned-prefix 摘要；
- planned duty、planned switch count、planned warmup count。

planned-prefix 特征必须通过 policy/env clone 或纯函数计算，不能推进真实 rollout
state。

禁止：

- truth-future event；
- truth-future continuous variables；
- candidate 实际执行后的 duty/switch/freshness；
- validation/final outcome statistics；
- teacher future actions。

## 6. Model

### 6.1 第一版模型

优先使用项目现有 `scikit-learn`，不新增 XGBoost/LightGBM 依赖：

- mean model：
  `GradientBoostingRegressor(loss="squared_error")`
- q25 model：
  `GradientBoostingRegressor(loss="quantile", alpha=0.25)`
- negative-margin model：
  `HistGradientBoostingClassifier` 或 logistic regression

模型分别训练，但共享同一 feature schema 和 start-group split。

不训练 `feasibility_score`。硬功耗、启动峰值和 SOC 由现有 projector/energy account 强制执行；warmup abort 作为显式 outcome diagnostic，而不是用分类器替代硬约束。

### 6.2 Label

每条 row 只有一个 scalar margin：

```text
y = static_objective - candidate_objective
negative_label = int(y < 0)
```

q25 model 直接对全部 scalar margins 使用 pinball/quantile loss。不存在逐 row 的“真实 q25 label”。

### 6.3 Grouping

所有 fit、cross-validation 和 calibration 均按 `start` 分组：

- 同一 start 下的所有 anchors/controllers 必须留在同一 fold；
- 不允许按 row 随机拆分；
- 优先 chronological blocked split。

### 6.4 One-sided calibration

在 `risk_cal_starts` 上冻结模型后计算：

- q25 empirical coverage：`P(y <= q25_pred)`，目标约 `0.25`；
- mean model MAE/Spearman；
- q25 pinball loss；
- negative model Brier score；
- 相对常数基线的改进。

使用 split-conformal one-sided correction 得到保守 lower bound：

```text
risk_lower_bound = q25_pred - conformal_correction
```

correction 只由 `risk_cal_starts` 计算。

## 7. Data/Model Gate

进入 validation 前必须满足：

### 7.1 数据充分性

- fit independent starts `>=32`
- calibration independent starts `>=12`
- 至少 `10%` dynamic rows margin 为正
- 至少两个 controller family/config 在 fit 中产生正 margin
- hard constraint violation 为 `0`

若所有 controller 在 train held-out 上 mean margin 都为负，停止 Branch H，不进入 validation。

若 negative-label prevalence 小于 `5%` 或大于 `95%`，不强行训练不稳定的
classifier；该 seed 只使用 quantile/conformal risk gate，并在报告中标记
`negative_classifier_disabled`。全正结果不能因为缺少负样本而被判为数据失败。

### 7.2 模型有效性

在 `risk_cal_starts`：

- q25 pinball loss 至少优于 constant train-q25 baseline `10%`；
- 在 negative classifier 启用时，其 Brier score 优于 prevalence baseline；
- empirical q25 coverage 位于 `[0.15, 0.35]`；
- predicted risk 分箱后，风险应基本单调；
- feature schema 不包含任何 future truth 或 realized outcome。

不再使用“q25 误差 < 0.05”这一无量纲依据不足的规则。

## 8. Policy

新增 `ForecastAwareMeanRiskControllerPolicy`。

窗口开始时：

1. 构造 validation-selected static anchor features。
2. 对预声明 controller grid 构造 causal planned-prefix features。
3. 预测：
   - expected margin；
   - q25 margin；
   - conformal lower bound；
   - probability of negative margin。
4. 仅保留：
   - `risk_lower_bound >= 0`
   - `P(margin < 0) <= 0.25`
   - planned hard constraints feasible
5. 从安全候选中选择 predicted mean margin 最大者。
6. 没有安全候选时执行 static anchor。
7. 选中 dynamic controller 后，由其在完整 256-step 窗口内执行因果 proxy-MPC。

第一版不允许中途重新选择 controller。

## 9. 实现步骤

### H0：协议锁定

- [x] 固定 margin 符号并增加 regression test。
- [x] 增加 train 内部 non-overlap start split。
- [x] 增加 train top-K anchor bank。
- [x] 写 feature availability 审计。

### H1：数据收集

- [x] 实现 `WindowRiskRecord/Dataset`。
- [x] 实现 paired static/candidate full-window collector。
- [x] 支持断点续跑和每条 window 写盘。
- [x] 保存完整 provenance 和 feature schema。
- [x] 单元测试 common-random-number pairing 和无 split overlap。

### H2：模型

- [x] 实现 mean/q25/negative models。
- [x] 实现 grouped chronological fit/calibration。
- [x] 实现 constant baselines、pinball/Brier/coverage diagnostics。
- [x] 实现按 independent start 分组的 one-sided conformal correction。
- [x] 保存模型、metrics 和 calibration rows。

### H3：策略和 runner

- [x] 实现 `ForecastAwareMeanRiskControllerPolicy`。
- [x] 接入独立的 `evaluate_mean_risk_controller.py` gate runner，避免继续扩张旧入口。
- [x] 增加 source-run 驱动的独立 H0-H4 执行路径。
- [ ] 正式 seed41 结果完成后接入 aggregate 和 CHANGELOG。

### H4：验证

- [x] 本地 tiny CPU engineering smoke，只验证符号、split、I/O、fallback。
- [ ] 远程 seed41 data/model pilot（运行中）。
- [ ] 只有 data/model gate 通过后，运行 seeds `41/42/44`。

## 10. 实验顺序与停止条件

### 10.1 Seed41 pilot

目的：验证数据和模型，不作为 claim。

初始规模：

- fit starts `16`
- calibration starts `8`
- controller configs `16`
- anchors per start `3`
- horizon `256`

输出必须报告：

- 独立 window rows 数；
- margin 分布；
- controller-wise positive rate；
- q25 baseline/model pinball；
- negative baseline/model Brier；
- q25 coverage；
- rollout 时间。

若 pilot 中没有任何正 margin controller，先审计 candidate family；不得直接扩大数据掩盖问题。

### 10.2 3-seed gate

正式规模：

- fit starts `>=32`
- calibration starts `>=12`
- controller configs `16/32`，在 launch 前锁定
- anchors per start `3`
- seeds `41/42/44`

validation deployable guard 保持：

- mean paired margin `>=0.001`
- q25 margin `>=0`
- minimum start margin `>=-0.01`
- negative starts `<=1`

成功条件：

- 至少 `2/3` seeds 在 validation 选择 dynamic controller；
- 至少 `2/3` seeds 在 final 优于 static；
- teacher `3/3` 保持正 margin；
- final conservative mean margin 为正。

失败条件：

- data/model gate 任一 seed 失败且无法通过协议级修复解决；
- 3-seed final `<2/3`；
- 选中的 controller 出现 hard constraint violation；
- risk model 不优于常数 baseline。

失败后停止新增局部 controller/threshold 变体，重新评估 action abstraction 和 decision horizon。

### 10.3 扩展

仅在 3-seed gate 通过后：

- n=5：至少 `4/5` final wins；
- n=15：至少 `12/15` final wins；
- 同时报告 fallback 数、paired q25、worst start 和 sign test。

## 11. 预计成本

工程实现与测试：约 `1-2` 个 Codex 工作日。

服务器实验：

- seed41 pilot：约 `1-3` 小时；
- 3-seed 正式数据收集和 gate：约 `8-16` 小时 wall-clock，可三 seed 并行；
- n=5/n=15 仅在前门控通过后估算。

实际时间以 seed41 collector benchmark 为准，不预先承诺固定运行时。

## 12. 需要复用的现有工作

直接复用：

- chronological split 和 `event_transport_rich` start selection；
- learned event/continuous forecasters；
- feasible mask enumeration 和 projector；
- validation-selected static comparator；
- proxy-MPC controller；
- paired objective evaluation；
- risk-band selection；
- aggregate/manifest/CHANGELOG 体系。

新增：

- train internal split；
- train anchor bank；
- paired window-risk dataset；
- grouped mean/quantile/risk models；
- conformal lower bound；
- window-level outer controller policy。

明确不复用：

- validation-selected anchor 作为训练 label anchor；
- step-level复制 window label；
- isolated-step sequence mutation；
- truth-future context；
- validation negative starts 用于特征开发。
