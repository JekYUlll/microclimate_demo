# v1 实验方向、参数调整与结果报告

日期：2026-06-06

## 1. 执行结论

当前尚未取得可称为“算法突破”的 deployable 结果，但已经完成三项关键工作：

1. 场景已从“静态 direct stack 长期开启即可解决任务”修正为具有稳定动态价值的复杂场景。
2. MPC teacher 的动态价值已被严格确认，不是个别 seed 或单一指标造成的假象。
3. deployable 路线已从模仿、阈值、检索、序列压缩推进到因果 forecast planner；最新 proxy-MPC 已使两个 seed 的平均验证 margin 转正，但仍未解决窗口下行风险。

因此，当前不是场景失效或 teacher 失效，而是 deployable policy 无法稳定复现 teacher 的多传感器时序互补行为。最新结果属于方向性进展，不满足论文主 claim。

## 2. 当前目标与判定标准

核心目标是在严格时间划分下，用部署时仅可获得的因果信息，学习一个受功耗、启动峰值、能量、预热和切换约束的 forecast-aware scheduler，并稳定优于 validation-selected static。

当前主 claim 门槛：

- deployable 在 final-test 至少 `4/5` seeds 优于 static；
- paired margin 均值为正；
- MPC teacher 至少 `4/5` seeds 优于同一 static，证明场景存在动态调度价值；
- final-test 不参与策略、超参数或阈值选择；
- 无通过验证门控的动态策略时，计为 static fallback，margin 为 `0`，不能从均值中删除。

主目标函数为：

`task_composite = frozen forecast-oracle loss + task_error_weight * transport task error`

当前 task columns：

- `snow_mass_flux_kg_m2_s`
- `snow_particle_mean_diameter_mm`
- `snow_particle_mean_velocity_ms`

## 3. 协议和目标函数调整

| 调整 | 原设置 | 新设置 | 结果 |
|---|---|---|---|
| 数据协议 | 容易混用训练、验证选择和最终回放 | 严格 chronological train/validation/final | 已固定 |
| static 对照 | 弱或固定指定 static | validation-selected feasible static | 已固定 |
| deployable 选择 | 仅看 validation objective | paired static margin | 已固定 |
| 风险选择 | mean margin | mean、median、q25、min、negative starts | 已固定 |
| fallback 统计 | 无 deployable 可能被记为 NaN | static fallback 记为 margin `0` | 已修复 |
| 目标函数 | oracle loss 主导 | task-composite | teacher 动态价值显著增强 |
| task weight | `0.20` | 测试 `0.25`，最终主用 `0.30` | seed41 改善，但证明单一权重不能解决所有 seed |
| validation starts | 常用 `4` | dense validation `12` | 减少偶然选择，但未解决跨制度迁移 |
| calibration horizon | 局部 `16/32` steps | full rollout `256` steps | 修复局部窗口与最终选择单位不一致 |

重要结果：

- `task_error_weight=0.20` 时，contextual-duty 虽改善 MAE/RMSE/DTW 和 task error，仍被 oracle loss 压过。
- seed41 在 `w=0.30` 时通过：deployable `1.344876`，static `1.373655`。
- seed42/44 的 break-even weight 分别要求 `<0.1441`、`<0.1416`，而 seed41 要求 `>0.1848`。因此继续调一个全局权重不是解决方案。

## 4. 场景重标定

### 4.1 旧场景问题

旧 `B=1.20` 场景的 15-seed 审计发现：

- selected static 平均功耗 `1.1619/1.20`；
- `met_station_core`、`laser_disdrometer`、`fc4_flux` 长期开启；
- top-10 static 中 laser 出现率 `97.3%`；
- `snow_particle_counter` 出现率 `0%`。

这意味着 static 可以持续保留直接 snow sensing stack，动态策略只是在廉价 context sensors 之间切换。该场景不足以激活目标算法的难点。

### 4.2 v5 constraint-active

主要调整：

- 强化 direct laser stack 的持续能量代价；
- 保留 proxy/particle/flux sensing 的可行组合；
- 比较 energy capacity `70` 与 `90`。

结构门控结果：

- v5/e70：通过，feasible masks `109`，laser duty over proxy `0.815897`；
- v5/e90：失败，laser duty over proxy `0.985734`，能量约束过松。

v5/e70 teacher 在 seeds `41/42/44` 上 `3/3` 优于 static，但 deployable 仍为 `1/3`。进一步发现 seed42/44 的 static 仍可保留过强的 `core+laser` 信息，因此停止在 v5 上增加 selector。

### 4.3 v6 complex-static-break

当前 sensor config：

| Sensor | power | startup peak | warmup | 主要作用 |
|---|---:|---:|---:|---|
| core | 0.18 | 0.22 | 0 | wind/temperature/humidity/pressure |
| radiometer | 0.06 | 0.08 | 0 | radiation context |
| surface IR | 0.08 | 0.10 | 1 | snow surface temperature |
| ultrasonic wind | 0.09 | 0.12 | 0 | high-quality wind |
| thermo-hygro | 0.09 | 0.12 | 0 | high-quality temperature/humidity |
| SPC | 0.28 | 0.40 | 2 | particle diameter/velocity proxy |
| laser | 1.18 | 1.50 | 3 | high-quality particle sensing |
| fc4 | 0.30 | 0.36 | 1 | snow mass flux |

v6 初始结构标定使用：

- budget `1.36`
- startup peak `1.75`
- capacity/initial energy `70`
- harvest `0.80`
- reserve `20`
- task weight `0.30`

随后加入 `event_transport_rich` 非重叠起点选择，不只要求事件率高，还要求 particle diameter/velocity 变化充分。

正式 static/teacher gate：

- teacher `3/3` 优于 static；
- mean teacher margin `+0.098113`，minimum `+0.048388`；
- task-error margins：`+0.1287/+0.2380/+0.2148`；
- static laser duty 为 `0`；
- teacher 使用 `17/20/18` 个 unique masks；
- teacher fc4 duty 约 `0.801`，SPC duty 约 `0.605`，laser duty `0`。

结论：v6 成功消除了 `core+laser` 静态捷径。当前动态价值来自 SPC、fc4 和 context sensors 的时序互补，不应再描述为 selective-laser 机制。

### 4.4 后期统一 formal diagnostic 协议

后期 planner/interface 诊断统一使用：

- budget `1.20`
- startup peak `1.60`
- capacity/initial energy `180`
- harvest `0.92`
- reserve `20`
- train/static/eval steps `128/256/256`
- train/static/eval rollouts `12/12/12`
- seeds `41/42/44`
- selection `event_transport_rich`

这组数值来自 split-protocol energy-account 输入，与最初 `B=1.36/e70` 场景标定属于不同实验层。它们共享 v6 sensor config 和 transport-rich 机制，但 objective 绝对值不可直接横向比较。

## 5. 已探索算法方向

### 5.1 行为模仿与 residual/value 路线

| 路线 | 关键调整 | 最佳结果 | 结论 |
|---|---|---|---|
| BC/DAgger BC | teacher action labels，support top-k | 标签可拟合，严格 static comparator 下失败 | 关闭 |
| Mask/anchor-mask BC | 从 action id 改为 per-sensor mask | 未稳定通过 | 关闭 |
| Residual BC | static 为默认，仅学习 deviation | `1/5` | 关闭 |
| Value-residual | 学习 candidate cost 后相对 static 偏离 | n=5 `4/5`，mean `+0.002213` | 小样本可行，幅度弱 |
| No-DAgger | 去掉 DAgger | 与 value-residual 类似 | DAgger 不是核心机制 |
| Anchor-advantage | 直接学习相对 static advantage | corrected run `0/5` | 关闭 |

### 5.2 Event threshold、验证风险和扩展

| 路线 | 参数/变化 | 结果 | 结论 |
|---|---|---|---|
| Event-threshold | learned multi-horizon event forecast | n=5 `4/5`，mean `+0.003758` | 初步可行 |
| Budget matrix | `B=1.05/1.20/1.35` | `1/5, 4/5, 1/5` | 无跨预算鲁棒性 |
| n=15 extension | seeds 扩展 | `10/15`，teacher `14/15` | 未达 `12/15` |
| dense validation | validation starts `4 -> 12` | n=5 `4/5`，mean `+0.007063` | 小样本改善 |
| static-margin risk | 加 mean/median/q25/negative starts | n=5 `4/5` | seed44 出现大负值 |
| positive-center | mean、median 必须为正，否则 static fallback | n=5 `4/5`，conservative mean `+0.011105` | 选择语义更干净 |
| extension | seeds `46--51` | combined `7/11` | 强 claim 数学失败 |
| prospective risk-band | q25 `>=-0.005`，negative starts `<=4` | seeds52--55 `1/4` | 验证统计不能可靠预测 final |

这条路线的重要结论不是 event forecast 无效，而是 validation regime 到 final regime 的排序相关性太弱：

- static validation-final Spearman `0.204`
- dynamic margin validation-final Spearman `0.280`
- q25-final Spearman `0.343`

### 5.3 Teacher rate、contextual duty 和序列模仿

| 路线 | 核心思想 | 结果 | 结论 |
|---|---|---|---|
| Teacher-rate | 匹配 teacher 平均 duty | negative-center seeds 未恢复 | 平均 duty 不足 |
| Contextual-duty | 状态条件 active probability + freshness feedback | 部分 seed 改善，整体失败 | 时序表达仍不足 |
| Teacher-cycle | 简单重放 teacher 周期 | 不稳定，seed42 validation `2.036201` | 关闭 |
| Sequence-mask GRU | 模仿 teacher mask sequence | exact match `0.996--1.000`，仍未被选中 | label fit 不等于 closed-loop value |
| Recurrent value | GRU candidate cost | `1/3`，部分退化为 static clone | 关闭 |
| Rank recurrent | rows `512 -> 1536`，rank loss `0.5` | accuracy 提升，仍 `1/3` | 数据量不是主因 |
| Cost-DAgger | rows `1536 -> 3072`，一次 on-policy pass | accuracy 提升至 `0.47--0.59`，结果不变 | covariate shift 不是唯一问题 |

### 5.4 Option、runtime guard 和朴素动态 baseline

| 路线 | 关键参数 | 结果 | 结论 |
|---|---|---|---|
| Option planner | learned event、freshness、SOC、dwell/cooldown | `1/3`，seed44 `+0.007593` | 有单点真实信号 |
| Rate-balanced option | rate balance `0/1/3` | `0/3` | duty 平衡无效 |
| Zero-negative startguard | negative validation starts `=0` | `0/3` | 过于保守且不稳定 |
| Runtime-risk guard | per-window threshold | paired rerun `0/3`，mean `-0.003977` | validation 正仍可能 final 负 |
| Dense runtime-risk | thresholds `0.8/1.0/1.2`，windows `4/8/16` | `0/3`，全部 fallback | 关闭阈值调优 |
| cyclic/dwell baseline | dwell `2/4/8/16`，preserve warming | `0/3`，mean `-0.024692` | 盲目轮询不能解释 teacher 优势 |

切换模式审计：

- static switch rate `0%`；
- event-threshold 约 `7.16%`；
- option planner 约 `23.92%`；
- MPC teacher 约 `70.45%`；
- teacher 三个以上传感器同时切换约 `15.97%`，学生低于 `1%`。

当前学生仍是“固定 core + 少量辅助通道”，teacher 才真正实现多传感器时序混合。

### 5.5 Teacher cost、trajectory 和 improvement gate

| 路线 | 结果 | 失败原因 |
|---|---|---|
| cost-KNN | `0/3`；best validation mean `-0.020~-0.046` | one-step cost memory 不足 |
| macro-option snippet | `0/3`；动态 row validation 已为负 | 相似片段重放不稳定 |
| teacher-improvement gate | `0/3` | first-action improvement label 与 sequence-level teacher value 不一致 |
| dense always-dynamic macro | `0/3`；所有 seed best mean < 0 | 不是 entry gate 过保守 |

随后进行 window-level teacher audit：

- train/validation/final 合计 `60/60` windows，teacher 全部优于 static；
- validation mean margins：`+0.079045/+0.069905/+0.096935`；
- final mean margins：`+0.076549/+0.072852/+0.103971`。

这排除了“teacher 只在 final 偶然有效”的解释。

### 5.6 Sequence/window outcome 和 privileged-context 诊断

| 路线 | 参数调整 | 结果 |
|---|---|---|
| Sequence-value | sequence bank `369--380`，约 `1900` rows | `0/3`；seed44 mean 正但 q25 负 |
| Full-bank | top-k `128 -> 512`；threshold 扩至 `0.5` | `0/3` |
| Oracle event context | 用 truth-future event 替代 learned event | `1/3`，mean `-0.004845` |
| Oracle continuous regime | 加 wind/surface/flux/diameter/velocity future | `1/3`，mean `+0.001014` |
| One-step anchor advantage | support `6/12`，privileged context | `0/3` |
| Window eligibility KNN | whole-window paired margin target | `0/3`，mean 多为正但 tails 不安全 |
| Window gate + macro executor | 更换 inner executor | `0/3`，仍为 negative q25/negative starts |

结论：缺失未来事件信息不是主因；即使使用 privileged continuous context，现有 sequence retrieval/compression 仍不能覆盖动态价值。

### 5.7 Learned world model、self-rollout 和 digital twin

| 路线 | 参数/数据 | 结果 |
|---|---|---|
| Rollout-value | raw action cost、transition model、depth `2`、beam `4` | `0/3`，validation 全负 |
| Self-distribution | teacher + planner own states；rows 约翻倍 | `0/3`，validation 更差 |
| Executed-step twin | static/teacher/random executed outcomes；每 seed `4608+4608` rows | `0/3` |
| Augmented sequence verifier | learned event + continuous；`805--829` sequences，约 `7.6k` rows | `0/3` |
| Multi-candidate window margin | option/macro/rate candidates | `0/3` |
| Full-rollout calibration | calibration `16/32 -> 256` steps | `0/3`，修复协议但 candidate family 仍弱 |

关键诊断：

- transition/cost loss 降低并未带来 validation improvement；
- 自策略分布采样也未解决问题；
- 当前模型优化的是 one-step absolute cost/feature delta，而 claim 需要 full-window static-anchor margin 和 tail risk。

## 6. 最近两代 forecast planner

### 6.1 Causal forecast-utility planner

输入：

- learned event forecast；
- learned continuous forecast；
- sensor-variable coverage；
- freshness、teacher-rate deficit；
- power、switch、dwell、SOC guard。

smoke：utility `10.040601`，static `10.067309`，teacher `10.008395`。

正式 3-seed：

| seed | best mean | q25 | min | negative starts |
|---:|---:|---:|---:|---:|
| 41 | -0.041270 | -0.100181 | -0.236599 | 8 |
| 42 | -0.056260 | -0.066191 | -0.129740 | 12 |
| 44 | -0.000786 | -0.024834 | -0.159545 | 7 |

结果：deployable `0/3`，teacher `3/3`。手工 scalar utility 明显错配，路线关闭。

### 6.2 Task-only proxy-MPC

相对 utility planner 的主要修正：

- 从单步 utility 改为 short-horizon beam search；
- continuous policy context 从 5 个通用 weather/transport columns 收缩为 3 个 task-transport columns；
- 显式维护 column-age freshness；
- 规划目标加入 static-anchor proxy improvement；
- 只在 teacher-supported feasible masks 中搜索。

forecast 模型：

- event lookback `8`，hidden `128`，epochs `40`；
- continuous lookback `8`，hidden `128`，epochs `40`；
- continuous training targets：
  `wind_speed`、`surface temperature`、`flux`、`diameter`、`velocity`；
- planner 实际使用：
  `flux/diameter/velocity`；
- scales：`1e-4/0.2/5.0`。

搜索空间：

| 参数 | 搜索值 |
|---|---|
| support top-k | `16` |
| event weight | `0.5/1.5` |
| magnitude weight | `1.0` |
| variability weight | `0.5` |
| freshness weight | `0.25` |
| target-rate weight | `0.0/0.5` |
| power/switch weight | `0.03/0.03` |
| dwell | `1/2` |
| planning depth | `2/3` |
| beam width | `4` |
| max branch | `8` |
| age weight | `0.25/0.75` |
| aggregation | `max` |
| anchor improvement | `0.0` |

风险门控：

- mean margin `>=0.001`
- minimum start margin `>=-0.01`
- negative starts `<=1`
- positive center required
- q25 margin `>=0`

smoke：

- proxy-MPC `10.122692`
- static `10.138463`
- teacher `10.115600`

正式 3-seed：

| seed | best mean margin | q25 | minimum | negative starts | selected |
|---:|---:|---:|---:|---:|---|
| 41 | -0.001605 | -0.038449 | -0.054134 | 7 | no |
| 42 | +0.005049 | -0.001126 | -0.032500 | 3 | no |
| 44 | +0.008540 | -0.003628 | -0.044961 | 4 | no |

最终 aggregate：

- deployable `0/3`
- teacher `3/3`
- teacher mean margin `+0.113506`
- claim fail

与 utility planner 相比，proxy-MPC 将 seed42/44 的最佳 mean margin 从负值推到正值，seed41 也接近零。这是最近最明确的方向性改善。但三个 seed 的 q25、minimum 和 negative starts 均不满足门控，说明平均规划质量已接近可用，主要剩余问题是窗口级 downside risk。

## 7. 已确认和未确认的 claim

已确认：

1. v6/event-transport 场景存在真实动态价值。
2. teacher 优势同时来自 oracle loss 和 transport task error，不是单指标造假。
3. teacher 在已审计 `60/60` train/validation/final windows 均优于 static。
4. warmup-aware cyclic/dwell 等朴素动态 baseline 不能替代 teacher。
5. learned event/continuous forecast 基础设施是 split-compliant、可部署的。
6. proxy-MPC 相比 scalar utility 在平均 margin 上有实质改善。

尚未确认：

1. deployable scheduler 稳定优于 validation-selected static。
2. `4/5` 或更大 seed 规模的主结果。
3. 跨预算、跨扰动和跨场景泛化。
4. forecast-aware planner 的优势来自可解释的 task-aware planning，而非个别参数组合。
5. 实际部署要求下的切换寿命、通信代价和硬件磨损优势。

## 8. 当前瓶颈

当前主要瓶颈不是：

- static baseline 选错；
- teacher 没有动态价值；
- event 数量不足；
- 训练样本数量太少；
- 单纯 teacher-state covariate shift；
- 缺失 perfect future event context；
- calibration horizon 不一致。

当前瓶颈是：

1. 学习目标仍不能准确估计 full-window relative margin。
2. planner 的平均行为接近有效，但少数窗口损失较大。
3. 现有模型没有显式学习 lower quantile、negative-start probability 或 CVaR。
4. 多步候选生成仍由手工 proxy score 主导，尚未从 paired rollout outcomes 学到真实风险。

## 9. 下一步路线

停止继续调 event/freshness/power 的手工权重。下一实现应为：

1. 在 train split 对 static anchor 和候选 planner 做 paired full-window self-rollout。
2. 直接学习：
   - expected static-anchor margin；
   - q25/CVaR margin；
   - probability of negative window；
   - constraint/warmup feasibility。
3. 用 learned mean-risk model 在 receding horizon 中选择候选 sequence；风险不通过时执行 static anchor。
4. 训练数据必须包含当前策略自身 rollout，避免只在 teacher 或静态分布上拟合。
5. 首先只跑 seeds `41/42/44`；只有 q25 和 negative-start 数显著改善才扩展到 `n=5`。

## 10. 总体判断

没有进入“场景无动态价值”的死胡同；teacher 证据反而非常稳定。也没有获得可投稿主 claim 所需的 deployable 突破。

当前最乐观、同时最客观的判断是：

- 方向已经从平均性能问题收敛为窗口 downside-risk 问题；
- proxy-MPC 是第一条在多数 seed 上获得正 mean validation margin 的纯因果 multi-step planner；
- 距离成功仍差一个能够直接学习 full-window relative outcome distribution 的风险模型；
- 若下一代 mean-risk self-rollout planner 仍在 seed41/42/44 上 `0/3`，应重新评估 action abstraction、teacher interface 和任务时间尺度，而不是继续做局部参数搜索。
