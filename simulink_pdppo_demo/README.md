# PD-PPO Simulink 固化模型演示工程

本文件夹可以直接打包发送给别人使用。工程不依赖外部论文仓库，也不需要重新训练模型。

演示目标是展示一个完整链路：

```text
完整传感器系统数据
-> Simulink 中的固化 PD-PPO 调度模型
-> 输出每个通道的开关状态
-> 生成残缺观测数据
-> Python 中的固化重构模型补全数据
-> 与原始完整数据对比误差
```

## 1. 文件夹内容

```text
simulink_pdppo_demo/
├── START_HERE.m                         MATLAB 一键入口
├── README.md                            本操作说明
├── requirements.txt                     Python 依赖
├── matlab/                              Simulink 建模和仿真脚本
├── python/                              Python 补全和对比脚本
├── frozen_policy/
│   ├── pdppo_demo_policy.mat            Simulink 使用的固化调度模型
│   └── pdppo_demo_policy.json           同一模型的人类可读版本
├── frozen_model/
│   └── channel_reconstructor.npz        Python 使用的固化通道重构模型
├── data/                                MATLAB 运行后生成演示输入
└── outputs/                             MATLAB/Python 运行后生成结果
```

## 2. 软件要求

MATLAB 端：

- MATLAB
- Simulink

Python 端：

- Python 3.9 或更新版本
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`

安装 Python 依赖：

```bash
cd simulink_pdppo_demo
python -m pip install -r requirements.txt
```

如果电脑里同时有多个 Python，可能需要使用：

```bash
python3 -m pip install -r requirements.txt
```

## 3. 第一步：运行 Simulink 调度仿真

打开 MATLAB。

在 MATLAB 的 Current Folder 中进入本文件夹：

```text
simulink_pdppo_demo
```

然后在 MATLAB 命令行执行：

```matlab
START_HERE
```

脚本会自动完成以下操作：

1. 加载固化调度模型：`frozen_policy/pdppo_demo_policy.mat`
2. 生成 240 个时间步的演示输入：`data/demo_inputs.mat`
3. 自动创建 Simulink 模型：`pdppo_scheduler_demo.slx`
4. 运行 Simulink 仿真
5. 导出调度结果和残缺观测数据
6. 生成调度可视化图片

运行完成后，应该能看到这些新文件：

```text
pdppo_scheduler_demo.slx
data/demo_inputs.mat
outputs/full_reference.csv
outputs/channel_mask.csv
outputs/simulink_partial_observations.csv
outputs/demo_schedule.png
```

### Simulink 模型结构

自动生成的模型包含三个核心部分：

```text
Demo Observation
    -> Frozen PD-PPO Scheduler
    -> Scheduled Mask

Demo Observation + Scheduled Mask
    -> Apply Channel Mask
    -> Partial Observation
```

含义：

- `Demo Observation` 是完整传感器系统状态输入。
- `Frozen PD-PPO Scheduler` 是固化的调度模型。
- `Scheduled Mask` 是 8 个通道的开关状态，1 表示开启，0 表示关闭。
- `Apply Channel Mask` 根据开关状态把未开启通道置为 `NaN`。
- `Partial Observation` 是仿真得到的残缺数据。

## 4. 第二步：运行 Python 补全和对比

Simulink 运行完成后，在终端中执行：

```bash
cd simulink_pdppo_demo
python python/impute_and_compare.py
```

如果系统使用 `python3`：

```bash
python3 python/impute_and_compare.py
```

Python 脚本会：

1. 读取 `outputs/full_reference.csv`
2. 读取 `outputs/channel_mask.csv`
3. 读取 `outputs/simulink_partial_observations.csv`
4. 加载固化补全模型：`frozen_model/channel_reconstructor.npz`
5. 对残缺数据进行补全
6. 与完整数据对比误差
7. 同时输出线性插值 baseline，方便对照

运行完成后，应该生成：

```text
outputs/imputation_metrics.csv
outputs/python_imputed_observations.csv
outputs/imputation_comparison.png
```

## 5. 输出文件怎么看

### `outputs/channel_mask.csv`

这是调度算法输出的 8 路开关状态。

每一列对应一个传感器系统通道：

```text
weather
pyranometer
surface_ir
highres_wind
thermo_hygro
particle_counter
laser
fc4_flux
```

数值含义：

- `1`：该时间步开启该通道
- `0`：该时间步关闭该通道

这份文件用于展示调度算法本身。

### `outputs/simulink_partial_observations.csv`

这是 Simulink 根据调度结果生成的残缺观测。

- 开启通道保留原始数值
- 关闭通道被置为 `NaN`

这份文件用于展示“调度之后传给后续算法的数据是什么样子”。

### `outputs/imputation_metrics.csv`

这是补全结果与完整原始数据之间的误差。

主要列：

- `observed_rate`：该通道被开启观测的比例
- `frozen_model_mae`：固化补全模型的平均绝对误差
- `frozen_model_rmse`：固化补全模型的均方根误差
- `linear_interp_mae`：线性插值 baseline 的平均绝对误差
- `linear_interp_rmse`：线性插值 baseline 的均方根误差

这份文件用于展示“调度造成的数据缺失是否还能被下游模型恢复”。

### `outputs/demo_schedule.png`

这是调度结果图，展示事件状态和 8 路通道开关随时间变化。

### `outputs/imputation_comparison.png`

这是补全对比图，展示：

- 完整原始数据
- Simulink 输出的残缺观测点
- 固化模型补全曲线
- 线性插值 baseline

## 6. 演示时应该怎么讲

可以这样介绍：

> 这个演示不是训练过程，而是部署阶段的固化模型推理。完整数据先进入
> Simulink，固化的 PD-PPO 调度模型在功率预算下决定哪些传感器系统通道开启。
> Simulink 根据调度结果生成残缺观测。随后 Python 调用另一个固化重构模型，
> 对残缺观测进行补全，并与完整数据比较误差。

不要这样说：

> Simulink 在训练 PD-PPO。

也不要这样说：

> 这是论文完整实验复现。

准确说法是：

> 这是一个固化模型的课程演示，用于展示传感器系统调度、功率受限采集、
> 残缺数据生成和后续补全评估的完整流程。

## 7. 固化模型说明

### 固化调度模型

文件：

```text
frozen_policy/pdppo_demo_policy.mat
```

它包含：

- 8 个传感器系统通道
- 每个通道的功耗
- 总功率预算
- 最小开启时间
- 通道占用比例上限
- 固定评分权重矩阵

Simulink 中的 `Frozen PD-PPO Scheduler` 会加载这个文件，并输出可执行的通道子集。

### 固化补全模型

文件：

```text
frozen_model/channel_reconstructor.npz
```

它包含固定的通道耦合系数和时间平滑参数。Python 脚本只加载参数并做推理，
不会训练模型。

## 8. 常见问题

### 问题 1：MATLAB 提示找不到 `START_HERE`

原因：MATLAB 当前文件夹不在 `simulink_pdppo_demo`。

处理：

1. 在 MATLAB 左侧 Current Folder 进入 `simulink_pdppo_demo`
2. 再执行：

```matlab
START_HERE
```

### 问题 2：MATLAB 提示找不到 `pdppo_demo_policy.mat`

正常情况下该文件已经在：

```text
frozen_policy/pdppo_demo_policy.mat
```

如果被误删，可以在终端执行：

```bash
cd simulink_pdppo_demo
python python/export_frozen_demo_models.py
```

然后重新运行 MATLAB：

```matlab
START_HERE
```

### 问题 3：Python 提示缺少 `pandas`、`numpy`、`matplotlib` 或 `scipy`

安装依赖：

```bash
cd simulink_pdppo_demo
python -m pip install -r requirements.txt
```

### 问题 4：Python 提示找不到 `outputs/full_reference.csv`

原因：还没有先运行 Simulink。

处理：

1. 先在 MATLAB 中执行 `START_HERE`
2. 确认 `outputs/` 下生成了 CSV
3. 再运行 Python 脚本

### 问题 5：Simulink 模型已经打开，再次运行报错

关闭当前打开的 `pdppo_scheduler_demo.slx`，然后重新执行：

```matlab
START_HERE
```

脚本会重新创建模型。

## 9. 如何替换成自己的数据

最简单的替换位置是：

```text
matlab/generate_demo_inputs.m
```

当前脚本会自动生成 240 个时间步的演示数据。若要接入自己的完整数据，需要把
自己的数据整理为 17 维输入：

```text
1--8   归一化后的 8 个通道状态
9--16  8 个通道的 AoI 或缺失时长
17     事件状态，0 表示普通时段，1 表示事件时段
```

并输出 MATLAB `timeseries`：

```matlab
pdppo_demo_obs = timeseries(obs, t);
```

其中：

- `obs` 的大小是 `时间步数 x 17`
- `t` 是时间索引列向量

对于课程演示，不建议改模型结构，只替换输入数据即可。

## 10. 如何重新导出固化模型

通常不需要执行这一步。只有当固化模型文件被删除时才需要。

```bash
cd simulink_pdppo_demo
python python/export_frozen_demo_models.py
```

该命令会重新生成：

```text
frozen_policy/pdppo_demo_policy.mat
frozen_policy/pdppo_demo_policy.json
frozen_model/channel_reconstructor.npz
```

## 11. 打包发送

直接打包整个文件夹即可：

```text
simulink_pdppo_demo
```

不要只发送 `matlab/` 或 `python/` 子文件夹，因为固化模型文件在：

```text
frozen_policy/
frozen_model/
```

推荐发送前确认这三个文件存在：

```text
frozen_policy/pdppo_demo_policy.mat
frozen_policy/pdppo_demo_policy.json
frozen_model/channel_reconstructor.npz
```
