# 输出结果目录

本目录用于存放 MATLAB/Simulink 和 Python 脚本生成的结果。

运行 MATLAB：

```matlab
START_HERE
```

会生成：

```text
full_reference.csv
channel_mask.csv
simulink_partial_observations.csv
demo_schedule.png
```

继续运行 Python：

```bash
python python/impute_and_compare.py
```

会生成：

```text
imputation_metrics.csv
python_imputed_observations.csv
imputation_comparison.png
```

其中：

- `channel_mask.csv` 展示调度模型输出的 8 路通道开关；
- `simulink_partial_observations.csv` 展示调度后形成的残缺观测；
- `imputation_metrics.csv` 展示补全结果与完整数据之间的误差；
- 两张 PNG 图片用于课程展示。
