# Python 固化模型后处理

本目录包含两个脚本：

```text
impute_and_compare.py          读取 Simulink 输出，调用固化补全模型并计算误差
export_frozen_demo_models.py   重新导出固化调度模型和固化补全模型
```

正常演示只需要运行：

```bash
python python/impute_and_compare.py
```

运行前必须先在 MATLAB/Simulink 中执行 `START_HERE`，因为 Python 脚本需要读取
`outputs/` 目录下的 CSV 文件。

Python 脚本会完成：

1. 读取完整参考数据；
2. 读取 Simulink 生成的残缺观测；
3. 读取通道开关 mask；
4. 加载 `frozen_model/channel_reconstructor.npz`；
5. 输出固化模型补全结果；
6. 同时输出线性插值 baseline；
7. 生成误差表和对比图。

如果固化模型文件被删除，可以重新生成：

```bash
python python/export_frozen_demo_models.py
```
