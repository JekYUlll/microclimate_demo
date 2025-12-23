#!/home/zhangzhuyu/.conda/envs/darts/bin/python

import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"

# 设置NCCL环境变量，优化多GPU通信
os.environ["NCCL_TIMEOUT"] = "1800"  # 30分钟超时
os.environ["NCCL_IB_TIMEOUT"] = "22"  # InfiniBand超时
os.environ["NCCL_DEBUG"] = "INFO"  # 调试信息（可选）

import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import pickle

from darts import TimeSeries
from darts.models import TCNModel
from sklearn.preprocessing import MinMaxScaler
from darts.metrics import mae, mse

# 配置
FEATURES = ["temperature", "humidity", "wind_speed"]
station = "长城站"
os.makedirs("plots", exist_ok=True)

# ============================================
# 1. 从数据库读取数据
# ============================================
print("=== 读取数据 ===")
db_config = {
    "host": "124.220.77.63",
    "user": "horeb",
    "password": "ZZYzzy4771430///",
    "database": "antarctic_data",
    "charset": "utf8mb4",
}

conn = mysql.connector.connect(**db_config)
query = """
SELECT station, record_time, temperature, humidity, wind_speed
FROM weather_data
WHERE station = %s
ORDER BY record_time ASC
"""
df = pd.read_sql(query, conn, params=(station,))
conn.close()

print(f"数据形状: {df.shape}")
print(f"时间范围: {df['record_time'].min()} 到 {df['record_time'].max()}")

# ============================================
# 2. 数据预处理
# ============================================
print("\n=== 数据预处理 ===")
df["record_time"] = pd.to_datetime(df["record_time"]).dt.tz_localize(None)
df = df.sort_values("record_time").reset_index(drop=True)

# 数值转换
df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")

# 重采样到5分钟频率
df_train = df.set_index("record_time")[FEATURES].resample("5min").mean()
df_train = df_train.interpolate(method="time").reset_index()

# 归一化
scaler = MinMaxScaler()
df_train[FEATURES] = scaler.fit_transform(df_train[FEATURES])

# 创建TimeSeries
series = TimeSeries.from_dataframe(
    df_train,
    time_col="record_time",
    value_cols=FEATURES,
    fill_missing_dates=True,
    freq="5min",
)
series = series.astype(np.float32)

print(f"TimeSeries长度: {len(series)}")

# ============================================
# 3. 划分训练集和验证集
# ============================================
print("\n=== 划分数据集 ===")
val_size = min(200, len(series) // 20)
train, val = series[:-val_size], series[-val_size:]

print(f"训练集长度: {len(train)}")
print(f"验证集长度: {len(val)}")
print(f"训练集时间范围: {train.start_time()} 到 {train.end_time()}")
print(f"验证集时间范围: {val.start_time()} 到 {val.end_time()}")

# ============================================
# 4. 训练TCN模型（多GPU）
# ============================================
print("\n=== 训练TCN模型（使用6块GPU） ===")
model = TCNModel(
    input_chunk_length=24,  # 输入窗口长度
    output_chunk_length=6,  # 输出窗口长度
    num_layers=2,  # TCN层数
    num_filters=64,  # 卷积核数量
    kernel_size=3,  # 卷积核大小
    dropout=0.1,  # Dropout率
    n_epochs=50,  # 训练轮数
    batch_size=128,  # 批次大小
    optimizer_kwargs={"lr": 0.001},
    random_state=42,
    force_reset=True,
    pl_trainer_kwargs={
        "accelerator": "gpu",
        "devices": [0, 1, 2, 3, 4, 5],
        "precision": 32,  # 全精度训练
        "strategy": "ddp",  # 数据并行策略
    },
)

print("开始训练...")
model.fit(train, val_series=val, verbose=True)

# ============================================
# 5. 保存模型
# ============================================
print("\n=== 保存模型 ===")
model.save("tcn_model.pt")
print("✅ 模型已保存到 tcn_model.pt")

with open("tcn_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ Scaler已保存到 tcn_scaler.pkl")

training_info = {
    "station": station,
    "train_length": len(train),
    "val_length": len(val),
    "freq": "5min",
    "input_chunk_length": 24,
    "output_chunk_length": 6,
    "features": FEATURES,
}
with open("tcn_training_info.pkl", "wb") as f:
    pickle.dump(training_info, f)
print("✅ 训练信息已保存到 tcn_training_info.pkl")

# ============================================
# 6. 预测和评估（切换到单GPU模式）
# ============================================
print("\n=== 预测和评估 ===")
# 重新加载模型用于预测（单GPU模式，避免多GPU预测问题）
print("重新加载模型用于预测...")
try:
    model_for_prediction = TCNModel.load("tcn_model.pt")
    # 设置单GPU预测配置
    if hasattr(model_for_prediction, "trainer_params"):
        model_for_prediction.trainer_params = {
            **model_for_prediction.trainer_params,
            "accelerator": "gpu",
            "devices": [0],  # 只使用第一个GPU进行预测
            "strategy": "auto",
        }
    print("✅ 模型重新加载成功，已切换到单GPU模式")
    forecast = model_for_prediction.predict(n=len(val), series=train)
except Exception as e:
    print(f"⚠️ 重新加载模型失败，使用原始模型: {e}")
    import traceback

    traceback.print_exc()
    forecast = model.predict(n=len(val), series=train)

print(f"预测完成，长度: {len(forecast)}")
print(f"预测时间范围: {forecast.start_time()} 到 {forecast.end_time()}")

# 计算误差指标
print("\n=== 预测误差统计 ===")
for col in FEATURES:
    actual = val.univariate_component(col)
    predicted = forecast.univariate_component(col)
    print(f"{col}: MAE={mae(actual, predicted):.4f}, MSE={mse(actual, predicted):.4f}")

# ============================================
# 7. 可视化
# ============================================
print("\n=== 生成可视化图表 ===")
plt.figure(figsize=(15, 8))
for i, col in enumerate(FEATURES):
    plt.subplot(len(FEATURES), 1, i + 1)
    val_series = val.univariate_component(col)
    forecast_series = forecast.univariate_component(col)

    val_series.plot(label=f"实际值", color="blue", linewidth=2)
    forecast_series.plot(label=f"预测值", color="red", linewidth=2)

    plt.title(f"{col} - 实际值 vs 预测值")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig(f"plots/{station}_tcn_prediction.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ 已保存: plots/{station}_tcn_prediction.png")

print("\n✅ TCN模型训练完成！")
