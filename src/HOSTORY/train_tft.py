#!/home/zhangzhuyu/.conda/envs/darts/bin/python

# ============================================
# 1. 导入依赖
# ============================================
# import torch
# print(torch.cuda.is_available())  # True
# print(torch.cuda.device_count())  # 应该返回 6
# print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 4090

import warnings
import os

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TQDM_DISABLE'] = '1'

# 设置NCCL环境变量，解决多GPU通信超时问题
os.environ['NCCL_TIMEOUT'] = '1800'  # 30分钟超时
os.environ['NCCL_IB_TIMEOUT'] = '22'  # InfiniBand超时
os.environ['NCCL_DEBUG'] = 'INFO'  # 调试信息（可选）

import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector
import numpy as np

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, mse

# 定义特征列
FEATURES = ["temperature", "humidity", "wind_speed"]

# ============================================
# 耦合关系函数定义
# ============================================
def add_coupling_features(df: pd.DataFrame, temperature_col: str = "temperature", 
                          humidity_col: str = "humidity", 
                          light_col: str = "light", 
                          wind_speed_col: str = "wind_speed") -> pd.DataFrame:
    """
    添加温度、湿度、光照、风速之间的数学耦合关系特征
    
    参数:
        df: 包含原始特征的DataFrame
        temperature_col: 温度列名
        humidity_col: 湿度列名
        light_col: 光照列名
        wind_speed_col: 风速列名
    
    返回:
        添加了耦合特征后的DataFrame
    """
    df_coupled = df.copy()
    
    # 确保所有需要的列都存在
    required_cols = [temperature_col, humidity_col, light_col, wind_speed_col]
    missing_cols = [col for col in required_cols if col not in df_coupled.columns]
    if missing_cols:
        print(f"⚠️ 警告：缺少以下列，将使用默认值0: {missing_cols}")
        for col in missing_cols:
            df_coupled[col] = 0.0
    
    # 提取变量（使用 .values 避免索引问题）
    T = df_coupled[temperature_col].values
    H = df_coupled[humidity_col].values
    L = df_coupled[light_col].values
    W = df_coupled[wind_speed_col].values
    
    # ============================================
    # 在这里添加您的耦合关系公式
    # ============================================
    # 示例公式（请根据实际物理关系修改）：
    
    # 示例1: 温度-湿度交互项（反映体感温度）
    df_coupled['temp_humidity_coupling'] = T * H / 100.0  # 归一化因子
    
    # 示例2: 光照-温度关系（反映太阳辐射对温度的影响）
    df_coupled['light_temp_coupling'] = L * T / 100.0  # 归一化因子
    
    # 示例3: 风速-温度关系（反映风冷效应）
    df_coupled['wind_temp_coupling'] = W * (T - 20.0) / 10.0  # 参考温度20度
    
    # 示例4: 湿度-风速关系（反映蒸发冷却）
    df_coupled['humidity_wind_coupling'] = H * W / 100.0
    
    # 示例5: 综合耦合项（多变量交互）
    df_coupled['multi_coupling'] = (T * H * W) / 10000.0
    
    # 示例6: 能量平衡项（光照-温度-湿度）
    df_coupled['energy_balance'] = L * (T + 273.15) * (1 - H/100.0) / 1000.0
    
    # ============================================
    # 请在此处替换为您的实际耦合公式
    # ============================================
    # 如果需要替换上面的示例公式，请修改这部分代码
    
    print(f"✅ 已添加耦合特征，新特征列: {[col for col in df_coupled.columns if col not in df.columns]}")
    
    return df_coupled

os.makedirs("plots", exist_ok=True)

# ============================================
# 2. 连接 MySQL 并读取数据
# ============================================

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
-- wind_dir  # 暂时注释掉风向，因为数据有问题
-- 如果数据库中有光照数据，请取消下面的注释并添加光照列名
-- , light 或 illumination 或 illuminance
FROM weather_data
ORDER BY record_time ASC
"""

df = pd.read_sql(query, conn)
conn.close()

# 转换时间列
df['record_time'] = pd.to_datetime(df['record_time'])

print(f"数据形状: {df.shape}")
print(f"时间范围: {df['record_time'].min()} 到 {df['record_time'].max()}")
print(f"站点数量: {df['station'].nunique()}")
print(f"各站点数据量:")
print(df['station'].value_counts())

df.head()

# ============================================
# 3. 数据可视化（单个站点示例）
# ============================================
station = "长城站"  # 要展示的气象站
df_station = df[df["station"] == station]

print(f"长城站数据量: {len(df_station)}")
print(f"长城站时间范围: {df_station['record_time'].min()} 到 {df_station['record_time'].max()}")
print(f"长城站数据预览:")
print(df_station.head())

plt.figure(figsize=(12, 5))
plt.plot(df_station['record_time'], df_station['temperature'], label='Temperature')
plt.plot(df_station['record_time'], df_station['humidity'], label='Humidity')
plt.title(f"Weather Data - {station}")
plt.legend()
# plt.show()
plt.tight_layout()
plt.savefig(f"plots/{station}_raw_data.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ 已保存: plots/{station}_raw_data.png")

# ============================================
# 4. 转换为 Darts TimeSeries
# ============================================

# 选择一个站点训练
df_train = df[df["station"] == station].copy()
df_train = df_train.sort_values('record_time').reset_index(drop=True)

print(f"训练数据形状: {df_train.shape}")
print(f"时间范围: {df_train['record_time'].min()} 到 {df_train['record_time'].max()}")

# 时间列为 datetime. 去掉 tz 信息
df_train["record_time"] = pd.to_datetime(df_train["record_time"], errors="coerce").dt.tz_localize(None)

value_cols = FEATURES
df_train[value_cols] = df_train[value_cols].apply(pd.to_numeric, errors="coerce")

df_train = df_train.set_index("record_time")[value_cols].resample("5min").mean()
df_train = df_train.interpolate(method="time")
df_train = df_train.reset_index()

# ============================================
# 添加数学耦合关系特征
# ============================================
print("\n=== 添加耦合关系特征 ===")

# 检查是否有光照数据，如果没有则创建占位符（可以用0或从其他特征推导）
if "light" not in df_train.columns:
    # 如果没有光照数据，可以从其他特征推导或使用默认值
    # 这里提供一个简单的占位符（可根据实际情况修改）
    print("⚠️ 未找到光照数据，使用占位符（可根据实际情况修改）")
    # 选项1: 使用0作为占位符
    df_train["light"] = 0.0
    # 选项2: 可以从时间推导（例如：白天时间有光照，夜间为0）
    # import datetime
    # df_train["light"] = df_train["record_time"].apply(
    #     lambda x: 1000.0 if 6 <= x.hour <= 18 else 0.0
    # )
    # 选项3: 从其他特征推导（例如：基于温度和湿度）
    # df_train["light"] = df_train["temperature"].apply(lambda x: max(0, x * 50))

# 添加耦合特征
df_train = add_coupling_features(df_train, 
                                  temperature_col="temperature",
                                  humidity_col="humidity", 
                                  light_col="light",
                                  wind_speed_col="wind_speed")

# 更新特征列列表，包含原始特征和耦合特征
original_features = FEATURES.copy()
coupling_features = [col for col in df_train.columns 
                     if col not in original_features + ["record_time", "station"]]
FEATURES = original_features + coupling_features

print(f"原始特征: {original_features}")
print(f"耦合特征: {coupling_features}")
print(f"总特征数: {len(FEATURES)}")

# 更新value_cols以包含所有特征（原始特征 + 耦合特征）
value_cols = FEATURES

# 创建TimeSeries
series = TimeSeries.from_dataframe(
    df_train,
    time_col="record_time",
    value_cols=value_cols,
    fill_missing_dates=True,
    freq="5min"
)

print(f"TimeSeries创建成功，长度: {len(series)}")

# 方法1：原始TFT方法（Darts Scaler + 重采样）- 注释掉
# scaler = Scaler()
# series_scaled = scaler.fit_transform(series)

# 方法2：与LSTM保持一致的方法（MinMaxScaler + 原始数据）
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# MinMax
scaler = MinMaxScaler()
data_values = df_train[value_cols].values
data_scaled = scaler.fit_transform(data_values)

# 创建标准化的DataFrame
df_scaled = df_train.copy()
df_scaled[value_cols] = data_scaled

# 创建TimeSeries - 需要指定频率，因为原始数据时间间隔不规则
series = TimeSeries.from_dataframe(
    df_scaled,
    time_col="record_time",
    value_cols=value_cols,
    fill_missing_dates=True,  # 需要填充缺失日期
    freq="5min"  # 指定5分钟频率，与原始方法保持一致
)

series_scaled = series
series_scaled = series_scaled.astype(np.float32)

# 绘制图表
plt.figure(figsize=(12, 6))
series_scaled.plot()
plt.title(f"Scaled Multivariate Weather Data - {station}")
# plt.show()
plt.tight_layout()
plt.savefig(f"plots/{station}_scaled_data.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"✅ 已保存: plots/{station}_scaled_data.png")


# ============================================
# 5. 划分训练集和验证集
# ============================================
# 使用最后2天的数据作为验证集，5分钟频率：1天 = 288个点，2天 = 576个点
# 确保验证集至少有2天（576个点）的数据，以便更好地评估模型效果
val_size = max(576, int(len(series_scaled) * 0.10))  # 至少2天（576个点），或总数据的10%
# 如果数据量足够，可以设置上限（比如最多3-4天）
if len(series_scaled) > 10000:
    val_size = min(val_size, 1152)  # 最多4天（1152个点），如果数据量很大
train, val = series_scaled[:-val_size], series_scaled[-val_size:]

print(f"训练集长度: {len(train)}")
print(f"验证集长度: {len(val)}")
print(f"训练集时间范围: {train.start_time()} 到 {train.end_time()}")
print(f"验证集时间范围: {val.start_time()} 到 {val.end_time()}")

# ============================================
# 6. 定义并训练 TFT 模型
# ============================================
# 注意：这里已经修改了use_static_covariates=False，因为数据中没有静态协变量
model = TFTModel(
    input_chunk_length=24,   # 输入窗口 24 步
    output_chunk_length=6,   # 预测未来 6 步
    hidden_size=32,
    n_epochs=20,            # 训练轮数
    batch_size=128,         # ✅ 增大batch_size，充分利用多GPU（从16增加到128）
    add_relative_index=True,
    use_static_covariates=False,  # 关闭静态协变量，因为没有提供
    random_state=42,
    optimizer_kwargs={"lr": 0.001},  # 学习率
    lr_scheduler_cls=None,           # 不用调度器
    dropout=0.1,                     # dropout 防止过拟合
    force_reset=True,                # 每次训练重新初始化
    pl_trainer_kwargs={
        "accelerator": "gpu",
        "devices": [0, 1, 2, 3, 4, 5],  # ✅ 使用全部6块GPU（从[0,5]改为全部）
        "precision": 32,  # ✅ 改成全精度，避免 fp16 溢出
        "strategy": "ddp",  # ✅ 使用DDP策略，比auto更高效
    },
)

print("开始训练模型...")
print(f"训练数据形状: {train.all_values().shape}")
print(f"验证数据形状: {val.all_values().shape}")

model.fit(train, val_series=val, verbose=True)

# ============================================
# 7. 保存模型（训练完成后立即保存）
# ============================================
print("=== 保存训练好的模型 ===")
import pickle

# 保存模型
model.save("tft_model.pt")
print("✅ 模型已保存到 tft_model.pt")

# 保存 Scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ Scaler 已保存到 scaler.pkl")

# 保存训练信息
training_info = {
    "station": station,
    "train_length": len(train),
    "val_length": len(val),
    "freq": "5min",
    "input_chunk_length": 24,
    "output_chunk_length": 6,
    "features": original_features,  # 只保存原始特征名
    "coupling_features": coupling_features  # 保存耦合特征名
}
with open("training_info.pkl", "wb") as f:
    pickle.dump(training_info, f)
print("✅ 训练信息已保存到 training_info.pkl")

# ============================================
# 8. 预测并可视化（重新加载模型用于预测，避免多GPU状态问题）
# ============================================
print("\n=== 开始预测 ===")
print(f"训练数据长度: {len(train)}")
print(f"验证数据长度: {len(val)}")

# 重新加载模型用于预测（单GPU模式，避免多GPU预测时的AssertionError）
print("重新加载模型用于预测...")
try:
    # 重新加载模型
    model_for_prediction = TFTModel.load("tft_model.pt")
    # 设置单GPU预测配置（避免多GPU预测时的AssertionError）
    if hasattr(model_for_prediction, 'trainer_params'):
        model_for_prediction.trainer_params = {
            **model_for_prediction.trainer_params,
            "accelerator": "gpu",
            "devices": [0],  # 只使用第一个GPU进行预测
            "strategy": "auto",
        }
    print("✅ 模型重新加载成功，已切换到单GPU模式")
except Exception as e:
    print(f"⚠️ 重新加载模型失败，使用原始模型: {e}")
    import traceback
    traceback.print_exc()
    model_for_prediction = model

if model_for_prediction is None:
    print("❌ 模型未加载成功")
else:
    try:
        print("使用验证集进行预测...")
        print(f"验证集长度: {len(val)} 个时间点（约 {len(val)*5/60:.1f} 小时，{len(val)*5/(60*24):.1f} 天）")
        
        # 使用historical_forecasts方法进行滚动预测，但严格限制在验证集范围内
        # 关键：只使用 train + val 作为输入序列，这样预测不会超出验证集
        print("开始滚动预测...")
        val_start_time = val.start_time()
        val_end_time = val.end_time()
        
        # 从训练集结束后的第一个时间点开始预测
        forecast_start = train.end_time() + series_scaled.freq
        
        # 关键修改：只使用 train + val 作为输入序列，而不是完整的 series_scaled
        # 这样可以确保预测不会超出验证集范围
        train_val_series = train.concatenate(val, ignore_time_axis=False)
        
        forecast = model_for_prediction.historical_forecasts(
            series=train_val_series,  # 只使用 train + val，而不是完整序列
            start=forecast_start,  # 从训练集结束后的第一个时间点开始
            forecast_horizon=6,  # 每次预测6步
            stride=6,  # 每次滚动6步（与output_chunk_length一致）
            retrain=False,
            verbose=True
        )
        
        # 如果返回的是列表，需要合并
        if isinstance(forecast, list):
            from darts import concatenate
            if len(forecast) > 0:
                forecast = concatenate(forecast, axis=0)
            else:
                raise ValueError("预测结果为空")
        
        # 精确截取：使用slice方法，严格限制到验证集的时间范围
        # 先使用slice_intersect获取交集
        forecast = forecast.slice_intersect(val)
        
        if len(forecast) == 0:
            raise ValueError("预测结果与验证集时间范围无交集")
        
        # 关键：使用slice方法，精确截取到验证集的开始和结束时间
        # 这样可以确保预测结果严格限制在验证集的时间范围内
        try:
            # 尝试使用slice方法，指定开始和结束时间
            forecast = forecast.slice(val_start_time, val_end_time)
            print(f"✅ 已使用slice方法精确截取到验证集时间范围")
        except Exception as e:
            # 如果slice方法失败，使用时间索引匹配的方法
            print(f"⚠️ slice方法失败，使用时间索引匹配: {e}")
            forecast_times = forecast.time_index
            val_times = val.time_index
            
            # 找到预测中与验证集时间完全匹配的部分
            valid_indices = []
            for i, t in enumerate(forecast_times):
                if t >= val_start_time and t <= val_end_time:
                    valid_indices.append(i)
            
            if len(valid_indices) > 0:
                # 只保留与验证集时间匹配的预测点
                forecast = forecast[valid_indices]
                print(f"✅ 已使用时间索引匹配，保留{len(forecast)}个与验证集时间匹配的点")
            else:
                # 如果时间索引不匹配，使用长度截取（最后手段）
                forecast = forecast[:len(val)]
                print(f"⚠️ 时间索引不匹配，已按长度截取到验证集长度（{len(forecast)}个点）")
        
        # 最终对齐：确保预测和验证集的时间范围完全一致
        # 如果预测长度仍然大于验证集，按长度截取
        if len(forecast) > len(val):
            forecast = forecast[:len(val)]
            print(f"⚠️ 最终对齐：预测长度已截取到验证集长度（{len(forecast)}个点）")
        
        # 如果预测长度小于验证集，截取验证集到相同长度以便比较
        if len(forecast) < len(val):
            val = val[:len(forecast)]
            print(f"⚠️ 最终对齐：验证集长度已截取到预测长度（{len(forecast)}个点）")
        
        # 最终验证：确保预测的时间范围不超过验证集
        if forecast.start_time() < val_start_time or forecast.end_time() > val_end_time:
            print(f"⚠️ 警告：预测时间范围仍然超出验证集范围")
            print(f"   预测范围: {forecast.start_time()} 到 {forecast.end_time()}")
            print(f"   验证集范围: {val_start_time} 到 {val_end_time}")

        print("✅ 预测完成")
        print(f"预测结果长度: {len(forecast)}")
        print(f"验证集长度: {len(val)}")
        print(f"预测时间范围: {forecast.start_time()} 到 {forecast.end_time()}")
        print(f"验证集时间范围: {val.start_time()} 到 {val.end_time()}")

        forecast_values = forecast.all_values()
        print(f"预测数据统计: min={forecast_values.min():.4f}, max={forecast_values.max():.4f}, mean={forecast_values.mean():.4f}")

        if np.all(forecast_values == 0):
            print("⚠️ 警告：预测结果全为0")
        elif np.isnan(forecast_values).all():
            print("⚠️ 警告：预测结果全为NaN")

        # 绘制结果 - 只可视化原始特征（耦合特征作为辅助特征，不单独可视化）
        # 如果需要可视化耦合特征，可以修改这里的 original_features 为 FEATURES
        features_to_plot = original_features if 'original_features' in locals() else FEATURES
        
        fig, axes = plt.subplots(len(features_to_plot), 1, figsize=(18, 5*len(features_to_plot)), sharex=True)
        if len(features_to_plot) == 1:
            axes = [axes]
        
        # 特征名称映射（英文）
        feature_names = {
            "temperature": "Temperature",
            "humidity": "Humidity", 
            "wind_speed": "Wind Speed",
            "light": "Light/Illumination",
            # 耦合特征名称映射
            "temp_humidity_coupling": "Temp-Humidity Coupling",
            "light_temp_coupling": "Light-Temp Coupling",
            "wind_temp_coupling": "Wind-Temp Coupling",
            "humidity_wind_coupling": "Humidity-Wind Coupling",
            "multi_coupling": "Multi-Variable Coupling",
            "energy_balance": "Energy Balance"
        }
        
        for i, col in enumerate(features_to_plot):
            ax = axes[i]
            val_series = val.univariate_component(col)
            forecast_series = forecast.univariate_component(col)
            
            # 确保验证集和预测结果的时间范围完全对齐
            # 使用slice_intersect确保只绘制重叠的时间范围
            val_plot = val_series.slice_intersect(forecast_series)
            forecast_plot = forecast_series.slice_intersect(val_series)
            
            # 如果slice_intersect后长度不一致，按较短的长度截取
            if len(val_plot) != len(forecast_plot):
                min_len = min(len(val_plot), len(forecast_plot))
                val_plot = val_plot[:min_len]
                forecast_plot = forecast_plot[:min_len]
            
            val_plot.plot(label="Actual", color='blue', linewidth=1.5, ax=ax)
            forecast_plot.plot(label="Predicted", color='red', linewidth=1.5, ax=ax, linestyle='--')

            feature_name = feature_names.get(col, col)
            hours = len(forecast) * 5 / 60
            days = hours / 24
            ax.set_title(f"{feature_name} - Actual vs Predicted (Range: {len(forecast)} points, ~{hours:.1f}h/{days:.2f}d)", 
                        fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 添加MAE和MSE信息（使用对齐后的数据）
            col_mae = mae(val_plot, forecast_plot)
            col_mse = mse(val_plot, forecast_plot)
            
            # 数值范围（使用英文，使用对齐后的数据）
            ax.text(
                0.02, 0.98,
                f"Actual range: [{val_plot.values().min():.3f}, {val_plot.values().max():.3f}]\n"
                f"Predicted range: [{forecast_plot.values().min():.3f}, {forecast_plot.values().max():.3f}]\n"
                f"MAE: {col_mae:.4f}, MSE: {col_mse:.4f}",
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9
            )
        
        plt.xlabel('Time', fontsize=11)
        plt.tight_layout()
        plt.savefig(f"plots/{station}_prediction.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ 已保存: plots/{station}_prediction.png")

        # 误差指标（mae和mse已在前面导入）
        # 只显示原始特征的误差统计（耦合特征作为辅助特征）
        print("\n=== 预测误差统计（原始特征）===")
        features_to_evaluate = original_features if 'original_features' in locals() else FEATURES
        for col in features_to_evaluate:
            actual = val.univariate_component(col)
            predicted = forecast.univariate_component(col)

            print(f"{col}: MAE={mae(actual, predicted):.4f}, MSE={mse(actual, predicted):.4f}")

    except Exception as e:
        print(f"❌ 预测失败: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()
