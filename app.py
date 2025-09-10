from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from darts import TimeSeries
from darts.models import TFTModel
import pandas as pd
import numpy as np
import logging
import os
import json
import pickle
from datetime import datetime

# ----------------------------
# 日志配置
# ----------------------------
log_filename = f"log/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs("log", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ----------------------------
# FastAPI 初始化
# ----------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------------------
# WebSocket连接管理器
# ----------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket连接建立，当前连接数: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket连接断开，当前连接数: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        disconnected = []
        for conn in self.active_connections:
            try:
                await conn.send_text(json.dumps(message, ensure_ascii=False))
            except Exception:
                disconnected.append(conn)
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# ----------------------------
# 模型与工具函数
# ----------------------------
series_cache = {}

try:
    model = TFTModel.load("tft_model.pt")
    logger.info("模型加载成功")
except Exception as e:
    model = None
    logger.error(f"模型加载失败: {e}")

try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    logger.info("Scaler加载成功")
except Exception:
    scaler = None
    logger.warning("未找到Scaler，预测将使用原始数据")

try:
    with open("training_info.pkl", "rb") as f:
        training_info = pickle.load(f)
    logger.info("训练信息加载成功")
except Exception:
    training_info = None


def build_timeseries(df: pd.DataFrame) -> TimeSeries | None:
    """根据DataFrame构造TimeSeries，自动尝试不同频率"""
    try:
        return TimeSeries.from_dataframe(df, time_col="time", value_cols=["temperature", "humidity", "wind_dir", "wind_speed"], fill_missing_dates=False)
    except Exception:
        for freq in ["H", "T"]:
            try:
                return TimeSeries.from_dataframe(df, time_col="time", value_cols=["temperature", "humidity", "wind_dir", "wind_speed"], fill_missing_dates=True, freq=freq)
            except Exception:
                continue
    return None


def make_prediction(series: TimeSeries, horizon: int = 6) -> pd.DataFrame:
    """对时间序列做预测并返回DataFrame"""
    # 缩放
    s = scaler.transform(series) if scaler else series
    forecast = model.predict(n=horizon, series=s)
    forecast = scaler.inverse_transform(forecast) if scaler else forecast

    df = forecast.to_dataframe().reset_index()
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if 'time' in df.columns:
        df['time'] = df['time'].astype(str)
    return df

# ----------------------------
# 请求体定义
# ----------------------------
class WeatherData(BaseModel):
    station: str
    record_time: datetime
    temperature: float
    humidity: int
    wind_dir: int
    wind_speed: float

# ----------------------------
# API 路由
# ----------------------------
@app.post("/ingest")
async def ingest(data_point: WeatherData):
    if model is None:
        return {"error": "模型未加载"}

    # 初始化站点缓存
    if data_point.station not in series_cache:
        series_cache[data_point.station] = pd.DataFrame(columns=["time", "temperature", "humidity", "wind_dir", "wind_speed"])

    df = series_cache[data_point.station]
    ts = pd.to_datetime(data_point.record_time).tz_localize(None)

    # 更新或新增数据
    if ts in df['time'].values:
        df.loc[df['time'] == ts, ["temperature", "humidity", "wind_dir", "wind_speed"]] = [
            data_point.temperature, data_point.humidity, data_point.wind_dir, data_point.wind_speed
        ]
    else:
        df = pd.concat([df, pd.DataFrame({"time":[ts], "temperature":[data_point.temperature], "humidity":[data_point.humidity], "wind_dir":[data_point.wind_dir], "wind_speed":[data_point.wind_speed]})], ignore_index=True)

    df = df.sort_values("time")
    series_cache[data_point.station] = df

    # 预测
    if len(df) >= 24:
        series = build_timeseries(df)
        if series is None:
            return {"error": "无法构造时间序列"}

        forecast_df = make_prediction(series)

        await manager.broadcast({
            "type": "prediction_update",
            "station": data_point.station,
            "forecast": forecast_df.to_dict(orient="records"),
            "timestamp": datetime.now().isoformat()
        })

        return {"station": data_point.station, "forecast": forecast_df.to_dict(orient="records")}

    return {"station": data_point.station, "status": "not enough data", "current_points": len(df)}


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/api")
def api_root():
    return {"message": "微气候预测服务运行中"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # 保持连接
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/predictions/{station}")
def get_predictions(station: str):
    if station not in series_cache:
        return {"error": f"站点 {station} 没有数据"}

    df = series_cache[station]
    if len(df) < 24:
        return {"station": station, "status": "not enough data", "current_points": len(df)}

    series = build_timeseries(df)
    if series is None:
        return {"error": "无法构造时间序列"}

    forecast_df = make_prediction(series)
    return {"station": station, "forecast": forecast_df.to_dict(orient="records"), "last_update": df["time"].iloc[-1].isoformat()}


@app.get("/stations")
def get_stations():
    return {
        s: {
            "data_points": len(df),
            "last_update": df["time"].iloc[-1].isoformat() if len(df) else None,
            "can_predict": len(df) >= 24
        }
        for s, df in series_cache.items()
    }
