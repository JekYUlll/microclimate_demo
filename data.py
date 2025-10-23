import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")  # 忽略南极低温环境下传感器偶发的微小数据波动警告

class SensorDirectParams:
    """
    基于文献1（Gallée等2012）、2（FlowCapt手册）、3（Parsivel2手册）、4（Bellot等2011），
    定义所有传感器可直接获取的参数类型，含参数详情与文献依据
    """
    # 小型气象站直接测量参数（文献1核心定义）
    METEOROLOGICAL = {
        "wind_speed_2m": {
            "unit": "m/s",
            "dtype": float,
            "sensor": "小型气象站",
            "doc_basis": "文献1：2m高度30min风速均值，是吹雪发生的核心判断指标（需>10m/s）；文献4：用于校准吹雪通量"
        },
        "wind_dir_2m": {
            "unit": "degree",
            "dtype": float,
            "sensor": "小型气象站",
            "doc_basis": "文献1：南极katabatic（下降风）方向数据，辅助分析风场对吹雪的驱动作用"
        },
        "air_temp_2m": {
            "unit": "°C",
            "dtype": float,
            "sensor": "小型气象站",
            "doc_basis": "文献1：需剔除风速<2m/s时的杂散发热数据，用于计算大气虚温、稳定度Ri数"
        },
        "air_rh_2m": {
            "unit": "%",
            "dtype": float,
            "sensor": "小型气象站",
            "doc_basis": "文献1：用于计算比湿、水汽压，支撑感热/潜热通量推导"
        },
        "atm_pressure": {
            "unit": "hPa",
            "dtype": float,
            "sensor": "小型气象站",
            "doc_basis": "文献1：用于计算空气密度（流体静力近似），是MAR模型大气动力学参数化输入"
        }
    }

    # 雪面温度红外传感器直接测量参数（文献1核心定义）
    SNOW_SURFACE = {
        "snow_surface_temp": {
            "unit": "°C",
            "dtype": float,
            "sensor": "雪面温度红外传感器",
            "doc_basis": "文献1：用于判断雪面融化（≥0°C）与再冻结过程，该过程会干扰吹雪模拟精度"
        }
    }

    # FlowCapt传感器直接测量参数（文献2规格+文献4校准依据）
    FLOWCAPT = {
        "flowcapt_raw_snow_flux": {
            "unit": "g/(m²·s)",
            "dtype": float,
            "sensor": "FlowCapt传感器",
            "doc_basis": "文献2：声学信号输出的原始吹雪通量；文献4：需用A比值（均值9.88）修正高风速下的高估问题"
        },
        "flowcapt_acoustic_signal": {
            "unit": "mV",
            "dtype": float,
            "sensor": "FlowCapt传感器",
            "doc_basis": "文献2：传感器内部声学压力信号，反映吹雪粒子撞击强度，辅助判断数据有效性"
        }
    }

    # Parsivel2传感器直接测量参数（文献3规格+文献4区分标准）
    PARSIVEL2 = {
        "precipitation_water_eq": {
            "unit": "mm",
            "dtype": float,
            "sensor": "Parsivel2传感器",
            "doc_basis": "文献3：激光光学原理测得的降水水当量（固态/液态），是SMB“收入项”核心"
        },
        "precip_particle_count": {
            "unit": "count/m³",
            "dtype": int,
            "sensor": "Parsivel2传感器",
            "doc_basis": "文献3：单位体积降水粒子数；文献4：结合粒子速度（>5m/s）区分降水与吹雪"
        },
        "precip_particle_avg_diam": {
            "unit": "mm",
            "dtype": float,
            "sensor": "Parsivel2传感器",
            "doc_basis": "文献3：粒子直径32类（0.2-25mm）加权均值，辅助判断降水类型（雪/雪粒）"
        },
        "precip_particle_avg_vel": {
            "unit": "m/s",
            "dtype": float,
            "sensor": "Parsivel2传感器",
            "doc_basis": "文献3：粒子速度32类（0.2-20m/s）加权均值；文献4：速度>5m/s判定为吹雪粒子"
        },
        "visibility": {
            "unit": "m",
            "dtype": float,
            "sensor": "Parsivel2传感器",
            "doc_basis": "文献3：通过激光散射强度换算，反映吹雪对大气透明度的影响"
        }
    }

    # 全辐射传感器直接测量参数（文献1能量平衡需求）
    RADIATION = {
        "shortwave_radiation_in": {
            "unit": "W/m²",
            "dtype": float,
            "sensor": "全辐射传感器",
            "doc_basis": "文献1：入射短波辐射，是雪面净辐射通量计算核心，影响雪面升华、融化"
        },
        "shortwave_radiation_out": {
            "unit": "W/m²",
            "dtype": float,
            "sensor": "全辐射传感器",
            "doc_basis": "文献1：反射短波辐射，用于计算雪面反照率（α=出射/入射）"
        },
        "longwave_radiation_in": {
            "unit": "W/m²",
            "dtype": float,
            "sensor": "全辐射传感器",
            "doc_basis": "文献1：入射长波辐射（来自大气），参与雪面能量平衡计算"
        },
        "longwave_radiation_out": {
            "unit": "W/m²",
            "dtype": float,
            "sensor": "全辐射传感器",
            "doc_basis": "文献1：出射长波辐射（来自雪面），需结合雪面温度（红外传感器）验证一致性"
        }
    }

    @classmethod
    def get_all_direct_params(cls) -> dict:
        """整合所有传感器直接测量参数（基于4篇文献）"""
        all_params = {}
        for attr in [cls.METEOROLOGICAL, cls.SNOW_SURFACE, cls.FLOWCAPT, cls.PARSIVEL2, cls.RADIATION]:
            all_params.update(attr)
        return all_params

    @classmethod
    def get_param_sensor_mapping(cls) -> dict:
        """获取“参数名-关联传感器”映射（依据4篇文献传感器功能定义）"""
        mapping = {}
        sensor_param_pairs = [
            ("小型气象站", cls.METEOROLOGICAL),
            ("雪面温度红外传感器", cls.SNOW_SURFACE),
            ("FlowCapt传感器", cls.FLOWCAPT),
            ("Parsivel2传感器", cls.PARSIVEL2),
            ("全辐射传感器", cls.RADIATION)
        ]
        for sensor, params in sensor_param_pairs:
            for param in params.keys():
                mapping[param] = sensor
        return mapping
    
class AntarcticMicroclimateDataLoader:
    """
    南极单点微气候数据加载器：基于文献1（数据质控）、2（FlowCapt数据）、3（Parsivel2数据）、4（校准），
    实现原始数据加载、预处理与机器学习输入特征提取
    """
    def __init__(self):
        self.direct_params = SensorDirectParams.get_all_direct_params()
        self.param_sensor_mapping = SensorDirectParams.get_param_sensor_mapping()
        self.required_params = list(self.direct_params.keys())  # 4篇文献定义的必选直接测量参数

    def load_raw_data(self, file_path: str, sep: str = ",") -> pd.DataFrame:
        """
        加载传感器原始数据（CSV格式）
        数据要求：列名匹配4篇文献定义的直接测量参数，索引为时间戳（datetime）
        """
        try:
            raw_data = pd.read_csv(file_path, sep=sep, parse_dates=[0], index_col=0)
            # 验证参数完整性（基于4篇文献必选参数）
            missing_params = [p for p in self.required_params if p not in raw_data.columns]
            if missing_params:
                raise ValueError(f"缺失4篇文献定义的必选参数：{missing_params}，请补充")
            # 数据类型转换（匹配文献定义的参数类型）
            for param in self.required_params:
                raw_data[param] = raw_data[param].astype(self.direct_params[param]["dtype"])
            print(f"数据加载完成：时间范围 {raw_data.index.min()} 至 {raw_data.index.max()}，共 {len(raw_data)} 条记录")
            return raw_data
        except Exception as e:
            raise RuntimeError(f"加载失败：{str(e)}")

    def preprocess_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理（严格遵循4篇文献质控标准）
        步骤：1. 异常值剔除（文献1+4）；2. 缺失值填充（适配南极数据特征）；3. 标准化
        """
        processed_data = raw_data.copy()

        # 1. 异常值剔除（依据文献1、3、4）
        ## （1）文献1：剔除风速<2m/s时的温度数据（杂散发热干扰）
        wind_low_mask = processed_data["wind_speed_2m"] < 2
        processed_data.loc[wind_low_mask, "air_temp_2m"] = np.nan
        ## （2）文献2+4：剔除FlowCapt负通量（物理无效）
        processed_data.loc[processed_data["flowcapt_raw_snow_flux"] < 0, "flowcapt_raw_snow_flux"] = np.nan
        ## （3）文献3：剔除Parsivel2超量程能见度（>20000m）
        processed_data.loc[processed_data["visibility"] > 20000, "visibility"] = np.nan

        # 2. 缺失值填充（文献1：南极微气候日变化稳定，用线性插值+24h滚动均值）
        processed_data = processed_data.interpolate(method="linear", limit_direction="both")
        processed_data = processed_data.fillna(processed_data.rolling(window=24, min_periods=1).mean())

        # 3. 标准化（避免量纲影响，适配机器学习模型）
        for param in self.required_params:
            mean = processed_data[param].mean()
            std = processed_data[param].std()
            processed_data[f"{param}_norm"] = (processed_data[param] - mean) / std

        print(f"预处理完成：剩余 {len(processed_data)} 条记录，无缺失值")
        return processed_data

    def get_ml_input(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """提取机器学习输入特征（仅含4篇文献定义的直接测量参数标准化版本）"""
        ml_features = processed_data[[f"{p}_norm" for p in self.required_params]]
        # 添加传感器标签（依据4篇文献传感器-参数映射）
        ml_features["sensor_type"] = [self.param_sensor_mapping[p.replace("_norm", "")] for p in ml_features.columns[:-1]]
        return ml_features