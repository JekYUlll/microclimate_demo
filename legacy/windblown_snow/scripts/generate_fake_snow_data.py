"""
用于生成风吹雪观测的合成 CSV（供沙箱模型测试）。

整体生成策略（更详细说明）：
1) 时间网格与长期结构
   - 以 freq_seconds 为步长生成规则时间序列（默认 1 秒），避免不规则采样带来的噪声。
   - 引入日变化（正弦项）作为辐射/气温/风速的基础周期性。
2) 状态变量的时间连续性
   - 关键气象量用 AR(1) 过程推进，保证序列平滑且有可学习的自相关结构。
   - 通过“状态均值 + 持续性 + 噪声”的方式控制尺度和波动。
3) 风暴状态作为外部门控
   - 使用二态马尔可夫过程（storm_state）模拟风暴出现/消散。
   - 风暴影响风速上升、降水增强、辐射衰减等多个变量，形成协同变化。
4) 输运阈值与雪源供给
   - 通过经验阈值 Ut 计算风速超越程度 ratio（决定是否发生明显输运）。
   - 雪源供给 snow_supply 由降水补充、由输运消耗，形成正反馈/负反馈。
5) 多物理量耦合
   - 雪通量与风速、供给因子、风暴强度相关；湿度影响降水增强。
   - 稳定度（ri_bulk）由温度梯度与风速综合估计。
6) 输出结构与守恒约束
   - 先生成总质量/数目通量，再用 Dirichlet 分配到粒径/速度分箱。
   - 分箱后的通量总和与原始总通量一致，便于谱建模。

输出数据结构（概览）：
- timestamp：ISO-8601 时间字符串，规则采样网格。
- 气象：气温/雪面温度、相对湿度、气压、风速/风向、太阳辐射。
- 输运：雪质量通量、数目通量、摩擦速度、湍流强度。
- 稳定度：体 Richardson 数（ri_bulk）+ 稳定度类别 stability_flag。
- 诊断量：热通量估计、阈值超越、耦合比等。
- 质量/元信息：quality_flag、data_source、missing_reason（占位）。
- 谱分箱：按粒径与速度分箱的质量/数目通量，总和守恒。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# 粒径分箱（微米），用于把总通量分配到各个谱分箱列。
SIZE_BINS_UM: List[Tuple[int, int]] = [(50, 100), (100, 200), (200, 400), (400, 800)]
# 速度分箱（m/s），同样用于谱分箱列。
VELOCITY_BINS_MS: List[Tuple[int, int]] = [(0, 2), (2, 4), (4, 6), (6, 8)]
# 由体 Richardson 数推断的稳定度类别。
STABILITY_FLAGS = ["stable", "neutral", "unstable"]


def build_parser() -> argparse.ArgumentParser:
    """合成数据生成的命令行参数。"""
    parser = argparse.ArgumentParser(description="Generate synthetic wind-blown snow CSV.")
    parser.add_argument(
        "--rows",
        type=int,
        default=2000,
        help="Number of observations to generate.",
    )
    parser.add_argument(
        "--freq-seconds",
        "--freq",
        dest="freq_seconds",
        type=int,
        default=1,
        help="Sampling interval in seconds between rows (default: 1s).",
    )
    parser.add_argument(
        "--freq-minutes",
        type=float,
        default=None,
        help="Deprecated: sampling interval in minutes; if set, overrides --freq-seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("legacy/windblown_snow/data/synthetic/windblown_snow_sample.csv"),
        help="Output CSV path.",
    )
    return parser


def _ar1_scaled(
    prev: float,
    mean: float,
    phi_base: float,
    noise_base: float,
    dt_scale: float,
    rng: np.random.Generator,
) -> float:
    """
    带时间步长缩放的 AR(1)：
    - phi_base/noise_base 对应“基准步长”（历史默认 10 分钟）。
    - dt_scale = 当前步长 / 基准步长。
    """
    phi_dt = float(phi_base ** dt_scale)
    noise_dt = float(noise_base * np.sqrt(dt_scale))
    return phi_dt * prev + (1 - phi_dt) * mean + rng.normal(0.0, noise_dt)


def _solar_flux(second_of_day: float, rng: np.random.Generator) -> float:
    """带噪声的简单日变化太阳辐射（W/m^2）。"""
    phase = 2 * np.pi * (second_of_day / (24 * 60 * 60))
    base = max(0.0, np.sin(phase))
    return max(0.0, 600.0 * base + rng.normal(0.0, 40.0))


def _distribute(total: float, bins: int, rng: np.random.Generator) -> np.ndarray:
    """用 Dirichlet 分布把总量分配到 bins 个分箱。"""
    if total <= 0:
        return np.zeros(bins)
    weights = rng.dirichlet(np.ones(bins))
    return total * weights


def generate_rows(rows: int, freq_seconds: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    """
    生成风吹雪观测的合成数据。

    关键数据机制说明：
    - 时间是规则网格，步长为 freq_seconds。
    - 状态变量具备 AR(1) 结构并叠加日变化。
    - 风暴状态影响风速/降水/辐射衰减。
    - 雪输运由风速阈值超越和雪源供给控制。
    - 谱分箱按构造保证总质量/数目通量守恒。
    """
    start = datetime.now(tz=timezone.utc).replace(microsecond=0)
    data: List[Dict[str, Any]] = []

    # 初始化状态，保证序列平滑起步。
    wind_speed = max(0.5, rng.normal(8.0, 1.5))
    wind_dir = rng.uniform(0, 360)
    air_temp = rng.normal(-12.0, 3.0)
    snow_temp = air_temp - rng.normal(1.5, 0.5)
    pressure = rng.normal(70000, 400)
    rh = np.clip(rng.normal(70, 8), 20, 100)
    snow_supply = float(rng.uniform(0.2, 0.8))
    storm_state = rng.random() < 0.1
    storm_intensity = 1.0
    # 以原来 10 分钟步长作为“基准”，所有动态项按 dt_scale 缩放
    base_step_seconds = 10 * 60
    dt_scale = max(1e-6, float(freq_seconds) / base_step_seconds)

    for i in range(rows):
        ts = start + timedelta(seconds=i * freq_seconds)
        second_of_day = (ts.hour * 3600) + (ts.minute * 60) + ts.second

        # 两状态风暴过程，转移概率较低。
        if storm_state:
            # 0.03 是 10 分钟步长下的转移概率；随 dt 缩放
            if rng.random() < 1 - (1 - 0.03) ** dt_scale:
                storm_state = False
        else:
            if rng.random() < 1 - (1 - 0.01) ** dt_scale:
                storm_state = True

        # 风暴强度是平滑的潜在乘子（AR1）。
        storm_intensity = float(
            np.clip(
                _ar1_scaled(
                    storm_intensity,
                    mean=1.4 if storm_state else 1.0,
                    phi_base=0.7,
                    noise_base=0.05,
                    dt_scale=dt_scale,
                    rng=rng,
                ),
                0.9,
                1.8,
            )
        )

        # 辐射具有日变化；风暴会衰减辐射。
        solar = _solar_flux(second_of_day, rng)
        if storm_state:
            solar *= 0.4 + 0.2 * rng.random()

        # 气温：日变化 + 风暴偏移 + AR1 持续性。
        temp_mean = -12.0 + 3.0 * np.sin(2 * np.pi * second_of_day / (24 * 60 * 60)) + (0.8 if storm_state else 0.0)
        air_temp = _ar1_scaled(air_temp, mean=temp_mean, phi_base=0.92, noise_base=0.6, dt_scale=dt_scale, rng=rng)
        # 雪面温度跟随气温，带滞后并受辐射影响。
        snow_temp = _ar1_scaled(
            snow_temp,
            mean=air_temp - 2.0 + 0.002 * solar,
            phi_base=0.9,
            noise_base=0.4,
            dt_scale=dt_scale,
            rng=rng,
        )

        # 风速：日变化 + 风暴抬升 + AR1 持续性。
        wind_mean = 8.0 + 1.5 * np.sin(2 * np.pi * second_of_day / (24 * 60 * 60)) + (3.5 if storm_state else 0.0)
        wind_speed = max(
            0.2,
            _ar1_scaled(
                wind_speed,
                mean=wind_mean * storm_intensity,
                phi_base=0.85,
                noise_base=0.9,
                dt_scale=dt_scale,
                rng=rng,
            ),
        )
        wind_dir = (wind_dir + rng.normal(0.0, 12.0)) % 360
        pressure = _ar1_scaled(
            pressure,
            mean=70000 - (300 if storm_state else 0),
            phi_base=0.95,
            noise_base=80,
            dt_scale=dt_scale,
            rng=rng,
        )
        rh_mean = 75.0 - 0.3 * (air_temp + 10.0) + (8.0 if storm_state else 0.0)
        rh = float(
            np.clip(
                _ar1_scaled(rh, mean=rh_mean, phi_base=0.9, noise_base=4.0, dt_scale=dt_scale, rng=rng),
                20,
                100,
            )
        )

        # 湍流代理量与稳定度（体 Richardson 数）。
        turbulence = float(np.clip(0.15 + 0.05 * wind_speed + rng.normal(0.0, 0.05), 0.0, 1.0))
        friction_velocity = max(0.05, 0.08 * wind_speed + rng.normal(0.0, 0.03))
        ri_bulk = float(np.clip(0.15 * (air_temp - snow_temp) - 0.08 * wind_speed + rng.normal(0.0, 0.2), -1.5, 1.5))

        # 雪输运阈值（简化经验形式）。
        Ut = 6.975 + 0.0033 * (air_temp + 27.27) ** 2
        ratio = max(0.0, (wind_speed - Ut) / max(Ut, 0.1))
        # 降水在风暴和高湿时更强。
        precip_rate = max(0.0, rng.gamma(shape=1.1, scale=0.3))
        if storm_state:
            precip_rate += max(0.0, rng.gamma(shape=1.4, scale=0.8))
        if rh > 85:
            precip_rate *= 1.0 + (rh - 85.0) / 40.0

        # 雪源供给随降水增加、随输运消耗。
        snow_supply = float(
            np.clip(
                snow_supply + (0.02 * precip_rate - 0.25 * ratio) * dt_scale,
                0.0,
                1.0,
            )
        )
        supply_factor = 0.2 + 0.8 * snow_supply
        if storm_state:
            supply_factor *= 1.1

        # 质量通量与（风速-阈值）的三次方和供给因子相关。
        snow_mass_flux = max(0.0, 0.0015 * (ratio ** 3) * supply_factor * (1 + rng.normal(0.0, 0.25)))
        snow_mass_flux += max(0.0, rng.lognormal(mean=-10.5, sigma=0.8) - 2e-5)

        # 用名义粒径/密度把质量通量转换为数目通量。
        diameter = 200e-6
        rho_ice = 917.0
        vol = (4 / 3) * np.pi * (diameter / 2) ** 3
        snow_number_flux = max(0.0, snow_mass_flux / (rho_ice * vol) * rng.uniform(0.8, 1.2))

        stability_flag = "neutral"
        if ri_bulk > 0.2:
            stability_flag = "stable"
        elif ri_bulk < -0.2:
            stability_flag = "unstable"

        # 核心标量特征（输入/输出量）。
        row: Dict[str, Any] = {
            "timestamp": ts.isoformat(),
            "air_temperature_c": air_temp,
            "relative_humidity": rh,
            "air_pressure_pa": pressure,
            "wind_speed_ms": wind_speed,
            "wind_direction_deg": wind_dir,
            "solar_radiation_wm2": solar,
            "snow_surface_temperature_c": snow_temp,
            "snow_mass_flux_kg_m2_s": snow_mass_flux,
            "snow_number_flux_m2_s": snow_number_flux,
            "turbulence_intensity": turbulence,
            "friction_velocity_ms": friction_velocity,
            "ri_bulk": ri_bulk,
            "stability_flag": stability_flag,
            "snow_density_kg_m3": max(100.0, 150.0 + 20.0 * wind_speed + rng.normal(0.0, 10.0)),
            "snow_grain_temp_c": snow_temp + rng.normal(0.0, 0.5),
            "visibility_m": max(10.0, 2000.0 / (1.0 + snow_mass_flux * 3e5) + rng.normal(0.0, 50.0)),
            "precip_rate_mm_h": precip_rate,
            "quality_flag": rng.choice(
                ["good", "suspect", "missing"],
                p=[0.85, 0.12, 0.03] if wind_speed > 12 or precip_rate > 2 else [0.92, 0.07, 0.01],
            ),
            "data_source": rng.choice(["sim", "field_logger_a", "field_logger_b"]),
            "missing_reason": "",
        }

        # 衍生诊断量（简单特征）。
        row["wind_snow_coupling"] = snow_mass_flux / max(wind_speed, 0.1)
        row["threshold_exceedance"] = int(wind_speed > Ut)
        row["net_radiation_est"] = 0.7 * solar - 20.0
        row["sensible_heat_flux_est"] = 0.1 * (air_temp - snow_temp) * wind_speed
        row["latent_heat_flux_est"] = 0.02 * rh * wind_speed

        # 展开谱分箱列，确保质量/数目通量守恒。
        size_mass = _distribute(snow_mass_flux, len(SIZE_BINS_UM), rng)
        size_num = _distribute(snow_number_flux, len(SIZE_BINS_UM), rng)
        for idx, (low, high) in enumerate(SIZE_BINS_UM, start=1):
            prefix = f"size_bin_{idx}_{low}_{high}_um"
            row[f"{prefix}_number_flux_m2_s"] = size_num[idx - 1]
            row[f"{prefix}_mass_flux_kg_m2_s"] = size_mass[idx - 1]

        vel_mass = _distribute(snow_mass_flux, len(VELOCITY_BINS_MS), rng)
        vel_num = _distribute(snow_number_flux, len(VELOCITY_BINS_MS), rng)
        for idx, (low, high) in enumerate(VELOCITY_BINS_MS, start=1):
            prefix = f"velocity_bin_{idx}_{low}_{high}_ms"
            row[f"{prefix}_number_flux_m2_s"] = vel_num[idx - 1]
            row[f"{prefix}_mass_flux_kg_m2_s"] = vel_mass[idx - 1]

        data.append(row)

    return data


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    if args.freq_minutes is not None:
        freq_seconds = int(args.freq_minutes * 60)
    else:
        freq_seconds = int(args.freq_seconds)
    rows = generate_rows(rows=args.rows, freq_seconds=freq_seconds, rng=rng)

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
