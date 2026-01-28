## Wind-blown Snow Dummy (Legacy)

### 中文说明
**目的**
- 这是一个旧版的风吹雪 dummy 实验区，用于快速验证序列模型（轻量 Transformer）+ 物理惩罚项的可用性。
- 数据为合成数据，核心目标是“可学”，不代表真实物理统计分布。

**目录结构**
- `legacy/windblown_snow/train_dummy.ipynb`：主实验 notebook。
- `legacy/windblown_snow/scripts/generate_fake_snow_data.py`：合成数据生成器。
- `legacy/windblown_snow/data/synthetic/windblown_snow_sample.csv`：合成样本。
- `legacy/windblown_snow/src/`：旧版模型与工具（`tft_model.py`, `snow_dataset.py`, `physics_losses.py`, `physics_calculations.py`）。
- `legacy/windblown_snow/plots/`：执行 notebook 时保存的图。

**快速开始**
1) 生成合成数据：
   - `python legacy/windblown_snow/scripts/generate_fake_snow_data.py`
2) 打开并运行 notebook：
   - `legacy/windblown_snow/train_dummy.ipynb`

**数据生成逻辑（简述）**
- 采用 AR(1) 过程模拟风速/气温/湿度等随时间演化。
- 引入“风暴状态”与“雪源供给”，让雪通量具备阈值效应与持续性。
- 通量与风速阈值 `Ut`、供给状态、降水强度耦合。

**数据结构与特征（CSV 字段）**
每行是一个时间点（共 42 列）。下表给出字段名称、中文翻译与说明（含英文说明）：

| 字段 | 中文译名 | 说明（中文） | Description (EN) |
| --- | --- | --- | --- |
| `timestamp` | 时间戳 | ISO 8601 时间（UTC）。 | ISO 8601 timestamp (UTC). |
| `air_temperature_c` | 气温（℃） | 近地空气温度。 | Near-surface air temperature (C). |
| `relative_humidity` | 相对湿度（%） | 0–100 范围的相对湿度。 | Relative humidity in percent. |
| `air_pressure_pa` | 气压（Pa） | 近地气压。 | Air pressure (Pa). |
| `wind_speed_ms` | 风速（m/s） | 10 m 风速（合成）。 | Wind speed (m/s). |
| `wind_direction_deg` | 风向（°） | 0–360 度，按气象角度。 | Wind direction in degrees. |
| `solar_radiation_wm2` | 太阳辐射（W/m²） | 日变化 + 风暴衰减的辐射。 | Solar radiation with diurnal cycle. |
| `snow_surface_temperature_c` | 雪面温度（℃） | 与气温/辐射耦合的雪面温度。 | Snow surface temperature (C). |
| `snow_mass_flux_kg_m2_s` | 雪质量通量 | 单位面积质量通量（kg/m²/s）。 | Mass flux of snow (kg/m²/s). |
| `snow_number_flux_m2_s` | 雪粒数通量 | 单位面积数通量（1/m²/s）。 | Number flux of particles (1/m²/s). |
| `turbulence_intensity` | 湍流强度 | 与风速相关的无量纲强度。 | Dimensionless turbulence intensity. |
| `friction_velocity_ms` | 摩阻速度（m/s） | 与风速相关的摩阻速度。 | Friction velocity (m/s). |
| `ri_bulk` | 体 Richardson 数 | 稳定度指标（无量纲）。 | Bulk Richardson number. |
| `stability_flag` | 稳定度标记 | 由 `ri_bulk` 划分：stable/neutral/unstable。 | Stability flag from `ri_bulk`. |
| `snow_density_kg_m3` | 雪密度（kg/m³） | 流动雪的密度估计。 | Estimated snow density (kg/m³). |
| `snow_grain_temp_c` | 雪粒温度（℃） | 雪粒温度扰动版本。 | Snow grain temperature (C). |
| `visibility_m` | 能见度（m） | 与通量相关的能见度衰减。 | Visibility reduced by flux. |
| `precip_rate_mm_h` | 降水强度（mm/h） | 与风暴/湿度耦合的降水率。 | Precipitation rate (mm/h). |
| `quality_flag` | 质量标记 | good/suspect/missing 的模拟标记。 | Simulated quality flag. |
| `data_source` | 数据来源 | sim/field_logger_a/field_logger_b。 | Simulated data source label. |
| `missing_reason` | 缺失原因 | 预留字段，当前为空字符串。 | Placeholder; empty string. |
| `wind_snow_coupling` | 风雪耦合比 | `snow_mass_flux / max(wind_speed, 0.1)`。 | Mass-flux-to-wind ratio. |
| `threshold_exceedance` | 阈值超越 | 若 `wind_speed > Ut` 则为 1，否则 0。 | 1 if wind exceeds `Ut`. |
| `net_radiation_est` | 净辐射估计 | `0.7 * solar - 20`（W/m²）。 | Estimated net radiation. |
| `sensible_heat_flux_est` | 感热通量估计 | `0.1 * (T - Ts) * U`（W/m²）。 | Estimated sensible heat flux. |
| `latent_heat_flux_est` | 潜热通量估计 | `0.02 * RH * U`（W/m²）。 | Estimated latent heat flux. |
| `size_bin_1_50_100_um_number_flux_m2_s` | 粒径 50–100 μm 数通量 | Dirichlet 分配的粒径 bin 数通量。 | Size-bin number flux (50–100 μm). |
| `size_bin_1_50_100_um_mass_flux_kg_m2_s` | 粒径 50–100 μm 质量通量 | Dirichlet 分配的粒径 bin 质量通量。 | Size-bin mass flux (50–100 μm). |
| `size_bin_2_100_200_um_number_flux_m2_s` | 粒径 100–200 μm 数通量 | Dirichlet 分配的粒径 bin 数通量。 | Size-bin number flux (100–200 μm). |
| `size_bin_2_100_200_um_mass_flux_kg_m2_s` | 粒径 100–200 μm 质量通量 | Dirichlet 分配的粒径 bin 质量通量。 | Size-bin mass flux (100–200 μm). |
| `size_bin_3_200_400_um_number_flux_m2_s` | 粒径 200–400 μm 数通量 | Dirichlet 分配的粒径 bin 数通量。 | Size-bin number flux (200–400 μm). |
| `size_bin_3_200_400_um_mass_flux_kg_m2_s` | 粒径 200–400 μm 质量通量 | Dirichlet 分配的粒径 bin 质量通量。 | Size-bin mass flux (200–400 μm). |
| `size_bin_4_400_800_um_number_flux_m2_s` | 粒径 400–800 μm 数通量 | Dirichlet 分配的粒径 bin 数通量。 | Size-bin number flux (400–800 μm). |
| `size_bin_4_400_800_um_mass_flux_kg_m2_s` | 粒径 400–800 μm 质量通量 | Dirichlet 分配的粒径 bin 质量通量。 | Size-bin mass flux (400–800 μm). |
| `velocity_bin_1_0_2_ms_number_flux_m2_s` | 速度 0–2 m/s 数通量 | Dirichlet 分配的速度 bin 数通量。 | Velocity-bin number flux (0–2 m/s). |
| `velocity_bin_1_0_2_ms_mass_flux_kg_m2_s` | 速度 0–2 m/s 质量通量 | Dirichlet 分配的速度 bin 质量通量。 | Velocity-bin mass flux (0–2 m/s). |
| `velocity_bin_2_2_4_ms_number_flux_m2_s` | 速度 2–4 m/s 数通量 | Dirichlet 分配的速度 bin 数通量。 | Velocity-bin number flux (2–4 m/s). |
| `velocity_bin_2_2_4_ms_mass_flux_kg_m2_s` | 速度 2–4 m/s 质量通量 | Dirichlet 分配的速度 bin 质量通量。 | Velocity-bin mass flux (2–4 m/s). |
| `velocity_bin_3_4_6_ms_number_flux_m2_s` | 速度 4–6 m/s 数通量 | Dirichlet 分配的速度 bin 数通量。 | Velocity-bin number flux (4–6 m/s). |
| `velocity_bin_3_4_6_ms_mass_flux_kg_m2_s` | 速度 4–6 m/s 质量通量 | Dirichlet 分配的速度 bin 质量通量。 | Velocity-bin mass flux (4–6 m/s). |
| `velocity_bin_4_6_8_ms_number_flux_m2_s` | 速度 6–8 m/s 数通量 | Dirichlet 分配的速度 bin 数通量。 | Velocity-bin number flux (6–8 m/s). |
| `velocity_bin_4_6_8_ms_mass_flux_kg_m2_s` | 速度 6–8 m/s 质量通量 | Dirichlet 分配的速度 bin 质量通量。 | Velocity-bin mass flux (6–8 m/s). |

**关键生成公式（简化）**
```text
Ut = 6.975 + 0.0033 * (T + 27.27)^2
ratio = max((U - Ut) / max(Ut, 0.1), 0)

snow_mass_flux = 0.0015 * ratio^3 * supply_factor * (1 + noise) + lognormal_noise
snow_number_flux = snow_mass_flux / (rho_ice * (4/3*pi*(d/2)^3)) * k

friction_velocity_ms ≈ 0.08 * U + noise
turbulence_intensity ≈ 0.15 + 0.05 * U + noise
ri_bulk ≈ 0.15*(T - Ts) - 0.08*U + noise

wind_snow_coupling = snow_mass_flux / max(U, 0.1)
threshold_exceedance = 1 if U > Ut else 0
net_radiation_est = 0.7 * solar - 20
sensible_heat_flux_est = 0.1 * (T - Ts) * U
latent_heat_flux_est = 0.02 * RH * U

stability_flag = stable if ri_bulk > 0.2, unstable if ri_bulk < -0.2 else neutral
```
说明：`T`=air_temperature_c，`Ts`=snow_surface_temperature_c，`U`=wind_speed_ms，`RH`=relative_humidity。

**Notebook流程概览**
- 读取合成数据，构造特征/目标，并进行标准化。
- `build_loaders` 构建窗口化样本（`window_size=24`, `horizon=6`）。
- 训练 `SnowTFT`，损失= MSE + PINN 物理惩罚（可调权重 `pin_weight`）。
- 绘制单批 horizon 预测曲线与拼接长序列曲线。
- 用预测值计算载荷（`ParticleBin` + `summarize_loads`）。

**模型与物理约束（简述）**
- 模型：轻量级 Transformer（`SnowTFT`），包含编码器/解码器结构：
  - 编码器输入窗口内历史特征（风、气温、湿度、雪通量等）。
  - 解码器输入未来已知协变量（辐射、风向）。
  - 输出多目标预测（气温、风速、雪质量通量、雪面温度）。
- 物理约束：来自 `legacy/windblown_snow/src/physics_losses.py` 的软惩罚项，叠加在 MSE 上：
  - **风速阈值约束**：低于阈值 `Ut` 时通量应接近 0。
  - **超阈值尺度**：通量随 `(wind/Ut)^3` 增长。
  - **摩阻单调性**：通量与摩阻速度的相关性约束。
  - **湿度抑制**：高湿度时通量降低。
  - **辐射/雪面温度抑制**：高辐射/接近融点时形成硬壳，抑制通量。
  - **稳定度抑制**（可选）：稳定层结时通量减弱。

**可调参数**
- `window_size`, `horizon`：在 `train_dummy.ipynb` 中配置。
- `pin_weight`：物理惩罚权重（建议 0.0～0.05）。
- 合成数据规模/频率：`generate_fake_snow_data.py` 中 `--rows/--freq`。

**常见问题**
- `ModuleNotFoundError: tft_model`：确保从仓库根目录运行 notebook，或在 notebook 中的路径初始化 cell 已正确定位到 `legacy/windblown_snow/src`。
- 预测近似直线：尝试降低 `pin_weight`、增大样本数量，或增强合成数据波动（风暴强度/阈值参数）。

**注意事项**
- 该 dummy 与当前 AntAWS 管线无直接关系，仅保留用于历史对照与快速试验。

---

### English Guide
**Purpose**
- This is a legacy wind-blown snow dummy workspace for quick validation of a lightweight Transformer model with a physics penalty.
- The data is synthetic and optimized for learnability, not for strict physical realism.

**Structure**
- `legacy/windblown_snow/train_dummy.ipynb`: main notebook.
- `legacy/windblown_snow/scripts/generate_fake_snow_data.py`: synthetic data generator.
- `legacy/windblown_snow/data/synthetic/windblown_snow_sample.csv`: generated sample data.
- `legacy/windblown_snow/src/`: legacy model/tools (`tft_model.py`, `snow_dataset.py`, `physics_losses.py`, `physics_calculations.py`).
- `legacy/windblown_snow/plots/`: plots saved during notebook runs.

**Quick Start**
1) Generate synthetic data:
   - `python legacy/windblown_snow/scripts/generate_fake_snow_data.py`
2) Open and run the notebook:
   - `legacy/windblown_snow/train_dummy.ipynb`

**Synthetic Data Logic (high level)**
- AR(1)-style temporal dynamics for wind/temperature/humidity.
- A storm regime + snow supply state to introduce persistence and threshold effects.
- Flux coupled to wind threshold `Ut`, supply state, and precipitation.

**Data Schema & Features (CSV fields)**
The table above lists all 42 fields with CN translation and EN description.

**Key Generating Equations (simplified)**
```text
Ut = 6.975 + 0.0033 * (T + 27.27)^2
ratio = max((U - Ut) / max(Ut, 0.1), 0)

snow_mass_flux = 0.0015 * ratio^3 * supply_factor * (1 + noise) + lognormal_noise
snow_number_flux = snow_mass_flux / (rho_ice * (4/3*pi*(d/2)^3)) * k

friction_velocity_ms ≈ 0.08 * U + noise
turbulence_intensity ≈ 0.15 + 0.05 * U + noise
ri_bulk ≈ 0.15*(T - Ts) - 0.08*U + noise

wind_snow_coupling = snow_mass_flux / max(U, 0.1)
threshold_exceedance = 1 if U > Ut else 0
net_radiation_est = 0.7 * solar - 20
sensible_heat_flux_est = 0.1 * (T - Ts) * U
latent_heat_flux_est = 0.02 * RH * U

stability_flag = stable if ri_bulk > 0.2, unstable if ri_bulk < -0.2 else neutral
```
Notes: `T`=air_temperature_c, `Ts`=snow_surface_temperature_c, `U`=wind_speed_ms, `RH`=relative_humidity.

**Notebook Flow**
- Load data, build features/targets, standardize.
- Build windowed sequences (`window_size=24`, `horizon=6`).
- Train `SnowTFT` with `MSE + pin_weight * physics_penalty`.
- Plot per-horizon predictions and concatenated timelines.
- Derive loads from predictions (`ParticleBin` + `summarize_loads`).

**Model & Physics Constraints**
- Model: a lightweight Transformer (`SnowTFT`) with encoder/decoder:
  - Encoder ingests historical features (wind, temperature, humidity, flux proxies).
  - Decoder ingests known future covariates (solar radiation, wind direction).
  - Outputs multi-target forecasts (air temperature, wind speed, mass flux, snow surface temperature).
- Physics constraints: soft penalties from `legacy/windblown_snow/src/physics_losses.py`, added to MSE:
  - **Wind threshold**: flux should be near zero when wind < `Ut`.
  - **Above-threshold scaling**: flux grows with `(wind/Ut)^3`.
  - **Friction monotonicity**: flux correlates with friction velocity.
  - **Humidity suppression**: high RH reduces flux.
  - **Radiation/temperature crusting**: high radiation near melting suppresses flux.
  - **Stability suppression** (if flags available): stable stratification reduces flux.

**Tuning Knobs**
- `window_size`, `horizon`: configured in `train_dummy.ipynb`.
- `pin_weight`: physics penalty weight (try 0.0–0.05).
- Synthetic dataset size/frequency: `--rows/--freq` in `generate_fake_snow_data.py`.

**Troubleshooting**
- `ModuleNotFoundError: tft_model`: ensure the notebook runs from repo root, or verify the path setup cell locates `legacy/windblown_snow/src`.
- Flat predictions: reduce `pin_weight`, increase sample size, or increase synthetic variability.

**Notes**
- This legacy dummy is intentionally separated from the current AntAWS pipeline.
