# Wind-Blown Snow Data Schema

This note outlines the feature set and a portable storage format for extending the current EMD + sequence models (LSTM/TCN/TFT) to wind-blown snow forecasting. It assumes raw data arrive as time-stamped sensor feeds and may later be synthetically generated for training.

## Core observables
- `timestamp` (UTC, seconds precision)
- Meteorology: `air_temperature_c`, `relative_humidity`, `air_pressure_pa`, `wind_speed_ms`, `wind_direction_deg`, `solar_radiation_wm2`, `snow_surface_temperature_c`
- Snow transport: `snow_mass_flux_kg_m2_s` (areal mass flux), `snow_number_flux_m2_s` (optional if available)
- Spectra: size/velocity distributions as binned spectra (see proto below)
- Turbulence/flow: `turbulence_intensity`, `friction_velocity`, `ri_bulk` (bulk Richardson), `stability_flag` (stable/neutral/unstable)
- Optional environment: `snow_density`, `snow_grain_temp_c`, `visibility_m`, `precip_rate_mm_h`
- Quality: `quality_flag`, `data_source`, `missing_reason`

## Derived features (computed during preprocessing)
- Wind-blown snow coupling: `wind_snow_coupling = f(wind_speed, flux)`; `threshold_exceedance` (wind above saltation threshold)
- Energy balance proxies: `net_radiation_est`, `sensible_heat_flux_est`, `latent_heat_flux_est` (coarse)
- Temporal context: rolling means/variances (e.g., 10 min, 1 h), gradients (first differences), and harmonic terms (`sin/cos` of day-of-year)
- EMD outputs: IMF components and residuals per target variable stored alongside raw signals for reuse in downstream models

## Format choice
- Prefer **Protocol Buffers (proto3)** for compact, language-agnostic payloads and straightforward decoding into pandas/torch tensors. Existing `protobuf` dependency is already present in `environment.yml`.
- Thrift would also work but adds an extra runtime and is less common for ML data interchange in Python; proto is a simpler fit here.
- Store files as `*.pb` (binary) or `*.json` (text-format proto) and maintain a CSV/Parquet export for quick inspection.

## Proto layout
The accompanying `schemas/windblown_snow.proto` defines:
- `SpectrumBin`: bounded bin with size/velocity ranges and fluxes
- `SensorMeta`: location and device metadata
- `SnowObservation`: one timestamped record with scalar features plus spectra
- `SnowTimeSeries`: a batch of observations plus global attrs (sampling rate, coordinate system)

## Using the proto in this repo
1) Compile once: `protoc --python_out=src schemas/windblown_snow.proto` (adds `schemas/windblown_snow_pb2.py` under `src/schemas/`).
2) Load to DataFrame for EMD + models:
```python
from schemas import windblown_snow_pb2 as pb
obs_batch = pb.SnowTimeSeries.FromString(open("data/raw/snow.pb", "rb").read())
rows = []
for obs in obs_batch.observations:
    row = {
        "timestamp": obs.timestamp.ToDatetime(),
        "air_temperature_c": obs.air_temperature_c,
        "snow_mass_flux_kg_m2_s": obs.snow_mass_flux_kg_m2_s,
        "turbulence_intensity": obs.turbulence_intensity,
    }
    rows.append(row)
# df now feeds EMD decomposition + sequence datasets
```
3) When generating synthetic training data, emit the same proto so the pipeline remains identical between simulated and real feeds.

## Notes for synthetic data generation
- Match sampling cadence to sensors (e.g., 1–10 Hz for spectra, 1 min aggregates for model input). Aggregate to the model frequency during preprocessing.
- Simulate joint structure: couple wind speed/direction with flux and spectra; include diurnal cycles and gustiness. Inject controlled missingness and quality flags to stress-test the pipeline.
- Preserve spectrum bins even when flux is near zero to avoid variable-length drift; fill bins with zeros but keep bin edges constant.

