from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

import torch


def physics_penalty(
    batch_x: torch.Tensor,
    batch_y_hat: torch.Tensor,
    known_cols: Sequence[str],
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    index_map: Optional[Mapping[str, int]] = None,
    target_cols: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """
    Physics-inspired penalties for wind-blown snow forecasting.

    Assumptions:
    - batch_x is z-scored using feat_mean/std in the same column order as known_cols
    - batch_y_hat is in physical units (unnormalized)
    """
    def idx(name: str) -> int:
        if index_map is not None and name in index_map:
            return index_map[name]
        if name in known_cols:
            return known_cols.index(name)
        return -1

    required = [
        "wind_speed_ms",
        "friction_velocity_ms",
        "air_temperature_c",
        "snow_surface_temperature_c",
        "solar_radiation_wm2",
        "relative_humidity",
    ]
    missing = [r for r in required if idx(r) == -1]
    if missing:
        # Gracefully skip physics penalty if required drivers are absent
        return torch.zeros((), device=batch_x.device)

    wind_idx = idx("wind_speed_ms")
    fric_idx = idx("friction_velocity_ms")
    temp_idx = idx("air_temperature_c")
    snow_temp_idx = idx("snow_surface_temperature_c")
    solar_idx = idx("solar_radiation_wm2")
    rh_idx = idx("relative_humidity")
    stability_cols: List[int] = [index_map[c] if index_map and c in index_map else i for i, c in enumerate(known_cols) if c.startswith("stability_flag_")]

    # Un-normalize drivers
    wind = batch_x[:, :, wind_idx] * feat_std[wind_idx] + feat_mean[wind_idx]
    fric = batch_x[:, :, fric_idx] * feat_std[fric_idx] + feat_mean[fric_idx]
    air_temp = batch_x[:, :, temp_idx] * feat_std[temp_idx] + feat_mean[temp_idx]
    snow_temp = batch_x[:, :, snow_temp_idx] * feat_std[snow_temp_idx] + feat_mean[snow_temp_idx]
    solar = batch_x[:, :, solar_idx] * feat_std[solar_idx] + feat_mean[solar_idx]
    rh = batch_x[:, :, rh_idx] * feat_std[rh_idx] + feat_mean[rh_idx]

    flux_idx = 0
    if target_cols is not None and "snow_mass_flux_kg_m2_s" in target_cols:
        flux_idx = target_cols.index("snow_mass_flux_kg_m2_s")
    flux_hat = batch_y_hat[:, :, flux_idx]
    penalties = []

    # 1) 启动风速阈值 (Li & Pomeroy 1997): Ut = 6.975 + 0.0033*(T+27.27)^2
    temp_last = air_temp[:, -1]
    Ut = 6.975 + 0.0033 * (temp_last + 27.27) ** 2
    wind_last = wind[:, -1]
    flux_last = flux_hat[:, -1]
    below_mask = torch.relu(Ut - wind_last) / (Ut + 1e-6)
    penalties.append(((below_mask * torch.relu(flux_last)) ** 2).mean())

    # 2) 超阈值后雪输送 ~ (wind/Ut)^3
    ratio = torch.clamp((wind_last - Ut) / (Ut + 1e-3), min=0.0)
    expected_flux = torch.clamp(ratio ** 3, max=10.0)
    scale = flux_hat.abs().mean() + 1e-6
    penalties.append(((flux_last / scale - expected_flux) ** 2).mean())

    # 3) 摩阻单调性：排序相关
    rank_fric = torch.argsort(torch.argsort(fric[:, -1]))
    rank_flux = torch.argsort(torch.argsort(flux_last))
    corr = torch.corrcoef(torch.stack([rank_fric.float(), rank_flux.float()]))[0, 1]
    penalties.append(1 - torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))

    # 4) 湿度接近饱和时抑制通量
    rh_last = torch.clamp(rh[:, -1], 0, 120)
    sat_factor = torch.clamp((rh_last - 80.0) / 40.0, min=0.0)
    penalties.append(((sat_factor * flux_last) ** 2).mean())

    # 5) 辐射+雪面温度：高辐射/接近融点形成硬壳抑制通量
    rad_factor = torch.clamp((solar[:, -1] - 300.0) / 400.0, min=0.0)
    temp_factor = torch.exp(-((snow_temp[:, -1] + 2.0) / 5.0) ** 2)
    crust_penalty = (rad_factor * temp_factor * torch.relu(flux_last)) ** 2
    penalties.append(crust_penalty.mean())

    # 6) 稳定度抑制（若存在稳定度哑变量）
    if stability_cols:
        stable_mask = batch_x[:, -1, stability_cols].sum(dim=1)
        penalties.append(((stable_mask * flux_last) ** 2).mean())

    if not penalties:
        return torch.zeros((), device=batch_x.device)

    weights = [0.25, 0.2, 0.15, 0.15, 0.15, 0.1][: len(penalties)]
    total = torch.zeros((), device=batch_x.device, dtype=penalties[0].dtype)
    for w, p in zip(weights, penalties):
        total = total + float(w) * p
    return total
