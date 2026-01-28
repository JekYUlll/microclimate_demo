from __future__ import annotations

from typing import Dict, List

import torch


def physics_penalty(y_pred: torch.Tensor, scaler, cfg: Dict, target_cols: List[str]) -> torch.Tensor:
    """
    Soft physics constraints for AntAWS targets.

    y_pred: normalized predictions [B, H, T]
    scaler: StandardScaler for inverse transform
    """
    loss_cfg = cfg.get("loss", {})
    if loss_cfg.get("lambda_phys", 0.0) <= 0:
        return torch.zeros((), device=y_pred.device)

    # unnormalize
    mean = torch.tensor(scaler.target_mean, device=y_pred.device)
    std = torch.tensor(scaler.target_std, device=y_pred.device)
    y = y_pred * std + mean

    penalties = []
    if loss_cfg.get("use_bounds_term", True):
        if "wind_speed_ms" in target_cols:
            idx = target_cols.index("wind_speed_ms")
            penalties.append(torch.relu(-y[:, :, idx]).mean())
        if "relative_humidity_pct" in target_cols:
            idx = target_cols.index("relative_humidity_pct")
            penalties.append(torch.relu(y[:, :, idx] - 100).mean())
            penalties.append(torch.relu(-y[:, :, idx]).mean())

    if loss_cfg.get("use_clausius_clapeyron_term", True):
        temp_idx = None
        if "temperature_c" in target_cols:
            temp_idx = target_cols.index("temperature_c")
        elif "air_temperature_c" in target_cols:
            temp_idx = target_cols.index("air_temperature_c")
        rh_idx = target_cols.index("relative_humidity_pct") if "relative_humidity_pct" in target_cols else None

        if temp_idx is not None and rh_idx is not None:
            temp_c = y[:, :, temp_idx]
            rh = y[:, :, rh_idx]
            denom = torch.clamp(272.62 + temp_c, min=1.0)
            es_hpa = 6.11 * torch.exp(22.46 * temp_c / denom)
            vapor_hpa = (rh / 100.0) * es_hpa

            # Penalize supersaturation / negative vapor pressure.
            penalties.append(torch.relu(vapor_hpa - es_hpa).mean())
            penalties.append(torch.relu(-vapor_hpa).mean())

            if loss_cfg.get("use_vapor_pressure_smooth", True):
                diffs = vapor_hpa[:, 1:] - vapor_hpa[:, :-1]
                penalties.append(torch.nn.functional.smooth_l1_loss(diffs, torch.zeros_like(diffs)))

    if loss_cfg.get("use_coherence_term", True):
        diffs = y[:, 1:, :] - y[:, :-1, :]
        penalties.append(torch.nn.functional.smooth_l1_loss(diffs, torch.zeros_like(diffs)))

    if not penalties:
        return torch.zeros((), device=y_pred.device)
    return torch.stack(penalties).mean()
