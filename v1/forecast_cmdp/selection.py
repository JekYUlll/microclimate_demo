from __future__ import annotations

import math
from statistics import median
from typing import Mapping


DEPLOYABLE_SELECTION_CRITERIA = (
    "mean_objective",
    "static_margin_guard",
    "static_margin_risk",
)


def choose_deployable_validation_row(
    rows: list[dict[str, object]],
    *,
    criterion: str,
    min_mean_margin: float = 0.0,
    min_start_margin: float = -math.inf,
    max_negative_starts: int = 1_000_000,
    require_guard_pass: bool = False,
    require_positive_center: bool = False,
    require_risk_band: bool = False,
    risk_min_q25_margin: float = -math.inf,
    risk_max_negative_starts: int = 1_000_000,
) -> dict[str, object] | None:
    if not rows:
        raise ValueError("No deployable validation rows to select from")
    criterion = str(criterion)
    if criterion not in DEPLOYABLE_SELECTION_CRITERIA:
        raise ValueError(f"Unsupported deployable selection criterion: {criterion}")
    if criterion == "mean_objective":
        return sorted(rows, key=_mean_objective_key)[0]
    for row in rows:
        row["objective_margin_median"] = _margin_quantile(row, q=0.5)
        row["objective_margin_q25"] = _margin_quantile(row, q=0.25)
        row["static_margin_positive_center"] = _has_positive_center(
            row,
            min_mean_margin=float(min_mean_margin),
        )
        row["static_margin_guard_pass"] = _passes_static_margin_guard(
            row,
            min_mean_margin=float(min_mean_margin),
            min_start_margin=float(min_start_margin),
            max_negative_starts=int(max_negative_starts),
        )
    if criterion == "static_margin_guard":
        selected = sorted(rows, key=_static_margin_guard_key)[0]
    else:
        selected = sorted(
            rows,
            key=lambda row: _static_margin_risk_key(
                row,
                min_mean_margin=float(min_mean_margin),
            ),
        )[0]
    if (
        bool(require_guard_pass)
        and not bool(selected.get("static_margin_guard_pass", False))
    ):
        return None
    if (
        bool(require_positive_center)
        and criterion == "static_margin_risk"
        and not bool(selected.get("static_margin_positive_center", False))
    ):
        return None
    if (
        bool(require_risk_band)
        and criterion == "static_margin_risk"
        and not _passes_static_risk_band(
            selected,
            min_mean_margin=float(min_mean_margin),
            min_q25_margin=float(risk_min_q25_margin),
            max_negative_starts=int(risk_max_negative_starts),
        )
    ):
        return None
    return selected


def _mean_objective_key(row: Mapping[str, object]) -> tuple[float, float, int]:
    return (
        _finite_float(row.get("objective"), default=math.inf),
        _finite_float(row.get("power_mean"), default=math.inf),
        int(row.get("warmup_abort_count", 0) or 0),
    )


def _static_margin_guard_key(row: Mapping[str, object]) -> tuple[int, float, int, float, float, int]:
    guard_pass = bool(row.get("static_margin_guard_pass", False))
    return (
        0 if guard_pass else 1,
        _finite_float(row.get("objective"), default=math.inf),
        int(row.get("negative_start_count", 1_000_000) or 0),
        -_finite_float(row.get("objective_margin_mean"), default=-math.inf),
        -_finite_float(row.get("objective_margin_min"), default=-math.inf),
        int(row.get("warmup_abort_count", 0) or 0),
    )


def _static_margin_risk_key(
    row: Mapping[str, object],
    *,
    min_mean_margin: float,
) -> tuple[int, int, int, float, float, float, float, float, float, int]:
    guard_pass = bool(row.get("static_margin_guard_pass", False))
    mean_margin = _finite_float(row.get("objective_margin_mean"), default=-math.inf)
    median_margin = _finite_float(row.get("objective_margin_median"), default=-math.inf)
    q25_margin = _finite_float(row.get("objective_margin_q25"), default=-math.inf)
    min_margin = _finite_float(row.get("objective_margin_min"), default=-math.inf)
    negative_starts = int(row.get("negative_start_count", 1_000_000) or 0)
    positive_center = _has_positive_center(row, min_mean_margin=float(min_mean_margin))
    return (
        0 if guard_pass else 1,
        0 if positive_center else 1,
        negative_starts,
        -median_margin,
        -q25_margin,
        -mean_margin,
        -min_margin,
        _finite_float(row.get("objective"), default=math.inf),
        _finite_float(row.get("power_mean"), default=math.inf),
        int(row.get("warmup_abort_count", 0) or 0),
    )


def _has_positive_center(row: Mapping[str, object], *, min_mean_margin: float) -> bool:
    mean_margin = _finite_float(row.get("objective_margin_mean"), default=-math.inf)
    median_margin = _finite_float(row.get("objective_margin_median"), default=-math.inf)
    return mean_margin >= float(min_mean_margin) and median_margin >= 0.0


def _passes_static_margin_guard(
    row: Mapping[str, object],
    *,
    min_mean_margin: float,
    min_start_margin: float,
    max_negative_starts: int,
) -> bool:
    mean_margin = _finite_float(row.get("objective_margin_mean"), default=-math.inf)
    min_margin = _finite_float(row.get("objective_margin_min"), default=-math.inf)
    negative_starts = int(row.get("negative_start_count", 1_000_000) or 0)
    return (
        mean_margin >= float(min_mean_margin)
        and min_margin >= float(min_start_margin)
        and negative_starts <= int(max_negative_starts)
    )


def _passes_static_risk_band(
    row: Mapping[str, object],
    *,
    min_mean_margin: float,
    min_q25_margin: float,
    max_negative_starts: int,
) -> bool:
    q25_margin = _finite_float(row.get("objective_margin_q25"), default=-math.inf)
    negative_starts = int(row.get("negative_start_count", 1_000_000) or 0)
    return (
        _has_positive_center(row, min_mean_margin=float(min_mean_margin))
        and q25_margin >= float(min_q25_margin)
        and negative_starts <= int(max_negative_starts)
    )


def _margin_quantile(row: Mapping[str, object], *, q: float) -> float:
    margins = _margin_values(row)
    if not margins:
        if q == 0.5:
            return _finite_float(row.get("objective_margin_mean"), default=-math.inf)
        return _finite_float(row.get("objective_margin_min"), default=-math.inf)
    values = sorted(margins)
    if q == 0.5:
        return float(median(values))
    pos = (len(values) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    frac = pos - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _margin_values(row: Mapping[str, object]) -> list[float]:
    static_values = row.get("static_start_objectives")
    candidate_values = row.get("candidate_start_objectives")
    if not isinstance(static_values, (list, tuple)) or not isinstance(candidate_values, (list, tuple)):
        return []
    out: list[float] = []
    for static, candidate in zip(static_values, candidate_values):
        margin = _finite_float(static, default=math.nan) - _finite_float(candidate, default=math.nan)
        if math.isfinite(margin):
            out.append(float(margin))
    return out


def _finite_float(value: object, *, default: float) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out
