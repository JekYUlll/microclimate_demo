from __future__ import annotations

import math
from typing import Mapping


DEPLOYABLE_SELECTION_CRITERIA = ("mean_objective", "static_margin_guard")


def choose_deployable_validation_row(
    rows: list[dict[str, object]],
    *,
    criterion: str,
    min_mean_margin: float = 0.0,
    min_start_margin: float = -math.inf,
    max_negative_starts: int = 1_000_000,
) -> dict[str, object]:
    if not rows:
        raise ValueError("No deployable validation rows to select from")
    criterion = str(criterion)
    if criterion not in DEPLOYABLE_SELECTION_CRITERIA:
        raise ValueError(f"Unsupported deployable selection criterion: {criterion}")
    if criterion == "mean_objective":
        return sorted(rows, key=_mean_objective_key)[0]
    for row in rows:
        row["static_margin_guard_pass"] = _passes_static_margin_guard(
            row,
            min_mean_margin=float(min_mean_margin),
            min_start_margin=float(min_start_margin),
            max_negative_starts=int(max_negative_starts),
        )
    return sorted(rows, key=_static_margin_guard_key)[0]


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


def _finite_float(value: object, *, default: float) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out
