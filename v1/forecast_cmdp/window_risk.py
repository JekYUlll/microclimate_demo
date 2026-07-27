from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import shutil
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .protocol import SelectedStarts, choose_non_overlapping_starts


BLOCKED_FEATURE_TOKENS = (
    "truth_future",
    "future_truth",
    "teacher_future",
    "validation_outcome",
    "final_outcome",
    "candidate_realized",
    "realized_margin",
)


@dataclass(frozen=True)
class RiskStartSplit:
    fit: tuple[int, ...]
    calibration: tuple[int, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class ControllerSpec:
    controller_id: str
    parameters: dict[str, object]


@dataclass(frozen=True)
class WindowOutcome:
    objective: float
    power_mean: float = float("nan")
    warmup_abort_count: int = 0
    constraint_violation_count: int = 0


@dataclass(frozen=True)
class WindowRiskRecord:
    seed: int
    split_name: str
    start: int
    anchor_action_idx: int
    anchor_mask: tuple[bool, ...]
    controller_id: str
    controller_config: dict[str, object]
    paired_seed_offset: int
    static_objective: float
    candidate_objective: float
    margin: float
    power_mean: float
    warmup_abort_count: int
    constraint_violation_count: int
    feature_vector: tuple[float, ...]
    feature_names: tuple[str, ...]

    @property
    def key(self) -> tuple[int, str, int, int, str, int]:
        return (
            int(self.seed),
            str(self.split_name),
            int(self.start),
            int(self.anchor_action_idx),
            str(self.controller_id),
            int(self.paired_seed_offset),
        )

    def to_json_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["anchor_mask"] = [bool(x) for x in self.anchor_mask]
        row["feature_vector"] = [float(x) for x in self.feature_vector]
        row["feature_names"] = [str(x) for x in self.feature_names]
        return row

    @classmethod
    def from_json_dict(cls, row: Mapping[str, object]) -> "WindowRiskRecord":
        return cls(
            seed=int(row["seed"]),
            split_name=str(row["split_name"]),
            start=int(row["start"]),
            anchor_action_idx=int(row["anchor_action_idx"]),
            anchor_mask=tuple(bool(x) for x in row["anchor_mask"]),  # type: ignore[arg-type]
            controller_id=str(row["controller_id"]),
            controller_config=dict(row["controller_config"]),  # type: ignore[arg-type]
            paired_seed_offset=int(row["paired_seed_offset"]),
            static_objective=float(row["static_objective"]),
            candidate_objective=float(row["candidate_objective"]),
            margin=float(row["margin"]),
            power_mean=float(row["power_mean"]),
            warmup_abort_count=int(row["warmup_abort_count"]),
            constraint_violation_count=int(row["constraint_violation_count"]),
            feature_vector=tuple(float(x) for x in row["feature_vector"]),  # type: ignore[arg-type]
            feature_names=tuple(str(x) for x in row["feature_names"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class WindowRiskDataset:
    features: np.ndarray
    margins: np.ndarray
    negative_labels: np.ndarray
    starts: np.ndarray
    anchor_action_indices: np.ndarray
    controller_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    records: tuple[WindowRiskRecord, ...]

    def save_npz(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            features=np.asarray(self.features, dtype=np.float32),
            margins=np.asarray(self.margins, dtype=np.float32),
            negative_labels=np.asarray(self.negative_labels, dtype=np.int8),
            starts=np.asarray(self.starts, dtype=np.int64),
            anchor_action_indices=np.asarray(self.anchor_action_indices, dtype=np.int64),
            controller_ids=np.asarray(self.controller_ids, dtype=str),
            feature_names=np.asarray(self.feature_names, dtype=str),
        )


StaticEvaluator = Callable[[int, int, int], WindowOutcome]
CandidateEvaluator = Callable[[int, int, ControllerSpec, int], WindowOutcome]
FeatureBuilder = Callable[
    [int, int, ControllerSpec, int],
    tuple[np.ndarray, tuple[str, ...]],
]
ControllerFilter = Callable[[int, ControllerSpec], bool]


def static_candidate_margin(static_objective: float, candidate_objective: float) -> float:
    """Return positive margin when the candidate improves over static."""

    return float(static_objective) - float(candidate_objective)


def split_train_risk_starts(
    truth: pd.DataFrame,
    *,
    bounds: tuple[int, int],
    window_steps: int,
    horizon: int,
    fit_count: int,
    calibration_count: int,
    selection: str,
    stride: int,
    event_column: str,
    seed: int,
) -> RiskStartSplit:
    fit_count = int(fit_count)
    calibration_count = int(calibration_count)
    if fit_count <= 0 or calibration_count <= 0:
        raise ValueError("fit_count and calibration_count must both be positive")
    selection_name = str(selection)
    effective_window_steps = int(window_steps)
    effective_horizon = int(horizon)
    if selection_name == "uniform":
        effective_window_steps = int(window_steps) + int(horizon) + 1
        effective_horizon = 0
    selected: SelectedStarts = choose_non_overlapping_starts(
        truth,
        bounds=bounds,
        window_steps=effective_window_steps,
        horizon=effective_horizon,
        count=fit_count + calibration_count,
        selection=selection_name,
        stride=int(stride),
        event_column=str(event_column),
        seed=int(seed),
    )
    ordered = tuple(sorted(int(x) for x in selected.starts))
    fit = ordered[:fit_count]
    calibration = ordered[fit_count:]
    assert_non_overlapping_windows(
        (*fit, *calibration),
        window_steps=int(window_steps),
        horizon=int(horizon),
    )
    diagnostics = {
        **dict(selected.diagnostics),
        "fit_count": len(fit),
        "calibration_count": len(calibration),
        "fit_starts": [int(x) for x in fit],
        "calibration_starts": [int(x) for x in calibration],
        "chronological_blocked": True,
    }
    return RiskStartSplit(fit=fit, calibration=calibration, diagnostics=diagnostics)


def assert_non_overlapping_windows(
    starts: Sequence[int],
    *,
    window_steps: int,
    horizon: int,
) -> None:
    ordered = sorted(int(x) for x in starts)
    required_gap = int(window_steps) + int(horizon) + 1
    for left, right in zip(ordered, ordered[1:]):
        if right - left < required_gap:
            raise ValueError(
                f"Risk windows overlap: starts {left} and {right}, required gap {required_gap}"
            )


def select_train_anchor_bank(
    static_table: pd.DataFrame,
    *,
    top_k: int,
) -> tuple[int, ...]:
    required = {"action_idx", "objective_loss_mean"}
    missing = required.difference(static_table.columns)
    if missing:
        raise ValueError(f"Static table is missing columns: {sorted(missing)}")
    ranked = (
        static_table.sort_values(
            ["objective_loss_mean", "power_mean", "warmup_abort_count", "action_idx"],
            na_position="last",
        )
        if {"power_mean", "warmup_abort_count"}.issubset(static_table.columns)
        else static_table.sort_values(["objective_loss_mean", "action_idx"], na_position="last")
    )
    values: list[int] = []
    for value in ranked["action_idx"].tolist():
        action_idx = int(value)
        if action_idx not in values:
            values.append(action_idx)
        if len(values) >= max(1, int(top_k)):
            break
    if not values:
        raise ValueError("Static table contains no anchor actions")
    return tuple(values)


def assign_balanced_anchors(
    starts: Sequence[int],
    anchor_bank: Sequence[int],
    *,
    anchors_per_start: int,
    always_include_best: bool = True,
) -> dict[int, tuple[int, ...]]:
    bank = tuple(dict.fromkeys(int(x) for x in anchor_bank))
    if not bank:
        raise ValueError("anchor_bank must not be empty")
    count = min(max(1, int(anchors_per_start)), len(bank))
    assignments: dict[int, tuple[int, ...]] = {}
    for start_idx, start in enumerate(starts):
        selected = [bank[0]] if bool(always_include_best) else []
        cursor = (
            int(start_idx)
            if bool(always_include_best)
            else int(start_idx) * int(count)
        )
        while len(selected) < count:
            if bool(always_include_best) and len(bank) > 1:
                candidate = bank[1 + (cursor % (len(bank) - 1))]
            else:
                candidate = bank[cursor % len(bank)]
            cursor += 1
            if candidate not in selected:
                selected.append(candidate)
        assignments[int(start)] = tuple(selected)
    return assignments


def audit_feature_names(feature_names: Sequence[str]) -> dict[str, object]:
    names = tuple(str(x) for x in feature_names)
    blocked = sorted(
        name
        for name in names
        if any(token in name.lower() for token in BLOCKED_FEATURE_TOKENS)
    )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    return {
        "feature_count": len(names),
        "blocked_features": blocked,
        "duplicate_features": duplicates,
        "pass": not blocked and not duplicates and bool(names),
    }


def build_window_risk_dataset(records: Sequence[WindowRiskRecord]) -> WindowRiskDataset:
    rows = tuple(records)
    if not rows:
        raise ValueError("No window-risk records")
    feature_names = rows[0].feature_names
    width = len(feature_names)
    if width == 0:
        raise ValueError("Window-risk records have no feature names")
    for row in rows:
        if row.feature_names != feature_names:
            raise ValueError("Window-risk records do not share one feature schema")
        if len(row.feature_vector) != width:
            raise ValueError("Window-risk feature vector width does not match schema")
        expected = static_candidate_margin(row.static_objective, row.candidate_objective)
        if not np.isclose(float(row.margin), expected):
            raise ValueError("Window-risk record uses an inconsistent margin sign")
    return WindowRiskDataset(
        features=np.asarray([row.feature_vector for row in rows], dtype=np.float32),
        margins=np.asarray([row.margin for row in rows], dtype=np.float32),
        negative_labels=np.asarray([float(row.margin) < 0.0 for row in rows], dtype=np.int8),
        starts=np.asarray([row.start for row in rows], dtype=np.int64),
        anchor_action_indices=np.asarray([row.anchor_action_idx for row in rows], dtype=np.int64),
        controller_ids=tuple(row.controller_id for row in rows),
        feature_names=feature_names,
        records=rows,
    )


def filter_exact_anchor_boundaries(
    dataset: WindowRiskDataset,
) -> tuple[WindowRiskDataset, dict[str, int]]:
    names = tuple(str(name) for name in dataset.feature_names)
    previous_prefix = "residual_boundary_previous_mask_"
    anchor_prefix = "residual_anchor_mask_"
    previous = {
        name.removeprefix(previous_prefix): idx
        for idx, name in enumerate(names)
        if name.startswith(previous_prefix)
    }
    anchors = {
        name.removeprefix(anchor_prefix): idx
        for idx, name in enumerate(names)
        if name.startswith(anchor_prefix)
    }
    if not previous and not anchors:
        return dataset, {
            "input_rows": int(len(dataset.records)),
            "exact_rows": int(len(dataset.records)),
            "dropped_rows": 0,
        }
    if set(previous) != set(anchors) or not previous:
        raise ValueError(
            "Residual boundary filtering requires matching previous/anchor masks"
        )
    ordered = tuple(sorted(previous))
    previous_values = dataset.features[
        :,
        [previous[sensor_id] for sensor_id in ordered],
    ]
    anchor_values = dataset.features[
        :,
        [anchors[sensor_id] for sensor_id in ordered],
    ]
    keep = np.all(
        np.isclose(previous_values, anchor_values, atol=1.0e-6),
        axis=1,
    )
    selected = [
        record
        for record, retain in zip(dataset.records, keep, strict=True)
        if bool(retain)
    ]
    if not selected:
        raise ValueError("Exact-anchor filtering removed every residual row")
    filtered = build_window_risk_dataset(selected)
    return filtered, {
        "input_rows": int(len(dataset.records)),
        "exact_rows": int(len(selected)),
        "dropped_rows": int(len(dataset.records) - len(selected)),
    }


def collect_paired_window_risk_dataset(
    *,
    out_dir: str | Path,
    seed: int,
    split_name: str,
    starts: Sequence[int],
    anchor_assignments: Mapping[int, Sequence[int]],
    anchor_masks: Mapping[int, Sequence[bool]],
    controllers: Sequence[ControllerSpec],
    static_evaluator: StaticEvaluator,
    candidate_evaluator: CandidateEvaluator,
    feature_builder: FeatureBuilder,
    controller_filter: ControllerFilter | None = None,
    seed_offset_base: int = 210_000,
) -> WindowRiskDataset:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    jsonl_path = output / "window_risk_rows.jsonl"
    records = load_window_risk_records(jsonl_path)
    completed = {record.key for record in records}
    static_cache: dict[tuple[int, int, int], WindowOutcome] = {}

    for start_idx, start in enumerate(int(x) for x in starts):
        anchors = tuple(int(x) for x in anchor_assignments[int(start)])
        for anchor_pos, anchor_idx in enumerate(anchors):
            if anchor_idx not in anchor_masks:
                raise ValueError(f"Missing mask for anchor action {anchor_idx}")
            anchor_mask = tuple(bool(x) for x in anchor_masks[anchor_idx])
            for controller in controllers:
                if controller_filter is not None and not bool(
                    controller_filter(int(anchor_idx), controller)
                ):
                    continue
                paired_seed_offset = (
                    int(seed_offset_base)
                    + int(start_idx) * 10_000
                    + int(anchor_pos) * 1_000
                )
                key = (
                    int(seed),
                    str(split_name),
                    int(start),
                    int(anchor_idx),
                    str(controller.controller_id),
                    int(paired_seed_offset),
                )
                if key in completed:
                    continue
                static_key = (int(start), int(anchor_idx), int(paired_seed_offset))
                if static_key not in static_cache:
                    static_cache[static_key] = static_evaluator(
                        int(start),
                        int(anchor_idx),
                        int(paired_seed_offset),
                    )
                static_outcome = static_cache[static_key]
                feature, feature_names = feature_builder(
                    int(start),
                    int(anchor_idx),
                    controller,
                    int(paired_seed_offset),
                )
                audit = audit_feature_names(feature_names)
                if not bool(audit["pass"]):
                    raise ValueError(f"Window-risk feature audit failed: {audit}")
                feature_arr = np.asarray(feature, dtype=float).reshape(-1)
                if feature_arr.size != len(feature_names) or not np.all(np.isfinite(feature_arr)):
                    raise ValueError("Window-risk features must be finite and match feature_names")
                candidate_outcome = candidate_evaluator(
                    int(start),
                    int(anchor_idx),
                    controller,
                    int(paired_seed_offset),
                )
                record = WindowRiskRecord(
                    seed=int(seed),
                    split_name=str(split_name),
                    start=int(start),
                    anchor_action_idx=int(anchor_idx),
                    anchor_mask=anchor_mask,
                    controller_id=str(controller.controller_id),
                    controller_config=dict(controller.parameters),
                    paired_seed_offset=int(paired_seed_offset),
                    static_objective=float(static_outcome.objective),
                    candidate_objective=float(candidate_outcome.objective),
                    margin=static_candidate_margin(
                        static_outcome.objective,
                        candidate_outcome.objective,
                    ),
                    power_mean=float(candidate_outcome.power_mean),
                    warmup_abort_count=int(candidate_outcome.warmup_abort_count),
                    constraint_violation_count=int(candidate_outcome.constraint_violation_count),
                    feature_vector=tuple(float(x) for x in feature_arr),
                    feature_names=tuple(str(x) for x in feature_names),
                )
                append_window_risk_record(jsonl_path, record)
                records.append(record)
                completed.add(record.key)

    dataset = build_window_risk_dataset(records)
    write_window_risk_artifacts(output, dataset)
    return dataset


def append_window_risk_record(path: str | Path, record: WindowRiskRecord) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_json_dict(), sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()


def load_window_risk_records(path: str | Path) -> list[WindowRiskRecord]:
    target = Path(path)
    if not target.exists():
        return []
    records: list[WindowRiskRecord] = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            records.append(WindowRiskRecord.from_json_dict(json.loads(text)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid window-risk JSONL row {line_number}: {exc}") from exc
    return records


def refresh_window_risk_features(
    output: str | Path,
    *,
    feature_builder: FeatureBuilder,
) -> WindowRiskDataset:
    out_dir = Path(output)
    jsonl_path = out_dir / "window_risk_rows.jsonl"
    records = load_window_risk_records(jsonl_path)
    if not records:
        raise ValueError(f"No window-risk records to refresh in {out_dir}")
    refreshed: list[WindowRiskRecord] = []
    for record in records:
        controller = ControllerSpec(
            controller_id=str(record.controller_id),
            parameters=dict(record.controller_config),
        )
        feature, feature_names = feature_builder(
            int(record.start),
            int(record.anchor_action_idx),
            controller,
            int(record.paired_seed_offset),
        )
        audit = audit_feature_names(feature_names)
        if not bool(audit["pass"]):
            raise ValueError(f"Refreshed window-risk feature audit failed: {audit}")
        feature_arr = np.asarray(feature, dtype=float).reshape(-1)
        if feature_arr.size != len(feature_names) or not np.all(np.isfinite(feature_arr)):
            raise ValueError("Refreshed features must be finite and match feature_names")
        refreshed.append(
            replace(
                record,
                feature_vector=tuple(float(x) for x in feature_arr),
                feature_names=tuple(str(x) for x in feature_names),
            )
        )
    backup_path = out_dir / "window_risk_rows_pre_refresh.jsonl"
    if not backup_path.exists():
        shutil.copy2(jsonl_path, backup_path)
    temporary = out_dir / "window_risk_rows.refreshing.jsonl"
    with temporary.open("w", encoding="utf-8") as handle:
        for record in refreshed:
            handle.write(json.dumps(record.to_json_dict(), sort_keys=True, ensure_ascii=True) + "\n")
    temporary.replace(jsonl_path)
    dataset = build_window_risk_dataset(refreshed)
    write_window_risk_artifacts(out_dir, dataset)
    return dataset


def write_window_risk_artifacts(output: str | Path, dataset: WindowRiskDataset) -> None:
    out_dir = Path(output)
    rows = []
    for record in dataset.records:
        row = record.to_json_dict()
        row["anchor_mask"] = json.dumps(row["anchor_mask"], separators=(",", ":"))
        row["controller_config"] = json.dumps(row["controller_config"], sort_keys=True, separators=(",", ":"))
        row["feature_vector"] = json.dumps(row["feature_vector"], separators=(",", ":"))
        row["feature_names"] = json.dumps(row["feature_names"], separators=(",", ":"))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "window_risk_rows.csv", index=False)
    dataset.save_npz(out_dir / "window_risk_dataset.npz")
    (out_dir / "window_risk_feature_schema.json").write_text(
        json.dumps(
            {
                "feature_names": list(dataset.feature_names),
                "audit": audit_feature_names(dataset.feature_names),
                "margin_definition": "static_objective - candidate_objective",
                "sample_unit": "one paired full-window rollout outcome",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (out_dir / "window_risk_collection_manifest.json").write_text(
        json.dumps(
            {
                "rows": int(dataset.margins.size),
                "independent_starts": int(np.unique(dataset.starts).size),
                "anchors": int(np.unique(dataset.anchor_action_indices).size),
                "controllers": len(set(dataset.controller_ids)),
                "positive_rate": float(np.mean(dataset.margins > 0.0)),
                "negative_rate": float(np.mean(dataset.margins < 0.0)),
                "margin_mean": float(np.mean(dataset.margins)),
                "margin_q25": float(np.quantile(dataset.margins, 0.25)),
                "margin_min": float(np.min(dataset.margins)),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def controller_specs_from_grid(
    grid: Iterable[Mapping[str, object]],
    *,
    prefix: str = "proxy",
) -> tuple[ControllerSpec, ...]:
    specs = []
    for idx, parameters in enumerate(grid):
        specs.append(
            ControllerSpec(
                controller_id=f"{prefix}_{idx:03d}",
                parameters=dict(parameters),
            )
        )
    return tuple(specs)
