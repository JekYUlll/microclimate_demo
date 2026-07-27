#!/usr/bin/env python3
"""Train split-compliant intervention-effect verifiers.

The input rows are paired raw-dynamic-vs-anchor intervention audits.  This
script does not touch validation/final rollouts.  It asks a narrower question:
can train-only effect labels produce a seed-held-out verifier whose accepted
dynamic deviations have positive mean and non-negative lower tail?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RESULT_COLUMNS = {
    "effect_margin",
    "anchor_objective",
    "dynamic_objective",
    "anchor_oracle_loss_mean",
    "dynamic_oracle_loss_mean",
}

ADMIN_COLUMNS = {
    "split",
    "seed",
    "start",
    "current_idx",
    "raw_action_bits",
    "selected_action_bits",
    "previous_action_bits",
}

GUARD_COLUMNS = {
    "selected_is_anchor",
    "anchor_guard_applied",
    "component_guard_applied",
}

COMPACT_EXACT_COLUMNS = {
    "predicted_anchor_minus_raw",
    "component_task_margin_mean",
    "component_total_margin_mean",
    "prefix_steps",
    "branch_steps",
    "relative_step",
    "replan_id",
    "elapsed_steps",
    "soc",
    "soc_ratio",
    "energy_deficit_steps",
    "energy_deficit_total",
    "raw_anchor_hamming",
    "raw_previous_hamming",
    "anchor_previous_hamming",
    "raw_active_count",
    "anchor_active_count",
}

COMPACT_PREFIXES = (
    "context_learned_event_p_",
    "last_obs_",
    "observed_mask_",
    "history_mean_",
    "history_std_",
    "mask_history_mean_",
)

DEFAULT_METHODS = (
    "score_predicted_advantage",
    "score_component_total",
    "score_component_task",
    "logistic_cls",
    "ridge_reg",
    "hist_gbdt_cls",
    "hist_gbdt_reg",
    "rf_cls",
    "rf_reg",
)


@dataclass(frozen=True)
class VerifierConfig:
    input: str
    output: str
    feature_mode: str
    methods: list[str]
    base_scopes: list[str]
    calibration_modes: list[str]
    min_calib_accept: int
    min_calib_seed_accept: int
    min_test_accept: int
    min_mean: float
    min_q25: float
    positive_margin: float
    random_state: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate intervention-effect verifiers with leave-one-seed "
            "group validation."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "v1/artifacts/robust_rollout_multiseed_summary_20260607/"
            "robust_intervention_effect_train_multiseed_rows.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "v1/artifacts/robust_rollout_multiseed_summary_20260607/"
            "effect_verifier_loo_20260607"
        ),
    )
    parser.add_argument(
        "--feature-mode",
        choices=("causal", "with_guard", "compact"),
        default="causal",
        help=(
            "causal excludes current guard decisions; with_guard includes them; "
            "compact keeps only hand-selected causal/runtime summaries."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Verifier score models to evaluate.",
    )
    parser.add_argument(
        "--base-scopes",
        nargs="+",
        choices=("selected_dynamic", "all_raw"),
        default=["selected_dynamic", "all_raw"],
        help=(
            "selected_dynamic means the verifier only rejects dynamics that the "
            "current planner would execute; all_raw is diagnostic."
        ),
    )
    parser.add_argument(
        "--calibration-modes",
        nargs="+",
        choices=("aggregate", "per_seed"),
        default=["aggregate", "per_seed"],
    )
    parser.add_argument("--min-calib-accept", type=int, default=12)
    parser.add_argument("--min-calib-seed-accept", type=int, default=3)
    parser.add_argument("--min-test-accept", type=int, default=3)
    parser.add_argument("--min-mean", type=float, default=0.0)
    parser.add_argument("--min-q25", type=float, default=0.0)
    parser.add_argument(
        "--positive-margin",
        type=float,
        default=0.0,
        help="Classification target is effect_margin > positive_margin.",
    )
    parser.add_argument("--random-state", type=int, default=20260607)
    return parser.parse_args()


def require_sklearn():
    try:
        from sklearn.ensemble import (  # noqa: F401
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.impute import SimpleImputer  # noqa: F401
        from sklearn.linear_model import LogisticRegression, Ridge  # noqa: F401
        from sklearn.pipeline import make_pipeline  # noqa: F401
        from sklearn.preprocessing import StandardScaler  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard.
        raise SystemExit(
            "scikit-learn is required for train_effect_verifier.py"
        ) from exc


def boolish(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0
    lowered = series.astype(str).str.lower()
    return lowered.isin({"1", "true", "yes", "y"})


def numeric_table(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    table = pd.DataFrame(index=df.index)
    for column in feature_columns:
        values = df[column]
        if pd.api.types.is_bool_dtype(values):
            table[column] = values.astype(float)
        elif pd.api.types.is_numeric_dtype(values):
            table[column] = pd.to_numeric(values, errors="coerce")
        else:
            coerced = pd.to_numeric(values, errors="coerce")
            if coerced.notna().any():
                table[column] = coerced
    table = table.replace([np.inf, -np.inf], np.nan)
    return table


def select_feature_columns(df: pd.DataFrame, feature_mode: str) -> list[str]:
    exclude = set(RESULT_COLUMNS) | set(ADMIN_COLUMNS)
    if feature_mode != "with_guard":
        exclude |= set(GUARD_COLUMNS)

    columns: list[str] = []
    for column in df.columns:
        if column in exclude:
            continue
        if feature_mode == "compact":
            keep = column in COMPACT_EXACT_COLUMNS or column.startswith(COMPACT_PREFIXES)
            if not keep:
                continue
        values = df[column]
        if pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values):
            columns.append(column)
            continue
        coerced = pd.to_numeric(values, errors="coerce")
        if coerced.notna().any():
            columns.append(column)
    return columns


def build_model(method: str, random_state: int):
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if method.startswith("score_"):
        return None
    if method == "logistic_cls":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=random_state,
            ),
        )
    if method == "ridge_reg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0, random_state=random_state),
        )
    if method == "hist_gbdt_cls":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            l2_regularization=0.01,
            min_samples_leaf=12,
            random_state=random_state,
        )
    if method == "hist_gbdt_reg":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            l2_regularization=0.01,
            min_samples_leaf=12,
            random_state=random_state,
        )
    if method == "rf_cls":
        return RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    if method == "rf_reg":
        return RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"unknown method: {method}")


def score_column_for_method(method: str) -> str:
    if method == "score_predicted_advantage":
        return "predicted_anchor_minus_raw"
    if method == "score_component_total":
        return "component_total_margin_mean"
    if method == "score_component_task":
        return "component_task_margin_mean"
    raise ValueError(f"{method} is not a scalar-score method")


def fit_scores(
    method: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_score: pd.DataFrame,
    df_train: pd.DataFrame,
    df_score: pd.DataFrame,
    positive_margin: float,
    random_state: int,
) -> tuple[np.ndarray, object | None]:
    if method.startswith("score_"):
        column = score_column_for_method(method)
        if column not in df_score.columns:
            raise ValueError(f"missing score column {column}")
        return pd.to_numeric(df_score[column], errors="coerce").to_numpy(dtype=float), None

    model = build_model(method, random_state)
    if method.endswith("_cls"):
        labels = (y_train > positive_margin).astype(int)
        if len(np.unique(labels)) < 2:
            return np.full(len(x_score), np.nan), model
        model.fit(x_train, labels)
        proba = model.predict_proba(x_score)
        return np.asarray(proba[:, 1], dtype=float), model

    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_score), dtype=float), model


def effect_stats(effect: np.ndarray, accepted: np.ndarray) -> dict[str, float | int | None]:
    accepted = np.asarray(accepted, dtype=bool)
    values = np.asarray(effect, dtype=float)[accepted]
    if len(values) == 0:
        return {
            "n": 0,
            "mean": None,
            "q25": None,
            "positive_rate": None,
            "min": None,
        }
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "q25": float(np.quantile(values, 0.25)),
        "positive_rate": float(np.mean(values > 0.0)),
        "min": float(np.min(values)),
    }


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.asarray([], dtype=float)
    quantiles = np.linspace(0.0, 1.0, 101)
    thresholds = np.unique(np.quantile(finite, quantiles))
    return np.sort(thresholds)


def calibration_passes(
    df: pd.DataFrame,
    effect: np.ndarray,
    accepted: np.ndarray,
    mode: str,
    min_accept: int,
    min_seed_accept: int,
    min_mean: float,
    min_q25: float,
) -> tuple[bool, dict[str, float | int | None]]:
    aggregate = effect_stats(effect, accepted)
    if int(aggregate["n"]) < min_accept:
        return False, aggregate
    if aggregate["mean"] is None or aggregate["q25"] is None:
        return False, aggregate
    if float(aggregate["mean"]) < min_mean or float(aggregate["q25"]) < min_q25:
        return False, aggregate
    if mode == "aggregate":
        return True, aggregate

    for seed in sorted(df["seed"].unique()):
        seed_mask = df["seed"].to_numpy() == seed
        seed_stats = effect_stats(effect[seed_mask], accepted[seed_mask])
        if int(seed_stats["n"]) < min_seed_accept:
            return False, aggregate
        if seed_stats["mean"] is None or seed_stats["q25"] is None:
            return False, aggregate
        if float(seed_stats["mean"]) < min_mean or float(seed_stats["q25"]) < min_q25:
            return False, aggregate
    return True, aggregate


def choose_threshold(
    df_calib: pd.DataFrame,
    scores_calib: np.ndarray,
    effect_calib: np.ndarray,
    base_mask_calib: np.ndarray,
    mode: str,
    min_accept: int,
    min_seed_accept: int,
    min_mean: float,
    min_q25: float,
) -> tuple[float | None, dict[str, float | int | None], str]:
    best_threshold: float | None = None
    best_stats: dict[str, float | int | None] = effect_stats(effect_calib, np.zeros_like(base_mask_calib))
    best_key: tuple[float, float, float] | None = None
    for threshold in candidate_thresholds(scores_calib):
        accepted = base_mask_calib & (scores_calib >= threshold)
        passes, stats = calibration_passes(
            df_calib,
            effect_calib,
            accepted,
            mode=mode,
            min_accept=min_accept,
            min_seed_accept=min_seed_accept,
            min_mean=min_mean,
            min_q25=min_q25,
        )
        if not passes:
            continue
        key = (
            float(stats["n"] or 0),
            float(stats["q25"] or 0.0),
            float(stats["mean"] or 0.0),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_stats = stats
    if best_threshold is None:
        return None, best_stats, "no_calibration_threshold"
    return best_threshold, best_stats, "ok"


def base_mask(df: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "all_raw":
        return np.ones(len(df), dtype=bool)
    if scope == "selected_dynamic":
        if "selected_is_anchor" not in df.columns:
            raise ValueError("selected_dynamic scope requires selected_is_anchor")
        return ~boolish(df["selected_is_anchor"]).to_numpy()
    raise ValueError(f"unknown base scope: {scope}")


def pooled_summary(rows: pd.DataFrame, min_test_accept: int) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    group_cols = ["method", "feature_mode", "base_scope", "calibration_mode"]
    for key, group in rows.groupby(group_cols, dropna=False):
        safe = (
            (group["heldout_accept_n"] >= min_test_accept)
            & (group["heldout_mean"] > 0.0)
            & (group["heldout_q25"] >= 0.0)
        )
        accepted = group["heldout_accept_n"] > 0
        pooled_values: list[float] = []
        for values in group["heldout_effect_values"]:
            if isinstance(values, list):
                pooled_values.extend(float(v) for v in values)
        pooled = np.asarray(pooled_values, dtype=float)
        row = {
            "method": key[0],
            "feature_mode": key[1],
            "base_scope": key[2],
            "calibration_mode": key[3],
            "heldout_seed_count": int(len(group)),
            "accepted_seed_count": int(accepted.sum()),
            "safe_seed_count": int(safe.sum()),
            "total_accept_n": int(group["heldout_accept_n"].sum()),
            "mean_accept_n": float(group["heldout_accept_n"].mean()),
            "pooled_mean": float(np.mean(pooled)) if len(pooled) else np.nan,
            "pooled_q25": float(np.quantile(pooled, 0.25)) if len(pooled) else np.nan,
            "min_seed_mean": float(group.loc[accepted, "heldout_mean"].min()) if accepted.any() else np.nan,
            "min_seed_q25": float(group.loc[accepted, "heldout_q25"].min()) if accepted.any() else np.nan,
            "ok_threshold_seeds": int((group["status"] == "ok").sum()),
        }
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if len(summary) == 0:
        return summary
    return summary.sort_values(
        ["safe_seed_count", "pooled_q25", "pooled_mean", "total_accept_n"],
        ascending=[False, False, False, False],
    )


def write_markdown(
    output: Path,
    config: VerifierConfig,
    summary: pd.DataFrame,
    rows: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Effect Verifier LOO Summary")
    lines.append("")
    lines.append("## Protocol")
    lines.append(f"- Input: `{config.input}`")
    lines.append(f"- Feature mode: `{config.feature_mode}`")
    lines.append(
        "- Held-out unit: seed. Thresholds are calibrated only on the other "
        "train-split seeds."
    )
    lines.append(
        f"- Safety criterion: accepted held-out rows need n >= "
        f"{config.min_test_accept}, mean > 0, q25 >= 0."
    )
    lines.append(f"- Feature count: `{len(feature_columns)}`")
    lines.append("")
    lines.append("## Top Configurations")
    top_cols = [
        "method",
        "base_scope",
        "calibration_mode",
        "safe_seed_count",
        "accepted_seed_count",
        "total_accept_n",
        "pooled_mean",
        "pooled_q25",
        "min_seed_mean",
        "min_seed_q25",
    ]
    if len(summary):
        lines.append(markdown_table(summary[top_cols].head(12)))
    else:
        lines.append("No verifier rows were produced.")
    lines.append("")
    if len(summary):
        best = summary.iloc[0]
        lines.append("## Best Held-Out Seed Rows")
        mask = (
            (rows["method"] == best["method"])
            & (rows["base_scope"] == best["base_scope"])
            & (rows["calibration_mode"] == best["calibration_mode"])
        )
        detail_cols = [
            "heldout_seed",
            "status",
            "threshold",
            "calib_accept_n",
            "calib_mean",
            "calib_q25",
            "heldout_base_n",
            "heldout_accept_n",
            "heldout_mean",
            "heldout_q25",
            "heldout_positive_rate",
        ]
        lines.append(markdown_table(rows.loc[mask, detail_cols]))
        lines.append("")
        lines.append("## Decision")
        if int(best["safe_seed_count"]) >= 4 and float(best["pooled_mean"]) > 0.0:
            lines.append(
                "- Train-heldout verifier evidence is promising enough to test "
                "as an online rejection guard in validation/final replay."
            )
        else:
            lines.append(
                "- No train-heldout verifier reaches the 4/5 safe-seed target. "
                "Do not wire this model into final replay as a claimed method "
                "without another representation change."
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small markdown table without pandas' optional tabulate dependency."""
    if len(frame.columns) == 0:
        return ""
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        cells: list[str] = []
        for value in row.tolist():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    require_sklearn()

    df = pd.read_csv(args.input)
    if "effect_margin" not in df.columns:
        raise ValueError("input must contain effect_margin")
    if "seed" not in df.columns:
        raise ValueError("input must contain seed")

    feature_columns = select_feature_columns(df, args.feature_mode)
    if not feature_columns:
        raise ValueError("no usable feature columns selected")
    x_all = numeric_table(df, feature_columns)
    y_all = pd.to_numeric(df["effect_margin"], errors="coerce").to_numpy(dtype=float)
    seeds = sorted(pd.to_numeric(df["seed"], errors="raise").astype(int).unique())

    result_rows: list[dict[str, object]] = []
    row_score_frames: list[pd.DataFrame] = []
    for heldout_seed in seeds:
        test_mask = df["seed"].astype(int).to_numpy() == heldout_seed
        train_mask = ~test_mask
        train_df = df.loc[train_mask].reset_index(drop=True)
        test_df = df.loc[test_mask].reset_index(drop=True)
        x_train = x_all.loc[train_mask].reset_index(drop=True)
        x_test = x_all.loc[test_mask].reset_index(drop=True)
        y_train = y_all[train_mask]
        y_test = y_all[test_mask]

        for method in args.methods:
            train_scores, model = fit_scores(
                method,
                x_train,
                y_train,
                x_train,
                train_df,
                train_df,
                positive_margin=args.positive_margin,
                random_state=args.random_state,
            )
            test_scores, _ = fit_scores(
                method,
                x_train,
                y_train,
                x_test,
                train_df,
                test_df,
                positive_margin=args.positive_margin,
                random_state=args.random_state,
            )
            del model

            for scope in args.base_scopes:
                train_base = base_mask(train_df, scope)
                test_base = base_mask(test_df, scope)
                for calibration_mode in args.calibration_modes:
                    threshold, calib_stats, status = choose_threshold(
                        train_df,
                        train_scores,
                        y_train,
                        train_base,
                        mode=calibration_mode,
                        min_accept=args.min_calib_accept,
                        min_seed_accept=args.min_calib_seed_accept,
                        min_mean=args.min_mean,
                        min_q25=args.min_q25,
                    )
                    if threshold is None:
                        test_accepted = np.zeros(len(test_df), dtype=bool)
                    else:
                        test_accepted = test_base & (test_scores >= threshold)
                    heldout_stats = effect_stats(y_test, test_accepted)
                    values = y_test[test_accepted].astype(float).tolist()
                    result_rows.append(
                        {
                            "method": method,
                            "feature_mode": args.feature_mode,
                            "base_scope": scope,
                            "calibration_mode": calibration_mode,
                            "heldout_seed": int(heldout_seed),
                            "status": status,
                            "threshold": threshold,
                            "calib_accept_n": int(calib_stats["n"]),
                            "calib_mean": calib_stats["mean"],
                            "calib_q25": calib_stats["q25"],
                            "calib_positive_rate": calib_stats["positive_rate"],
                            "heldout_base_n": int(test_base.sum()),
                            "heldout_accept_n": int(heldout_stats["n"]),
                            "heldout_mean": heldout_stats["mean"],
                            "heldout_q25": heldout_stats["q25"],
                            "heldout_positive_rate": heldout_stats["positive_rate"],
                            "heldout_min": heldout_stats["min"],
                            "heldout_effect_values": values,
                        }
                    )
                    row_score_frames.append(
                        pd.DataFrame(
                            {
                                "method": method,
                                "feature_mode": args.feature_mode,
                                "base_scope": scope,
                                "calibration_mode": calibration_mode,
                                "heldout_seed": int(heldout_seed),
                                "row_index": test_df.index.to_numpy(dtype=int),
                                "start": test_df["start"].to_numpy()
                                if "start" in test_df.columns
                                else np.nan,
                                "relative_step": test_df["relative_step"].to_numpy()
                                if "relative_step" in test_df.columns
                                else np.nan,
                                "score": test_scores,
                                "threshold": np.nan if threshold is None else threshold,
                                "accepted": test_accepted,
                                "effect_margin": y_test,
                                "selected_is_anchor": boolish(test_df["selected_is_anchor"]).to_numpy()
                                if "selected_is_anchor" in test_df.columns
                                else False,
                            }
                        )
                    )

    rows = pd.DataFrame(result_rows)
    summary = pooled_summary(rows, min_test_accept=args.min_test_accept)

    args.output.mkdir(parents=True, exist_ok=True)
    rows_for_csv = rows.drop(columns=["heldout_effect_values"])
    rows_for_csv.to_csv(args.output / "effect_verifier_loo_rows.csv", index=False)
    summary.to_csv(args.output / "effect_verifier_loo_summary.csv", index=False)
    if row_score_frames:
        pd.concat(row_score_frames, ignore_index=True).to_csv(
            args.output / "effect_verifier_row_scores.csv", index=False
        )
    config = VerifierConfig(
        input=str(args.input),
        output=str(args.output),
        feature_mode=args.feature_mode,
        methods=list(args.methods),
        base_scopes=list(args.base_scopes),
        calibration_modes=list(args.calibration_modes),
        min_calib_accept=int(args.min_calib_accept),
        min_calib_seed_accept=int(args.min_calib_seed_accept),
        min_test_accept=int(args.min_test_accept),
        min_mean=float(args.min_mean),
        min_q25=float(args.min_q25),
        positive_margin=float(args.positive_margin),
        random_state=int(args.random_state),
    )
    payload = {
        "config": asdict(config),
        "feature_columns": feature_columns,
        "best": summary.iloc[0].to_dict() if len(summary) else None,
    }
    (args.output / "effect_verifier_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_markdown(
        args.output / "effect_verifier_summary.md",
        config,
        summary,
        rows,
        feature_columns,
    )
    if len(summary):
        best = summary.iloc[0]
        print(
            "best",
            best["method"],
            best["base_scope"],
            best["calibration_mode"],
            "safe_seed_count=",
            int(best["safe_seed_count"]),
            "pooled_mean=",
            float(best["pooled_mean"]) if pd.notna(best["pooled_mean"]) else "nan",
            "pooled_q25=",
            float(best["pooled_q25"]) if pd.notna(best["pooled_q25"]) else "nan",
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
