#!/usr/bin/env python3
"""Train replan-level source selectors from window source-oracle labels."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

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

DEFAULT_METHODS = ("logistic_cls", "hist_gbdt_cls", "rf_cls")
DYNAMIC_LABELS = ("raw_bypass", "selected_dynamic")


@dataclass(frozen=True)
class SourceSelectorConfig:
    rows: str
    source_oracle: str
    output: str
    methods: list[str]
    min_calib_accept: int
    min_test_accept: int
    min_mean: float
    min_q25: float
    random_state: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train source selectors using source-oracle window labels and "
            "evaluate accepted dynamic effects by leave-one-seed."
        )
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=Path(
            "v1/artifacts/robust_rollout_multiseed_summary_20260607/"
            "robust_intervention_effect_train_multiseed_rows.csv"
        ),
    )
    parser.add_argument(
        "--source-oracle",
        type=Path,
        default=Path(
            "v1/artifacts/robust_rollout_multiseed_summary_20260607/"
            "effect_window_ceiling_20260607/effect_window_source_oracle.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "v1/artifacts/robust_rollout_multiseed_summary_20260607/"
            "source_selector_loo_20260607"
        ),
    )
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--min-calib-accept", type=int, default=8)
    parser.add_argument("--min-test-accept", type=int, default=3)
    parser.add_argument("--min-mean", type=float, default=0.0)
    parser.add_argument("--min-q25", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=20260607)
    return parser.parse_args()


def require_sklearn() -> None:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: F401
        from sklearn.impute import SimpleImputer  # noqa: F401
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        from sklearn.pipeline import make_pipeline  # noqa: F401
        from sklearn.preprocessing import StandardScaler  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("scikit-learn is required for train_source_selector.py") from exc


def boolish(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0
    return series.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = set(RESULT_COLUMNS) | set(ADMIN_COLUMNS) | {"source_oracle_label", "target_source"}
    columns: list[str] = []
    for column in df.columns:
        if column in exclude:
            continue
        values = df[column]
        if pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values):
            columns.append(column)
            continue
        coerced = pd.to_numeric(values, errors="coerce")
        if coerced.notna().any():
            columns.append(column)
    return columns


def numeric_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    table = pd.DataFrame(index=df.index)
    for column in columns:
        values = df[column]
        if pd.api.types.is_bool_dtype(values):
            table[column] = values.astype(float)
        else:
            table[column] = pd.to_numeric(values, errors="coerce")
    return table.replace([np.inf, -np.inf], np.nan)


def build_model(method: str, random_state: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if method == "logistic_cls":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=4000,
                random_state=random_state,
            ),
        )
    if method == "hist_gbdt_cls":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            l2_regularization=0.01,
            min_samples_leaf=8,
            random_state=random_state,
        )
    if method == "rf_cls":
        return RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"unknown method: {method}")


def attach_source_labels(rows: pd.DataFrame, source_oracle: pd.DataFrame) -> pd.DataFrame:
    source = source_oracle[["seed", "start", "source_oracle_label"]].copy()
    source["seed"] = source["seed"].astype(int)
    source["start"] = source["start"].astype(int)
    merged = rows.copy()
    merged["seed"] = merged["seed"].astype(int)
    merged["start"] = merged["start"].astype(int)
    merged = merged.merge(source, on=["seed", "start"], how="left")
    if merged["source_oracle_label"].isna().any():
        missing = merged.loc[merged["source_oracle_label"].isna(), ["seed", "start"]]
        raise ValueError(f"missing source labels for {len(missing)} rows")
    selected_dynamic = ~boolish(merged["selected_is_anchor"])
    targets = np.full(len(merged), "anchor", dtype=object)
    raw_source = merged["source_oracle_label"].astype(str).to_numpy() == "raw_bypass"
    selected_source = (
        merged["source_oracle_label"].astype(str).to_numpy() == "selected_dynamic"
    )
    targets[raw_source] = "raw_bypass"
    targets[selected_source & selected_dynamic.to_numpy()] = "selected_dynamic"
    merged["target_source"] = targets
    return merged


def effect_stats(effect: np.ndarray, accepted: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(effect, dtype=float)[np.asarray(accepted, dtype=bool)]
    if len(values) == 0:
        return {"n": 0, "mean": None, "q25": None, "positive_rate": None}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "q25": float(np.quantile(values, 0.25)),
        "positive_rate": float(np.mean(values > 0.0)),
    }


def accepted_dynamic(
    predictions: np.ndarray,
    selected_is_anchor: np.ndarray,
) -> np.ndarray:
    pred = np.asarray(predictions, dtype=object)
    selected_anchor = np.asarray(selected_is_anchor, dtype=bool)
    return (pred == "raw_bypass") | ((pred == "selected_dynamic") & (~selected_anchor))


def dynamic_probability(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    labels = [str(label) for label in classes.tolist()]
    values = np.zeros(proba.shape[0], dtype=float)
    for label in DYNAMIC_LABELS:
        if label in labels:
            values += proba[:, labels.index(label)]
    return values


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return np.asarray([], dtype=float)
    return np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 101)))


def choose_threshold(
    effect: np.ndarray,
    predictions: np.ndarray,
    selected_is_anchor: np.ndarray,
    scores: np.ndarray,
    *,
    min_accept: int,
    min_mean: float,
    min_q25: float,
) -> tuple[float | None, dict[str, float | int | None], str]:
    best_threshold: float | None = None
    best_stats = effect_stats(effect, np.zeros(len(effect), dtype=bool))
    best_key: tuple[float, float, float] | None = None
    for threshold in candidate_thresholds(scores):
        gated = np.where(scores >= threshold, predictions, "anchor")
        accepted = accepted_dynamic(gated, selected_is_anchor)
        stats = effect_stats(effect, accepted)
        if int(stats["n"]) < int(min_accept):
            continue
        if stats["mean"] is None or stats["q25"] is None:
            continue
        if float(stats["mean"]) < float(min_mean) or float(stats["q25"]) < float(min_q25):
            continue
        key = (
            float(stats["n"]),
            float(stats["q25"]),
            float(stats["mean"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_stats = stats
    if best_threshold is None:
        return None, best_stats, "no_calibration_threshold"
    return best_threshold, best_stats, "ok"


def summarize(rows: pd.DataFrame, min_test_accept: int) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for method, group in rows.groupby("method", dropna=False):
        accepted = group["heldout_accept_n"] > 0
        safe = (
            (group["heldout_accept_n"] >= int(min_test_accept))
            & (group["heldout_mean"] > 0.0)
            & (group["heldout_q25"] >= 0.0)
        )
        pooled: list[float] = []
        for values in group["heldout_effect_values"]:
            if isinstance(values, list):
                pooled.extend(float(v) for v in values)
        pooled_values = np.asarray(pooled, dtype=float)
        summary_rows.append(
            {
                "method": method,
                "heldout_seed_count": int(len(group)),
                "accepted_seed_count": int(accepted.sum()),
                "safe_seed_count": int(safe.sum()),
                "total_accept_n": int(group["heldout_accept_n"].sum()),
                "pooled_mean": float(np.mean(pooled_values)) if len(pooled_values) else np.nan,
                "pooled_q25": float(np.quantile(pooled_values, 0.25))
                if len(pooled_values)
                else np.nan,
                "min_seed_mean": float(group.loc[accepted, "heldout_mean"].min())
                if accepted.any()
                else np.nan,
                "min_seed_q25": float(group.loc[accepted, "heldout_q25"].min())
                if accepted.any()
                else np.nan,
                "ok_threshold_seeds": int((group["status"] == "ok").sum()),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(
        ["safe_seed_count", "pooled_q25", "pooled_mean", "total_accept_n"],
        ascending=[False, False, False, False],
    )


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(c) for c in frame.columns]
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


def write_markdown(output: Path, summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Source Selector LOO Summary")
    lines.append("")
    lines.append("## Top Configurations")
    lines.append(markdown_table(summary.head(12)))
    lines.append("")
    if len(summary):
        best = str(summary.iloc[0]["method"])
        detail = rows.loc[rows["method"] == best].drop(columns=["heldout_effect_values"])
        lines.append("## Best Held-Out Seeds")
        lines.append(markdown_table(detail))
        lines.append("")
        lines.append("## Decision")
        if int(summary.iloc[0]["safe_seed_count"]) >= 4:
            lines.append(
                "- Source selector train-heldout evidence is promising enough "
                "to test in planner replay."
            )
        else:
            lines.append(
                "- Source selector does not reach the 4/5 safe-seed target on "
                "train-heldout effect labels."
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    require_sklearn()
    raw_rows = pd.read_csv(args.rows)
    source = pd.read_csv(args.source_oracle)
    table = attach_source_labels(raw_rows, source)
    feature_columns = select_feature_columns(table)
    x_all = numeric_table(table, feature_columns)
    y_all = table["target_source"].astype(str).to_numpy()
    effect = pd.to_numeric(table["effect_margin"], errors="coerce").to_numpy(dtype=float)
    selected_anchor = boolish(table["selected_is_anchor"]).to_numpy()
    seeds = sorted(table["seed"].astype(int).unique().tolist())

    result_rows: list[dict[str, object]] = []
    row_scores: list[pd.DataFrame] = []
    for heldout_seed in seeds:
        test_mask = table["seed"].astype(int).to_numpy() == int(heldout_seed)
        train_mask = ~test_mask
        x_train = x_all.loc[train_mask].reset_index(drop=True)
        x_test = x_all.loc[test_mask].reset_index(drop=True)
        y_train = y_all[train_mask]
        train_effect = effect[train_mask]
        test_effect = effect[test_mask]
        train_selected_anchor = selected_anchor[train_mask]
        test_selected_anchor = selected_anchor[test_mask]
        test_rows = table.loc[test_mask].reset_index(drop=True)

        for method in args.methods:
            model = build_model(method, args.random_state)
            if len(np.unique(y_train)) < 2:
                continue
            model.fit(x_train, y_train)
            train_pred = model.predict(x_train)
            test_pred = model.predict(x_test)
            train_proba = model.predict_proba(x_train)
            test_proba = model.predict_proba(x_test)
            classes = np.asarray(model.classes_)
            train_dyn_score = dynamic_probability(train_proba, classes)
            test_dyn_score = dynamic_probability(test_proba, classes)
            threshold, calib_stats, status = choose_threshold(
                train_effect,
                train_pred,
                train_selected_anchor,
                train_dyn_score,
                min_accept=args.min_calib_accept,
                min_mean=args.min_mean,
                min_q25=args.min_q25,
            )
            if threshold is None:
                gated_test = np.full(len(test_pred), "anchor", dtype=object)
            else:
                gated_test = np.where(test_dyn_score >= threshold, test_pred, "anchor")
            test_accept = accepted_dynamic(gated_test, test_selected_anchor)
            heldout_stats = effect_stats(test_effect, test_accept)
            result_rows.append(
                {
                    "method": method,
                    "heldout_seed": int(heldout_seed),
                    "status": status,
                    "threshold": threshold,
                    "calib_accept_n": int(calib_stats["n"]),
                    "calib_mean": calib_stats["mean"],
                    "calib_q25": calib_stats["q25"],
                    "heldout_accept_n": int(heldout_stats["n"]),
                    "heldout_mean": heldout_stats["mean"],
                    "heldout_q25": heldout_stats["q25"],
                    "heldout_positive_rate": heldout_stats["positive_rate"],
                    "pred_raw_bypass": int(np.sum(gated_test == "raw_bypass")),
                    "pred_selected_dynamic": int(np.sum(gated_test == "selected_dynamic")),
                    "pred_anchor": int(np.sum(gated_test == "anchor")),
                    "heldout_effect_values": test_effect[test_accept].astype(float).tolist(),
                }
            )
            row_scores.append(
                pd.DataFrame(
                    {
                        "method": method,
                        "heldout_seed": int(heldout_seed),
                        "seed": test_rows["seed"].astype(int),
                        "start": test_rows["start"].astype(int),
                        "relative_step": test_rows["relative_step"].astype(int),
                        "target_source": y_all[test_mask],
                        "pred_source": gated_test,
                        "dynamic_score": test_dyn_score,
                        "threshold": np.nan if threshold is None else threshold,
                        "accepted": test_accept,
                        "effect_margin": test_effect,
                    }
                )
            )

    rows = pd.DataFrame(result_rows)
    summary = summarize(rows, min_test_accept=args.min_test_accept)
    args.output.mkdir(parents=True, exist_ok=True)
    rows.drop(columns=["heldout_effect_values"]).to_csv(
        args.output / "source_selector_loo_rows.csv",
        index=False,
    )
    summary.to_csv(args.output / "source_selector_loo_summary.csv", index=False)
    if row_scores:
        pd.concat(row_scores, ignore_index=True).to_csv(
            args.output / "source_selector_row_scores.csv",
            index=False,
        )
    payload = {
        "config": asdict(
            SourceSelectorConfig(
                rows=str(args.rows),
                source_oracle=str(args.source_oracle),
                output=str(args.output),
                methods=list(args.methods),
                min_calib_accept=int(args.min_calib_accept),
                min_test_accept=int(args.min_test_accept),
                min_mean=float(args.min_mean),
                min_q25=float(args.min_q25),
                random_state=int(args.random_state),
            )
        ),
        "feature_columns": feature_columns,
        "label_counts": table["target_source"].value_counts().to_dict(),
        "best": summary.iloc[0].to_dict() if len(summary) else None,
    }
    (args.output / "source_selector_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_markdown(args.output / "source_selector_summary.md", summary, rows)
    if len(summary):
        best = summary.iloc[0]
        print(
            "best",
            best["method"],
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
