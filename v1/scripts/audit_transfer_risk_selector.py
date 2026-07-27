#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from audit_policy_transfer import collect_transfer_rows, markdown_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether validation-row features can reject risky selected deployable policies."
    )
    parser.add_argument("suite_roots", nargs="+", help="One or more claim-suite roots.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--no-deduplicate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = [Path(value) for value in args.suite_roots]
    rows = collect_transfer_rows(roots)
    if not rows:
        raise FileNotFoundError(f"No transfer rows found under {roots}")
    data = pd.DataFrame(rows)
    selected = data.loc[data["is_selected"].astype(bool)].copy()
    if selected.empty:
        raise FileNotFoundError("No selected deployable rows with final metrics were found")
    selected = normalize_selected_rows(selected)
    if not bool(args.no_deduplicate):
        selected = selected.drop_duplicates(
            subset=[
                "seed",
                "policy",
                "validation_margin_mean",
                "validation_margin_median",
                "validation_margin_q25",
                "validation_margin_min",
                "validation_negative_starts",
                "final_margin",
            ]
        ).reset_index(drop=True)
    selected.to_csv(out_dir / "transfer_risk_selected_rows.csv", index=False)

    fixed = evaluate_fixed_rules(selected)
    fixed.to_csv(out_dir / "transfer_risk_fixed_rules.csv", index=False)

    loo = leave_one_seed_out(selected)
    loo.to_csv(out_dir / "transfer_risk_leave_one_seed_out.csv", index=False)

    report = render_report(selected, fixed, loo)
    (out_dir / "transfer_risk_selector_audit.md").write_text(report, encoding="utf-8")
    print(report)


def normalize_selected_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric = [
        "validation_margin_mean",
        "validation_margin_median",
        "validation_margin_q25",
        "validation_margin_min",
        "validation_negative_starts",
        "validation_objective",
        "final_margin",
        "transfer_gap",
        "final_power_mean",
        "final_warmup_abort_count",
    ]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["policy"] = out["policy"].astype(str)
    out["validation_guard_pass"] = out["validation_guard_pass"].astype(bool)
    out["validation_positive_center"] = out["validation_positive_center"].astype(bool)
    out["final_win"] = out["final_margin"] > 0.0
    return out


def evaluate_fixed_rules(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, deploy in fixed_rule_masks(df).items():
        rows.append(score_rule(df, name, deploy))
    return pd.DataFrame(rows).sort_values(
        ["effective_margin_mean", "wins", "n_deployed"],
        ascending=[False, False, False],
    )


def fixed_rule_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    q25_values = sorted({float(x) for x in df["validation_margin_q25"].dropna().to_numpy()})
    mean_values = sorted({float(x) for x in df["validation_margin_mean"].dropna().to_numpy()})
    masks: dict[str, pd.Series] = {
        "always_deploy": pd.Series(True, index=df.index),
        "guard_pass": df["validation_guard_pass"].astype(bool),
        "positive_center": df["validation_positive_center"].astype(bool),
        "event_threshold_only": df["policy"].eq("forecast_aware_event_threshold")
        & df["validation_positive_center"].astype(bool),
        "contextual_duty_only": df["policy"].eq("forecast_aware_contextual_duty")
        & df["validation_positive_center"].astype(bool),
    }
    for max_neg in range(0, 8):
        masks[f"positive_center_neg_le_{max_neg}"] = (
            df["validation_positive_center"].astype(bool)
            & (df["validation_negative_starts"] <= max_neg)
        )
    for threshold in q25_values:
        masks[f"positive_center_q25_ge_{threshold:.6g}"] = (
            df["validation_positive_center"].astype(bool)
            & (df["validation_margin_q25"] >= threshold)
        )
    for threshold in mean_values:
        masks[f"positive_center_mean_ge_{threshold:.6g}"] = (
            df["validation_positive_center"].astype(bool)
            & (df["validation_margin_mean"] >= threshold)
        )
    return masks


def score_rule(df: pd.DataFrame, name: str, deploy: pd.Series) -> dict[str, object]:
    deploy_bool = deploy.reindex(df.index).fillna(False).astype(bool)
    final_win = df["final_win"].astype(bool)
    final_margin = df["final_margin"].astype(float)
    effective_margin = final_margin.where(deploy_bool, 0.0)
    wins = int((deploy_bool & final_win).sum())
    return {
        "rule": name,
        "n": int(len(df)),
        "n_deployed": int(deploy_bool.sum()),
        "wins": wins,
        "win_rate": wins / len(df) if len(df) else np.nan,
        "effective_margin_mean": float(np.nanmean(effective_margin.to_numpy(dtype=float))),
        "effective_margin_median": float(np.nanmedian(effective_margin.to_numpy(dtype=float))),
        "avoided_losses": int((~deploy_bool & ~final_win).sum()),
        "missed_wins": int((~deploy_bool & final_win).sum()),
        "deployed_losses": int((deploy_bool & ~final_win).sum()),
    }


def leave_one_seed_out(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in sorted(df["seed"].dropna().unique()):
        train = df.loc[df["seed"] != seed].copy()
        test = df.loc[df["seed"] == seed].copy()
        if train.empty or test.empty:
            continue
        fixed = evaluate_fixed_rules(train)
        best = fixed.iloc[0]
        masks = fixed_rule_masks(test)
        deploy = masks.get(str(best["rule"]), pd.Series(False, index=test.index))
        score = score_rule(test, str(best["rule"]), deploy)
        rows.append(
            {
                "heldout_seed": int(seed),
                "chosen_rule": str(best["rule"]),
                "train_wins": int(best["wins"]),
                "train_deployed_losses": int(best["deployed_losses"]),
                "train_effective_margin_mean": float(best["effective_margin_mean"]),
                **{f"test_{key}": value for key, value in score.items() if key != "rule"},
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("heldout_seed")


def render_report(selected: pd.DataFrame, fixed: pd.DataFrame, loo: pd.DataFrame) -> str:
    lines: list[str] = ["# Transfer-Risk Selector Audit", ""]
    lines.append("## Data")
    lines.append("")
    lines.append(
        f"Selected rows: `{len(selected)}`; seeds: `{selected['seed'].nunique()}`; "
        f"loss rows: `{int((~selected['final_win'].astype(bool)).sum())}`."
    )
    lines.append("")
    display_cols = [
        "seed",
        "policy",
        "validation_margin_mean",
        "validation_margin_median",
        "validation_margin_q25",
        "validation_negative_starts",
        "validation_guard_pass",
        "final_margin",
        "final_win",
    ]
    lines.append(markdown_table(selected[display_cols].sort_values(["seed", "policy"])))
    lines.extend(["", "## Fixed Rules", ""])
    lines.append(markdown_table(fixed.head(20)))
    lines.extend(["", "## Leave-One-Seed-Out", ""])
    if loo.empty:
        lines.append("No leave-one-seed-out rows were available.")
    else:
        lines.append(markdown_table(loo))
        total = score_loo(loo)
        lines.extend(["", "## LOO Summary", ""])
        lines.append(markdown_table(pd.DataFrame([total])))
    lines.append("")
    return "\n".join(lines)


def score_loo(loo: pd.DataFrame) -> dict[str, object]:
    n = int(loo["test_n"].sum())
    wins = int(loo["test_wins"].sum())
    deployed_losses = int(loo["test_deployed_losses"].sum())
    missed_wins = int(loo["test_missed_wins"].sum())
    avoided_losses = int(loo["test_avoided_losses"].sum())
    mean_margin = float(np.average(loo["test_effective_margin_mean"], weights=loo["test_n"]))
    return {
        "n": n,
        "wins": wins,
        "win_rate": wins / n if n else np.nan,
        "effective_margin_mean": mean_margin,
        "deployed_losses": deployed_losses,
        "avoided_losses": avoided_losses,
        "missed_wins": missed_wins,
    }


if __name__ == "__main__":
    main()
