from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml


def _safe_load_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _parse_run_id(run_id: str) -> Dict[str, str]:
    """从 RUN_ID 解析基础信息（回退方案）。"""
    # 形如: dataset_freq_missingness_params_imputer
    parts = run_id.split("_")
    result = {
        "dataset": "unknown",
        "freq": "unknown",
        "missingness": "unknown",
        "missingness_params": "",
        "imputer": "unknown",
    }
    if len(parts) < 4:
        return result

    # freq 通常形如 1s/10s/3h
    freq_idx = None
    for i, p in enumerate(parts):
        if p.endswith("s") or p.endswith("h") or p.endswith("d"):
            freq_idx = i
            break
    if freq_idx is None or freq_idx < 1:
        return result

    result["dataset"] = "_".join(parts[:freq_idx])
    result["freq"] = parts[freq_idx]

    # 余下: missingness + params + imputer
    tail = parts[freq_idx + 1 :]
    if len(tail) >= 2:
        result["imputer"] = tail[-1]
        result["missingness"] = tail[0]
        if len(tail) > 2:
            result["missingness_params"] = "_".join(tail[1:-1])
    return result


def parse_run_metadata(run_dir: Path) -> Dict[str, object]:
    run_id = run_dir.name
    cfg = _safe_load_yaml(run_dir / "config_used.yaml")
    meta = _parse_run_id(run_id)

    # 如果 config_used.yaml 完整，优先使用
    if cfg:
        dataset = cfg.get("dataset", {})
        run = cfg.get("run", {})
        missing = cfg.get("missingness", {})
        impute = cfg.get("imputation", {})
        if dataset.get("name"):
            meta["dataset"] = dataset.get("name")
        if run.get("base_freq"):
            meta["freq"] = str(run.get("base_freq")).lower()
        if missing.get("name"):
            meta["missingness"] = missing.get("name")
        if impute.get("name"):
            meta["imputer"] = impute.get("name")
        # 参数字符串
        params = []
        if meta["missingness"] == "mcar" and "p_missing" in missing:
            params.append(f"p{missing.get('p_missing')}")
        if meta["missingness"] == "block" and "n_blocks" in missing:
            params.append(f"b{missing.get('n_blocks')}")
        if meta["missingness"] == "duty_cycle":
            if "period_steps" in missing:
                params.append(f"p{missing.get('period_steps')}")
            if "on_steps" in missing:
                params.append(f"on{missing.get('on_steps')}")
        if meta["missingness"] == "round_robin":
            if "budget_k" in missing:
                params.append(f"k{missing.get('budget_k')}")
            if "min_on_steps" in missing:
                params.append(f"minon{missing.get('min_on_steps')}")
        if meta["missingness"] == "info_priority":
            if "budget_k" in missing:
                params.append(f"k{missing.get('budget_k')}")
            if "min_on_steps" in missing:
                params.append(f"minon{missing.get('min_on_steps')}")
        meta["missingness_params"] = "_".join(params)

    meta["run_id"] = run_id
    return meta


def load_metrics_long(run_dir: Path) -> pd.DataFrame:
    """读取 metrics_long 或 metrics_overall，并返回长表。"""
    metrics_long_path = run_dir / "tables" / "metrics_long.csv"
    metrics_overall_path = run_dir / "tables" / "metrics_overall.csv"
    if metrics_long_path.exists():
        df = pd.read_csv(metrics_long_path)
        df = df.rename(columns={"metric": "metric", "value": "value"})
        return df
    if metrics_overall_path.exists():
        df = pd.read_csv(metrics_overall_path)
        rows = []
        for _, row in df.iterrows():
            model = row.get("model")
            for col in df.columns:
                if not isinstance(col, str):
                    continue
                if col.startswith("rmse_h"):
                    h = int(col.replace("rmse_h", ""))
                    rows.append({"model": model, "horizon": h, "metric": "rmse", "value": row[col]})
                if col.startswith("mae_h"):
                    h = int(col.replace("mae_h", ""))
                    rows.append({"model": model, "horizon": h, "metric": "mae", "value": row[col]})
                if col.startswith("mape_h"):
                    h = int(col.replace("mape_h", ""))
                    rows.append({"model": model, "horizon": h, "metric": "mape", "value": row[col]})
        return pd.DataFrame(rows)
    return pd.DataFrame()
