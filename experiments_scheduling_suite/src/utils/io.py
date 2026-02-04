from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """读取 YAML 配置文件。"""
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def save_yaml(data: Dict[str, Any], path: Path) -> None:
    """保存 YAML 配置文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并配置，extra 覆盖 base。"""
    merged = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def ensure_dirs(paths: Dict[str, Any]) -> None:
    """创建必要的目录。"""
    for key, path in paths.items():
        Path(path).mkdir(parents=True, exist_ok=True)


def read_csv_with_timestamp(path: Path) -> pd.DataFrame:
    """读取 CSV 并解析 timestamp 列。"""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("CSV 缺少 timestamp 列")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def build_run_id(dataset_cfg: Dict[str, Any], base_cfg: Dict[str, Any], missing_cfg: Dict[str, Any], impute_cfg: Dict[str, Any]) -> str:
    """根据配置生成 RUN_ID。"""
    dataset_name = dataset_cfg.get("name", "dataset")
    freq = str(base_cfg.get("run", {}).get("base_freq", "1S")).lower().replace("/", "")
    missing_name = missing_cfg.get("name", "missing")
    impute_name = impute_cfg.get("name", "impute")
    extra = ""
    if missing_name == "round_robin":
        extra = f"_k{missing_cfg.get('budget_k', '')}_minon{missing_cfg.get('min_on_steps', '')}"
    if missing_name == "duty_cycle":
        extra = f"_p{missing_cfg.get('period_steps', '')}_on{missing_cfg.get('on_steps', '')}"
    if missing_name == "block":
        extra = f"_b{missing_cfg.get('n_blocks', '')}"
    if missing_name == "mcar":
        extra = f"_p{missing_cfg.get('p_missing', '')}"
    return f"{dataset_name}_{freq}_{missing_name}{extra}_{impute_name}"
