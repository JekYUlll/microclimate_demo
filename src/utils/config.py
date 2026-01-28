from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def load_config(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    cfg: Dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return resolve_config(cfg)


def resolve_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    root = repo_root()
    exp_id = cfg.get("exp_id")
    if not exp_id:
        exp_id = datetime.utcnow().strftime("exp_%Y%m%d_%H%M%S")
        cfg["exp_id"] = exp_id

    paths = cfg.setdefault("paths", {})
    paths["raw_root"] = Path(paths.get("raw_root", root / "data" / "AntAWS" / "3_hourly"))
    paths["processed_root"] = Path(paths.get("processed_root", root / "data" / "processed"))
    paths["reports_root"] = Path(paths.get("reports_root", root / "reports"))
    paths["models_root"] = Path(paths.get("models_root", root / "models"))

    cfg["processed_dir"] = paths["processed_root"] / exp_id
    cfg["reports_dir"] = paths["reports_root"] / exp_id
    cfg["models_dir"] = paths["models_root"] / exp_id

    cfg.setdefault("freq", "3H")
    cfg.setdefault("resample_freq", None)
    cfg.setdefault("horizons", [1, 2, 4, 8])
    cfg.setdefault("window_size", 24)
    cfg.setdefault("stride", 1)

    columns = cfg.setdefault("columns", {})
    columns["targets"] = _as_list(columns.get("targets", []))

    features = cfg.setdefault("features", {})
    features.setdefault("rolling_windows", [2, 8])
    features.setdefault("diff_steps", [1, 2])
    features.setdefault("include_time_features", True)
    features.setdefault("include_missing_features", True)

    split = cfg.setdefault("split", {})
    split.setdefault("mode", "ratio")
    split.setdefault("train", 0.7)
    split.setdefault("val", 0.1)
    split.setdefault("test", 0.2)

    impute = cfg.setdefault("impute", {})
    impute.setdefault("strategy", "A")
    impute.setdefault("max_gap_steps", 4)
    impute.setdefault("strategy_b", "stl")

    models = cfg.setdefault("models", {})
    models.setdefault("tft", {})
    models.setdefault("tcn", {})
    models.setdefault("tabular", {})

    loss = cfg.setdefault("loss", {})
    loss.setdefault("lambda_phys", 0.0)
    loss.setdefault("use_bounds_term", True)
    loss.setdefault("use_coherence_term", True)

    cfg.setdefault("seed", 42)
    return cfg


def station_file(cfg: Dict[str, Any], station_id: str, freq: str | None = None) -> Path:
    freq = freq or cfg.get("freq", "3H")
    freq_tag = str(freq).lower()
    filename = f"{station_id}_{freq_tag}.csv"
    return Path(cfg["paths"]["raw_root"]) / filename


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    Path(cfg["processed_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["reports_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["models_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["reports_dir"]) / "tables").mkdir(parents=True, exist_ok=True)
    (Path(cfg["reports_dir"]) / "figures").mkdir(parents=True, exist_ok=True)
    (Path(cfg["reports_dir"]) / "preds").mkdir(parents=True, exist_ok=True)


def dump_config_snapshot(cfg: Dict[str, Any], path: Path) -> None:
    serializable: Dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, Path):
            serializable[k] = str(v)
        elif isinstance(v, dict):
            serializable[k] = {kk: str(vv) if isinstance(vv, Path) else vv for kk, vv in v.items()}
        else:
            serializable[k] = v
    path.write_text(yaml.safe_dump(serializable, sort_keys=False))
