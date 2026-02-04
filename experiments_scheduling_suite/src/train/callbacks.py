from __future__ import annotations

from typing import Dict

import json
from pathlib import Path


def save_history(history: Dict[str, list[float]], path: Path) -> None:
    """保存训练日志（loss 曲线）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2))
