from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def set_seed(seed: int, deterministic: bool = True) -> None:
    """设置随机种子，保证可复现性。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
