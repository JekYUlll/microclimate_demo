from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def pairwise_wilcoxon(error_by_strategy: Dict[str, np.ndarray]) -> pd.DataFrame:
    """对不同策略的误差序列进行 Wilcoxon 配对检验。"""
    strategies = sorted(error_by_strategy.keys())
    rows = []
    for a, b in combinations(strategies, 2):
        ea = error_by_strategy[a]
        eb = error_by_strategy[b]
        n = min(len(ea), len(eb))
        if n == 0:
            continue
        ea = ea[:n]
        eb = eb[:n]
        try:
            stat, p = wilcoxon(ea, eb)
        except Exception:
            stat, p = np.nan, np.nan
        rows.append({"strategy_a": a, "strategy_b": b, "stat": stat, "p_value": p, "n": n})
    return pd.DataFrame(rows)
