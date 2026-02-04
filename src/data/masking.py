from __future__ import annotations

"""
传感器采样/调度的缺失掩码生成器。

核心概念：
- mask 是一个与原始数据同形状的 0/1 矩阵：1 表示该变量该时刻被观测，0 表示被屏蔽（置为 NaN）。
- 温度（temp）默认始终可观测，其余变量根据策略按预算 k 进行选择。

支持的策略（sampling.strategy）：
1) oracle / full / p0：全部观测（上界）。
2) temp_only / p1：仅观测温度（下界）。
3) round_robin / p2：按固定顺序轮询，每次选 k 个，保持至少 min_on_steps 步。
4) duty_cycle / p3：每个传感器按周期 period_steps 开/关 on_steps，随机相位；
   可选 enforce_budget=True 对同一时刻总开机数进行裁剪。
5) block / block_off / p4：模拟长期故障，按年期望块数与期望缺失长度生成块状缺失。
6) info_priority / p5：基于“信息优先级”权重选 top-k（默认训练集相关系数）。

关键参数（sampling.*）：
- budget_k：每步允许的非温度传感器数量（可用 \"all\"）。
- min_on_steps：一旦开启，至少持续的步数（避免频繁抖动）。
- warmup_steps：新开启后前若干步视为不可用（模拟加热稳定期）。
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class MaskStats:
    variable: str
    observed_ratio: float
    missing_ratio: float
    longest_on: int
    longest_off: int
    mean_on: float
    mean_off: float
    on_count: int
    off_count: int


def _infer_temp_col(columns: Sequence[str], sampling_cfg: Dict) -> str:
    # 优先使用显式指定的 temp_col；否则从 target_cols 或列名中猜测
    temp_col = sampling_cfg.get("temp_col")
    if temp_col and temp_col in columns:
        return temp_col
    # try config-provided target columns
    target_cols = sampling_cfg.get("target_cols") or []
    for col in target_cols:
        if col in columns and "temp" in str(col).lower():
            return col
    # fallback: first column containing temp
    for col in columns:
        if "temp" in str(col).lower():
            return col
    raise ValueError("Unable to infer temperature column; set sampling.temp_col")


def _parse_budget_k(budget_k: object, n_candidates: int) -> int:
    # 将预算 k 解析为整数并裁剪到 [0, n_candidates]
    if budget_k is None:
        return n_candidates
    if isinstance(budget_k, str) and budget_k.lower() == "all":
        return n_candidates
    try:
        k = int(budget_k)
    except (TypeError, ValueError):
        return n_candidates
    return max(0, min(k, n_candidates))


def _run_lengths(mask: np.ndarray, value: bool) -> List[int]:
    # 统计连续区间长度（用于计算最长开/关时长）
    lengths: List[int] = []
    current = 0
    for flag in mask:
        if bool(flag) == value:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
                current = 0
    if current > 0:
        lengths.append(current)
    return lengths


def _apply_warmup(mask: np.ndarray, warmup_steps: int) -> np.ndarray:
    # 对每次“由关到开”的转折，强制前 warmup_steps 步仍为关闭
    if warmup_steps <= 0:
        return mask
    out = mask.copy()
    n_steps, n_cols = out.shape
    for j in range(n_cols):
        col = out[:, j]
        on_transitions = np.where((col[1:] == 1) & (col[:-1] == 0))[0] + 1
        for start in on_transitions:
            end = min(n_steps, start + warmup_steps)
            col[start:end] = 0
        out[:, j] = col
    return out


def _round_robin_mask(
    n_steps: int,
    n_candidates: int,
    k: int,
    min_on_steps: int,
) -> np.ndarray:
    # 轮询策略：按固定顺序滚动选取 k 个传感器，每个选择块持续 min_on_steps
    if n_candidates == 0 or k <= 0:
        return np.zeros((n_steps, n_candidates), dtype=int)
    k = min(k, n_candidates)
    block_len = max(1, min_on_steps)
    blocks = int(np.ceil(n_steps / block_len))
    mask = np.zeros((n_steps, n_candidates), dtype=int)
    pointer = 0
    for b in range(blocks):
        start = b * block_len
        end = min(n_steps, (b + 1) * block_len)
        selected = [(pointer + i) % n_candidates for i in range(k)]
        mask[start:end, selected] = 1
        pointer = (pointer + k) % n_candidates
    return mask


def _duty_cycle_mask(
    n_steps: int,
    n_candidates: int,
    k: int,
    period_steps: int,
    on_steps: int,
    rng: np.random.Generator,
    enforce_budget: bool = True,
    tie_break: str = "random",
) -> np.ndarray:
    # 占空比策略：每个传感器在 period_steps 周期内开启 on_steps
    if n_candidates == 0:
        return np.zeros((n_steps, n_candidates), dtype=int)
    period_steps = max(1, int(period_steps))
    on_steps = max(0, min(int(on_steps), period_steps))
    t = np.arange(n_steps)
    mask = np.zeros((n_steps, n_candidates), dtype=int)
    for j in range(n_candidates):
        phase = int(rng.integers(0, period_steps))
        mask[:, j] = ((t + phase) % period_steps < on_steps).astype(int)
    # 可选的全局预算裁剪：若同一时刻同时开启过多，则按规则剔除
    if enforce_budget and k < n_candidates:
        for i in range(n_steps):
            on_idx = np.flatnonzero(mask[i])
            if len(on_idx) > k:
                if tie_break == "random":
                    drop = rng.choice(on_idx, size=len(on_idx) - k, replace=False)
                else:
                    drop = on_idx[k:]
                mask[i, drop] = 0
    return mask


def _block_outage_mask(
    n_steps: int,
    n_candidates: int,
    expected_gap_steps: int,
    n_blocks_per_year: int,
    steps_per_year: int,
    rng: np.random.Generator,
) -> np.ndarray:
    # 块状缺失：为每个传感器随机生成若干缺失块（长度服从指数分布）
    if n_candidates == 0:
        return np.zeros((n_steps, n_candidates), dtype=int)
    expected_gap_steps = max(1, int(expected_gap_steps))
    steps_per_year = max(1, int(steps_per_year))
    n_years = max(1.0, n_steps / steps_per_year)
    blocks = max(1, int(round(n_blocks_per_year * n_years)))
    mask = np.ones((n_steps, n_candidates), dtype=int)
    for j in range(n_candidates):
        for _ in range(blocks):
            start = int(rng.integers(0, n_steps))
            length = max(1, int(rng.exponential(expected_gap_steps)))
            end = min(n_steps, start + length)
            mask[start:end, j] = 0
    return mask


def _corr_weight(
    x: pd.Series,
    y: pd.Series,
    lag: int,
) -> float:
    # 计算滞后相关性（绝对值），作为信息优先级权重
    if lag > 0:
        x = x.shift(lag)
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 2:
        return 0.0
    corr = np.corrcoef(df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy())[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(abs(corr))


def _corr_weights(
    df: pd.DataFrame,
    temp_col: str,
    candidates: Sequence[str],
    lag_steps: Sequence[int],
) -> Dict[str, float]:
    # 对每个候选变量取多个滞后下的最大相关系数作为权重
    weights: Dict[str, float] = {}
    for col in candidates:
        max_corr = 0.0
        for lag in lag_steps:
            max_corr = max(max_corr, _corr_weight(df[col], df[temp_col], int(lag)))
        weights[col] = max_corr
    return weights


def _info_priority_mask(
    df: pd.DataFrame,
    candidates: Sequence[str],
    temp_col: str,
    k: int,
    min_on_steps: int,
    info_cfg: Dict,
    rng: np.random.Generator,
    train_slice: Optional[slice] = None,
) -> np.ndarray:
    # 信息优先级：按权重选 top-k，并以 min_on_steps 为块切换
    n_steps = len(df)
    if len(candidates) == 0 or k <= 0:
        return np.zeros((n_steps, len(candidates)), dtype=int)

    lag_steps = info_cfg.get("lag_steps", [0, 1, 2, 4])
    weight_source = str(info_cfg.get("weight_source", "train_corr")).lower()
    update_every = info_cfg.get("update_every_steps")

    if train_slice is None:
        base_df = df
    else:
        base_df = df.iloc[train_slice]

    # 默认权重：训练集相关系数（避免信息泄露）
    base_weights = _corr_weights(base_df, temp_col, candidates, lag_steps)

    # 可选事件增强：当温度突变时短期提升风/辐射等传感器权重
    event_threshold_q = info_cfg.get("event_threshold_quantile")
    event_boost = float(info_cfg.get("event_boost", 1.0))
    event_duration = int(info_cfg.get("event_duration_steps", 0))
    if event_threshold_q is not None:
        delta = df[temp_col].diff().abs()
        train_delta = delta.iloc[train_slice] if train_slice is not None else delta
        threshold = float(train_delta.quantile(event_threshold_q)) if not train_delta.dropna().empty else None
    else:
        threshold = None

    boost_cols = info_cfg.get("event_boost_cols")
    if boost_cols:
        boost_cols = [c for c in boost_cols if c in candidates]
    else:
        boost_cols = [c for c in candidates if any(tag in c.lower() for tag in ["wind", "rad", "sw", "solar"])]

    mask = np.zeros((n_steps, len(candidates)), dtype=int)
    block_len = max(1, min_on_steps)
    boost_until = -1

    for block_start in range(0, n_steps, block_len):
        # update weights if requested (only when not strictly train-corr)
        weights = base_weights
        if update_every and weight_source != "train_corr" and block_start > 0:
            if block_start % int(update_every) == 0:
                weights = _corr_weights(df.iloc[:block_start], temp_col, candidates, lag_steps)
        # event boost
        if threshold is not None and block_start < len(df):
            if abs(float(df[temp_col].diff().iloc[block_start])) >= threshold:
                boost_until = block_start + event_duration
        weights_use = dict(weights)
        if block_start < boost_until and boost_cols:
            for col in boost_cols:
                weights_use[col] = weights_use.get(col, 0.0) * event_boost

        # 按权重选择 top-k
        sorted_cols = sorted(weights_use.items(), key=lambda kv: kv[1], reverse=True)
        selected_cols = [c for c, _ in sorted_cols[:k]]
        selected_idx = [candidates.index(c) for c in selected_cols]
        end = min(n_steps, block_start + block_len)
        mask[block_start:end, selected_idx] = 1
    return mask


def generate_mask(df: pd.DataFrame, sampling_cfg: Dict, seed: int | None = None) -> pd.DataFrame:
    # 统一入口：根据策略生成完整 mask（包含温度列）
    if df.empty:
        raise ValueError("Cannot generate mask for empty dataframe")
    rng = np.random.default_rng(seed)
    value_cols = [c for c in df.columns if c != "timestamp"]
    temp_col = _infer_temp_col(value_cols, sampling_cfg)
    temp_always_on = bool(sampling_cfg.get("temp_always_on", True))

    candidate_cols = sampling_cfg.get("candidate_cols")
    if candidate_cols is None:
        candidate_cols = [c for c in value_cols if c != temp_col]
    else:
        candidate_cols = [c for c in candidate_cols if c in value_cols and c != temp_col]

    strategy = str(sampling_cfg.get("strategy", "oracle")).lower()
    budget_k = _parse_budget_k(sampling_cfg.get("budget_k", "all"), len(candidate_cols))
    min_on_steps = int(sampling_cfg.get("min_on_steps", 1))
    warmup_steps = int(sampling_cfg.get("warmup_steps", 0))

    # 初始全 1（全观测），再根据策略覆写
    mask = pd.DataFrame(1, index=df.index, columns=value_cols, dtype=int)

    if strategy in {"oracle", "full", "p0"}:
        pass
    elif strategy in {"temp_only", "temp-only", "p1"}:
        mask.loc[:, value_cols] = 0
        if temp_col in mask.columns:
            mask[temp_col] = 1
    else:
        n_steps = len(df)
        n_candidates = len(candidate_cols)
        if strategy in {"round_robin", "round-robin", "p2"}:
            cand_mask = _round_robin_mask(n_steps, n_candidates, budget_k, min_on_steps)
        elif strategy in {"duty_cycle", "duty-cycle", "p3"}:
            duty_cfg = sampling_cfg.get("duty_cycle", {})
            period_steps = duty_cfg.get("period_steps", 8)
            on_steps = duty_cfg.get("on_steps", 1)
            enforce_budget = bool(duty_cfg.get("enforce_budget", True))
            tie_break = str(duty_cfg.get("tie_break", "random")).lower()
            cand_mask = _duty_cycle_mask(
                n_steps,
                n_candidates,
                budget_k,
                period_steps,
                on_steps,
                rng,
                enforce_budget=enforce_budget,
                tie_break=tie_break,
            )
        elif strategy in {"block", "block_off", "block-off", "p4"}:
            block_cfg = sampling_cfg.get("block", {})
            expected_gap_steps = int(block_cfg.get("expected_gap_steps", 8))
            n_blocks_per_year = int(block_cfg.get("n_blocks_per_year", 10))
            steps_per_year = block_cfg.get("steps_per_year")
            if steps_per_year is None:
                # 如果未指定年步数，则用时间戳推断
                ts = pd.to_datetime(df["timestamp"])
                if len(ts) > 1:
                    diffs = ts.diff().dropna()
                    median_delta = diffs.median()
                    steps_per_year = int(round(pd.Timedelta(days=365) / median_delta))
                else:
                    steps_per_year = 365 * 8
            cand_mask = _block_outage_mask(
                n_steps,
                n_candidates,
                expected_gap_steps,
                n_blocks_per_year,
                steps_per_year,
                rng,
            )
        elif strategy in {"info_priority", "info-priority", "p5"}:
            info_cfg = sampling_cfg.get("info_priority", {})
            train_slice = sampling_cfg.get("_train_slice") or sampling_cfg.get("train_slice")
            cand_mask = _info_priority_mask(
                df,
                candidate_cols,
                temp_col,
                budget_k,
                min_on_steps,
                info_cfg,
                rng,
                train_slice=train_slice,
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        if n_candidates:
            # 加入 warmup 约束并写回到候选列
            cand_mask = _apply_warmup(cand_mask, warmup_steps)
            mask[candidate_cols] = cand_mask

    if temp_always_on and temp_col in mask.columns:
        mask[temp_col] = 1

    return mask


def apply_mask(df: pd.DataFrame, mask_df: pd.DataFrame, sampling_cfg: Dict | None = None) -> pd.DataFrame:
    # 将 mask 应用于数据：mask=0 的位置置为 NaN
    if df.empty:
        return df.copy()
    out = df.copy()
    value_cols = [c for c in df.columns if c != "timestamp"]
    missing_mask = mask_df[value_cols].to_numpy() == 0
    out_values = out[value_cols].to_numpy().copy()
    out_values[missing_mask] = np.nan
    out[value_cols] = out_values
    return out


def mask_stats(mask_df: pd.DataFrame) -> pd.DataFrame:
    # 统计每列观测比例与连续开/关长度分布
    rows: List[Dict[str, object]] = []
    for col in mask_df.columns:
        series = mask_df[col].to_numpy().astype(int)
        on_mask = series == 1
        off_mask = ~on_mask
        on_runs = _run_lengths(series, True)
        off_runs = _run_lengths(series, False)
        rows.append(
            MaskStats(
                variable=col,
                observed_ratio=float(on_mask.mean()) if len(series) else 0.0,
                missing_ratio=float(off_mask.mean()) if len(series) else 0.0,
                longest_on=int(max(on_runs) if on_runs else 0),
                longest_off=int(max(off_runs) if off_runs else 0),
                mean_on=float(np.mean(on_runs) if on_runs else 0.0),
                mean_off=float(np.mean(off_runs) if off_runs else 0.0),
                on_count=int(on_mask.sum()),
                off_count=int(off_mask.sum()),
            ).__dict__
        )
    stats_df = pd.DataFrame(rows)

    # 汇总行：仅针对非温度传感器（启发式）
    temp_like = [c for c in mask_df.columns if "temp" in str(c).lower()]
    non_temp_cols = [c for c in mask_df.columns if c not in temp_like]
    if non_temp_cols:
        total_on = mask_df[non_temp_cols].sum(axis=1)
        rows_summary = {
            "variable": "__summary__",
            "observed_ratio": float(mask_df.to_numpy().mean()),
            "missing_ratio": float(1.0 - mask_df.to_numpy().mean()),
            "longest_on": int(total_on.max()),
            "longest_off": int(total_on.min()),
            "mean_on": float(total_on.mean()),
            "mean_off": float((len(non_temp_cols) - total_on).mean()),
            "on_count": int(total_on.sum()),
            "off_count": int((len(non_temp_cols) - total_on).sum()),
        }
        stats_df = pd.concat([stats_df, pd.DataFrame([rows_summary])], ignore_index=True)

    return stats_df
