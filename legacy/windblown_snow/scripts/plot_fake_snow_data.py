from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot synthetic wind-blown snow dataset diagnostics.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("legacy/windblown_snow/data/synthetic/windblown_snow_sample.csv"),
        help="Path to the synthetic CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("legacy/windblown_snow/plots"),
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Max rows to plot for scatter plots.",
    )
    return parser


def _maybe_downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.iloc[np.linspace(0, len(df) - 1, max_points).astype(int)]


def _plot_time_series(df: pd.DataFrame, outdir: Path) -> None:
    cols = [
        "air_temperature_c",
        "snow_surface_temperature_c",
        "wind_speed_ms",
        "solar_radiation_wm2",
        "relative_humidity",
        "snow_mass_flux_kg_m2_s",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for ax, col in zip(axes, cols):
        if col not in df.columns:
            ax.axis("off")
            continue
        ax.plot(df["timestamp"], df[col], linewidth=1.0)
        ax.set_title(col)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "timeseries_overview.png", dpi=160)
    plt.close(fig)


def _plot_scatter(df: pd.DataFrame, outdir: Path) -> None:
    sub = _maybe_downsample(df, 2000)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = sub.get("wind_speed_ms")
    y = sub.get("snow_mass_flux_kg_m2_s")
    if x is not None and y is not None:
        ax.scatter(x, y, s=8, alpha=0.6)
        ax.set_xlabel("wind_speed_ms")
        ax.set_ylabel("snow_mass_flux_kg_m2_s")
        ax.set_yscale("log")
        ax.set_title("Wind vs Snow Mass Flux")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "scatter_wind_vs_mass_flux.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = sub.get("air_temperature_c")
    y = sub.get("snow_surface_temperature_c")
    if x is not None and y is not None:
        ax.scatter(x, y, s=8, alpha=0.6)
        ax.set_xlabel("air_temperature_c")
        ax.set_ylabel("snow_surface_temperature_c")
        ax.set_title("Air vs Snow Surface Temperature")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "scatter_air_vs_snow_temp.png", dpi=160)
    plt.close(fig)


def _plot_histograms(df: pd.DataFrame, outdir: Path) -> None:
    cols = [
        "wind_speed_ms",
        "air_temperature_c",
        "relative_humidity",
        "precip_rate_mm_h",
        "snow_mass_flux_kg_m2_s",
        "ri_bulk",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, col in zip(axes, cols):
        if col not in df.columns:
            ax.axis("off")
            continue
        ax.hist(df[col].dropna(), bins=40, alpha=0.8)
        ax.set_title(col)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "histograms.png", dpi=160)
    plt.close(fig)


def _plot_stability(df: pd.DataFrame, outdir: Path) -> None:
    if "stability_flag" not in df.columns:
        return
    counts = df["stability_flag"].value_counts()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(counts.index.tolist(), counts.values) # type: ignore
    ax.set_title("Stability Flag Counts")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "stability_flag_counts.png", dpi=160)
    plt.close(fig)


def _plot_spectra_consistency(df: pd.DataFrame, outdir: Path) -> None:
    size_mass_cols: List[str] = [c for c in df.columns if c.startswith("size_bin_") and c.endswith("_mass_flux_kg_m2_s")]
    vel_mass_cols: List[str] = [c for c in df.columns if c.startswith("velocity_bin_") and c.endswith("_mass_flux_kg_m2_s")]
    total_col = "snow_mass_flux_kg_m2_s"
    if total_col not in df.columns:
        return

    if size_mass_cols:
        size_sum = df[size_mass_cols].sum(axis=1)
        residual = size_sum - df[total_col]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["timestamp"], residual, linewidth=1.0)
        ax.set_title("Size-bin mass flux residual (sum - total)")
        ax.set_ylabel("residual")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "size_bin_mass_residual.png", dpi=160)
        plt.close(fig)

    if vel_mass_cols:
        vel_sum = df[vel_mass_cols].sum(axis=1)
        residual = vel_sum - df[total_col]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["timestamp"], residual, linewidth=1.0)
        ax.set_title("Velocity-bin mass flux residual (sum - total)")
        ax.set_ylabel("residual")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "velocity_bin_mass_residual.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if not args.input.exists():
        raise SystemExit(f"Missing input CSV: {args.input}")

    df = pd.read_csv(args.input)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

    args.outdir.mkdir(parents=True, exist_ok=True)

    _plot_time_series(df, args.outdir)
    _plot_scatter(df, args.outdir)
    _plot_histograms(df, args.outdir)
    _plot_stability(df, args.outdir)
    _plot_spectra_consistency(df, args.outdir)

    print(f"Plots saved to {args.outdir}")


if __name__ == "__main__":
    main()
