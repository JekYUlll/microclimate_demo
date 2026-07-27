#!/usr/bin/env python3
"""Reconstruct Simulink-generated partial data and compare with full reference."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from frozen_reconstructor import FrozenChannelReconstructor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    full = pd.read_csv(OUT / "full_reference.csv")
    partial = pd.read_csv(OUT / "simulink_partial_observations.csv")
    mask = pd.read_csv(OUT / "channel_mask.csv")

    channels = [c for c in full.columns if c != "time"]

    linear = partial.copy()
    linear[channels] = (
        linear[channels]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

    reconstructor = FrozenChannelReconstructor()
    imputed = reconstructor.reconstruct(partial, mask)
    imputed[channels] = (
        imputed[channels]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )

    model_err = imputed[channels].to_numpy() - full[channels].to_numpy()
    linear_err = linear[channels].to_numpy() - full[channels].to_numpy()
    mae = np.nanmean(np.abs(model_err), axis=0)
    rmse = np.sqrt(np.nanmean(model_err**2, axis=0))
    linear_mae = np.nanmean(np.abs(linear_err), axis=0)
    linear_rmse = np.sqrt(np.nanmean(linear_err**2, axis=0))
    observed_rate = mask[channels].mean(axis=0).to_numpy()

    metrics = pd.DataFrame(
        {
            "channel": channels,
            "observed_rate": observed_rate,
            "frozen_model_mae": mae,
            "frozen_model_rmse": rmse,
            "linear_interp_mae": linear_mae,
            "linear_interp_rmse": linear_rmse,
        }
    )
    metrics.to_csv(OUT / "imputation_metrics.csv", index=False)
    imputed.to_csv(OUT / "python_imputed_observations.csv", index=False)

    plot_channels = ["weather", "particle_counter", "laser", "fc4_flux"]
    fig, axes = plt.subplots(len(plot_channels), 1, figsize=(10, 7), sharex=True)
    for ax, ch in zip(axes, plot_channels):
        ax.plot(full["time"], full[ch], color="black", linewidth=1.4, label="full")
        ax.scatter(
            partial["time"],
            partial[ch],
            s=8,
            color="#1f77b4",
            alpha=0.45,
            label="observed partial",
        )
        ax.plot(imputed["time"], imputed[ch], color="#d62728", linewidth=1.0, label="frozen model")
        ax.plot(linear["time"], linear[ch], color="#7f7f7f", linewidth=0.9, alpha=0.65, label="linear")
        ax.set_ylabel(ch)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("time step")
    fig.tight_layout()
    fig.savefig(OUT / "imputation_comparison.png", dpi=180)

    print("Wrote:")
    print(f"  {OUT / 'imputation_metrics.csv'}")
    print(f"  {OUT / 'python_imputed_observations.csv'}")
    print(f"  {OUT / 'imputation_comparison.png'}")
    print(metrics)


if __name__ == "__main__":
    main()
