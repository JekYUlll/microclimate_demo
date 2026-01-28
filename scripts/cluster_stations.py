from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.antaws_loader import load_antaws


DEFAULT_COLS = [
    "temperature_c",
    "pressure_hpa",
    "wind_speed_ms",
    "wind_dir_deg",
    "relative_humidity_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster AntAWS stations and plot grouping figure.")
    parser.add_argument("--root", type=Path, default=Path("data/AntAWS/3_hourly"), help="Root directory of station CSVs.")
    parser.add_argument("--pattern", type=str, default="*_3h.csv", help="Filename pattern.")
    parser.add_argument("--clusters", type=int, default=4, help="Number of clusters.")
    parser.add_argument("--encoding", type=str, default=None, help="Optional encoding override.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/station_clustering"), help="Output directory.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for clustering.")
    return parser.parse_args()


def station_id_from_path(path: Path) -> str:
    name = path.stem
    if name.endswith("_3h"):
        name = name[: -3]
    return name


def summarize_station(path: Path, encoding: str | None) -> Dict[str, float] | None:
    try:
        df, _ = load_antaws(path, encoding=encoding)
    except Exception:
        return None

    row: Dict[str, float] = {
        "rows": float(len(df)),
    }
    for col in DEFAULT_COLS:
        if col not in df.columns:
            row[f"{col}_mean"] = float("nan")
            row[f"{col}_std"] = float("nan")
            row[f"{col}_missing"] = 1.0
            continue
        series = df[col]
        row[f"{col}_mean"] = float(series.mean())
        row[f"{col}_std"] = float(series.std())
        row[f"{col}_missing"] = float(series.isna().mean())
    return row


def main() -> None:
    args = parse_args()
    files = sorted(args.root.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {args.root}/{args.pattern}")

    rows: List[Dict[str, float]] = []
    station_ids: List[str] = []
    skipped: List[str] = []

    for path in files:
        summary = summarize_station(path, args.encoding)
        if summary is None:
            skipped.append(path.name)
            continue
        station_ids.append(station_id_from_path(path))
        rows.append(summary)

    features = pd.DataFrame(rows)
    features.insert(0, "station_id", station_ids)

    # Fill missing values with column means to keep clustering stable.
    numeric_cols = features.select_dtypes(include=["number"]).columns
    features[numeric_cols] = features[numeric_cols].apply(lambda s: s.fillna(s.mean()))

    X = features[numeric_cols].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = max(2, int(args.clusters))
    kmeans = KMeans(n_clusters=k, random_state=args.random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=args.random_state)
    coords = pca.fit_transform(X_scaled)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    features["cluster"] = labels
    features["pca1"] = coords[:, 0]
    features["pca2"] = coords[:, 1]

    features.to_csv(out_dir / "station_clusters.csv", index=False)

    summary = features.groupby("cluster")[numeric_cols].mean().reset_index()
    summary["count"] = features.groupby("cluster").size().values
    summary.to_csv(out_dir / "station_clusters_summary.csv", index=False)

    plt.figure(figsize=(8, 6))
    for cluster_id in sorted(features["cluster"].unique()):
        sub = features[features["cluster"] == cluster_id]
        plt.scatter(sub["pca1"], sub["pca2"], s=22, alpha=0.8, label=f"cluster {cluster_id}")
    plt.title("AntAWS station clustering (PCA)")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "station_clusters.png")
    plt.close()

    if skipped:
        (out_dir / "skipped_files.txt").write_text("\n".join(skipped))

    print(f"Wrote clustering outputs to {out_dir}")


if __name__ == "__main__":
    main()
