from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .utils import ensure_dir


def plot_classification_results(df: pd.DataFrame, output_path: str | Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1"]
    melted = df.melt(id_vars=["model"], value_vars=metrics, var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(10, 5))
    for metric in metrics:
        metric_df = melted[melted["metric"] == metric]
        ax.plot(metric_df["model"], metric_df["score"], marker="o", label=metric)

    ax.set_title("Classification Metrics by Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()

    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_clustering_results(df: pd.DataFrame, output_path: str | Path) -> None:
    metrics = ["ari", "nmi", "silhouette"]
    melted = df.melt(id_vars=["model"], value_vars=metrics, var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(10, 5))
    for metric in metrics:
        metric_df = melted[melted["metric"] == metric]
        ax.plot(metric_df["model"], metric_df["score"], marker="o", label=metric)

    ax.set_title("Clustering Metrics by Representation")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()

    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_cluster_projection(
    features: np.ndarray,
    true_labels: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    reducer = PCA(n_components=2, random_state=42)
    points = reducer.fit_transform(features)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(points[:, 0], points[:, 1], c=true_labels, s=18, cmap="tab10")
    axes[0].set_title("True Labels")
    axes[1].scatter(points[:, 0], points[:, 1], c=cluster_labels, s=18, cmap="tab10")
    axes[1].set_title("Predicted Clusters")
    fig.suptitle(title)
    fig.tight_layout()

    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=200)
    plt.close(fig)
