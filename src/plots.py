from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .utils import ensure_dir


def plot_classification_results(df: pd.DataFrame, output_path: str | Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(df))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, metric in enumerate(metrics):
        offsets = x + (idx - 1.5) * width
        ax.bar(offsets, df[metric], width=width, label=metric)

    ax.set_title("Classification Metrics by Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()

    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_runtime_results(
    df: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, df["runtime_sec"], color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    fig.tight_layout()

    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_clustering_results(df: pd.DataFrame, output_path: str | Path) -> None:
    metrics = ["ari", "nmi", "silhouette"]
    x = np.arange(len(df))
    width = 0.22

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, metric in enumerate(metrics):
        offsets = x + (idx - 1) * width
        ax.bar(offsets, df[metric], width=width, label=metric)

    ax.set_title("Clustering Metrics by Representation")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()

    path = Path(output_path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    labels: list[str],
    output_path: str | Path,
    title: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
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
