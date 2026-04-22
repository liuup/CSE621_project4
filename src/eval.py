from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    silhouette_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    runtime_sec: float,
    average: str = "weighted",
) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "runtime_sec": float(runtime_sec),
    }


def clustering_metrics(
    y_true: np.ndarray,
    cluster_labels: np.ndarray,
    features: np.ndarray,
    runtime_sec: float,
) -> dict:
    unique_clusters = np.unique(cluster_labels)
    if len(unique_clusters) < 2 or len(unique_clusters) >= len(cluster_labels):
        silhouette = math.nan
    else:
        silhouette = float(silhouette_score(features, cluster_labels))

    return {
        "ari": float(adjusted_rand_score(y_true, cluster_labels)),
        "nmi": float(normalized_mutual_info_score(y_true, cluster_labels)),
        "silhouette": silhouette,
        "runtime_sec": float(runtime_sec),
    }
