from __future__ import annotations

from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data import DatasetBundle
from ..eval import clustering_metrics
from ..plots import plot_cluster_projection
from ..utils import ensure_dir, timer
from .embeddings import generate_encoder_embeddings


def _run_kmeans(features, n_clusters: int):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(features)
    return model, cluster_labels


def run_clustering_experiments(
    bundle: DatasetBundle,
    output_dir: str | Path,
    encoder_model_name: str,
    encoder_batch_size: int,
    encoder_max_length: int,
    tfidf_max_features: int,
    require_cuda: bool = True,
) -> list[dict]:
    ensure_dir(output_dir)
    results: list[dict] = []
    n_clusters = len(bundle.label_names)

    for pooling in ("cls", "mean"):
        with timer() as timing:
            features = generate_encoder_embeddings(
                bundle.test_texts,
                encoder_model_name,
                pooling,
                encoder_batch_size,
                encoder_max_length,
                require_cuda=require_cuda,
            )
            _, cluster_labels = _run_kmeans(features, n_clusters)

        metrics = clustering_metrics(bundle.test_labels, cluster_labels, features, timing.elapsed)
        results.append({"model": f"encoder_{pooling}_kmeans", **metrics})
        plot_cluster_projection(
            features,
            bundle.test_labels,
            cluster_labels,
            Path(output_dir) / f"cluster_projection_{pooling}.png",
            f"Encoder {pooling.upper()} embeddings",
        )

    with timer() as timing:
        vectorizer = TfidfVectorizer(max_features=tfidf_max_features, stop_words="english")
        features = vectorizer.fit_transform(bundle.test_texts).toarray()
        _, cluster_labels = _run_kmeans(features, n_clusters)

    metrics = clustering_metrics(bundle.test_labels, cluster_labels, features, timing.elapsed)
    results.append({"model": "classical_tfidf_kmeans", **metrics})
    plot_cluster_projection(
        features,
        bundle.test_labels,
        cluster_labels,
        Path(output_dir) / "cluster_projection_tfidf.png",
        "Classical TF-IDF representation",
    )

    return results
