from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ..data import DatasetBundle
from ..eval import classification_metrics
from ..utils import ensure_dir, timer


def run_classical_baselines(
    bundle: DatasetBundle,
    output_dir: str | Path,
    max_features: int = 20000,
) -> list[dict]:
    ensure_dir(output_dir)
    candidates = {
        "classical_tfidf_logreg": LogisticRegression(max_iter=2000, random_state=42),
        "classical_tfidf_linear_svm": LinearSVC(random_state=42),
        "classical_tfidf_nb": MultinomialNB(),
    }
    results: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name, estimator in candidates.items():
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=max_features, stop_words="english")),
                ("clf", estimator),
            ]
        )

        with timer() as timing:
            pipeline.fit(bundle.train_texts, bundle.train_labels)
            predictions = pipeline.predict(bundle.test_texts)

        metrics = classification_metrics(bundle.test_labels, predictions, timing.elapsed)
        results.append({"model": model_name, **metrics})
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "text": bundle.test_texts,
                    "true_label": [bundle.label_names[int(label)] for label in bundle.test_labels],
                    "predicted_label": [bundle.label_names[int(label)] for label in predictions],
                }
            )
        )

    pd.concat(prediction_frames, ignore_index=True).to_csv(
        Path(output_dir) / "classical_predictions.csv",
        index=False,
    )

    return results
