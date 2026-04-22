from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..data import DatasetBundle
from ..eval import classification_metrics
from ..utils import ensure_dir, timer


def run_classical_baseline(
    bundle: DatasetBundle,
    output_dir: str | Path,
    max_features: int = 20000,
) -> dict:
    ensure_dir(output_dir)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=max_features, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    with timer() as timing:
        pipeline.fit(bundle.train_texts, bundle.train_labels)
        predictions = pipeline.predict(bundle.test_texts)

    metrics = classification_metrics(bundle.test_labels, predictions, timing.elapsed)
    result = {"model": "classical_tfidf_logreg", **metrics}

    pd.DataFrame(
        {
            "text": bundle.test_texts,
            "true_label": [bundle.label_names[int(label)] for label in bundle.test_labels],
            "predicted_label": [bundle.label_names[int(label)] for label in predictions],
        }
    ).to_csv(Path(output_dir) / "classical_predictions.csv", index=False)

    return result
