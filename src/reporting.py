from __future__ import annotations

from pathlib import Path

import pandas as pd

from .plots import plot_confusion_matrix


def generate_classification_confusion_matrices(classification_dir: str | Path) -> None:
    classification_path = Path(classification_dir)
    output_dir = classification_path / "confusion_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_specs = [
        ("encoder_frozen_head", classification_path / "encoder_frozen_head_predictions.csv", None),
        ("encoder_zero_shot_similarity", classification_path / "encoder_zero_shot_predictions.csv", None),
        ("decoder_zero_shot", classification_path / "decoder_zero_shot_predictions.csv", None),
        ("decoder_few_shot", classification_path / "decoder_few_shot_predictions.csv", None),
        ("classical_tfidf_logreg", classification_path / "classical_predictions.csv", "classical_tfidf_logreg"),
        ("classical_tfidf_linear_svm", classification_path / "classical_predictions.csv", "classical_tfidf_linear_svm"),
        ("classical_tfidf_nb", classification_path / "classical_predictions.csv", "classical_tfidf_nb"),
    ]

    labels: list[str] | None = None
    for model_name, csv_path, filter_model in prediction_specs:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if filter_model is not None:
            df = df[df["model"] == filter_model]
        if df.empty:
            continue
        if labels is None:
            labels = sorted(df["true_label"].astype(str).unique().tolist())
        plot_confusion_matrix(
            y_true=df["true_label"].astype(str).tolist(),
            y_pred=df["predicted_label"].astype(str).tolist(),
            labels=labels,
            output_path=output_dir / f"{model_name}_confusion_matrix.png",
            title=f"Confusion Matrix: {model_name}",
        )
