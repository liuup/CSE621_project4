from __future__ import annotations

import argparse
from pathlib import Path

from src.classification.classical import run_classical_baseline
from src.classification.decoder_prompt import run_decoder_few_shot, run_decoder_zero_shot
from src.classification.encoder_head import TrainingConfig, run_encoder_frozen_head
from src.classification.encoder_zeroshot import run_encoder_zero_shot
from src.clustering.run_clustering import run_clustering_experiments
from src.config import RuntimeConfig
from src.data import load_bbc_dataset
from src.plots import plot_classification_results, plot_clustering_results
from src.runtime import initialize_runtime
from src.utils import ensure_dir, save_dataframe, set_seed


def parse_args() -> argparse.Namespace:
    defaults = RuntimeConfig()
    parser = argparse.ArgumentParser(description="CSE621 Project 4 experiments")
    parser.add_argument(
        "--task",
        choices=("all", "classification", "clustering"),
        default="all",
        help="Which experiment group to run.",
    )
    parser.add_argument("--limit-train", type=int, default=None, help="Optional cap on train samples.")
    parser.add_argument("--limit-test", type=int, default=None, help="Optional cap on test samples.")
    parser.add_argument("--encoder-model", default=defaults.encoder_model_name)
    parser.add_argument("--decoder-model", default=defaults.decoder_model_name)
    parser.add_argument("--encoder-batch-size", type=int, default=defaults.encoder_batch_size)
    parser.add_argument("--decoder-batch-size", type=int, default=defaults.decoder_batch_size)
    parser.add_argument("--encoder-max-length", type=int, default=defaults.encoder_max_length)
    parser.add_argument("--decoder-max-length", type=int, default=defaults.decoder_max_length)
    parser.add_argument("--decoder-max-new-tokens", type=int, default=defaults.decoder_max_new_tokens)
    parser.add_argument("--train-epochs", type=int, default=defaults.train_epochs)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--validation-size", type=float, default=defaults.validation_size)
    parser.add_argument("--shots-per-class", type=int, default=defaults.shots_per_class)
    parser.add_argument("--tfidf-max-features", type=int, default=defaults.tfidf_max_features)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=defaults.require_cuda)
    parser.add_argument("--prefer-bf16", action=argparse.BooleanOptionalAction, default=defaults.prefer_bf16)
    return parser.parse_args()


def run_classification(args: argparse.Namespace, bundle) -> None:
    classification_dir = ensure_dir(Path(args.output_dir) / "classification")
    results: list[dict] = []

    training_config = TrainingConfig(
        batch_size=args.encoder_batch_size,
        max_length=args.encoder_max_length,
        epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_size=args.validation_size,
    )

    results.append(
        run_encoder_frozen_head(
            bundle,
            classification_dir,
            args.encoder_model,
            training_config,
            require_cuda=args.require_cuda,
        )
    )
    results.append(
        run_encoder_zero_shot(
            bundle,
            classification_dir,
            args.encoder_model,
            args.encoder_batch_size,
            args.encoder_max_length,
            require_cuda=args.require_cuda,
        )
    )
    results.append(
        run_decoder_zero_shot(
            bundle,
            classification_dir,
            args.decoder_model,
            args.decoder_batch_size,
            args.decoder_max_length,
            args.decoder_max_new_tokens,
            require_cuda=args.require_cuda,
            prefer_bf16=args.prefer_bf16,
        )
    )
    results.append(
        run_decoder_few_shot(
            bundle,
            classification_dir,
            args.decoder_model,
            args.decoder_batch_size,
            args.decoder_max_length,
            args.decoder_max_new_tokens,
            args.shots_per_class,
            args.seed,
            require_cuda=args.require_cuda,
            prefer_bf16=args.prefer_bf16,
        )
    )
    results.append(run_classical_baseline(bundle, classification_dir, args.tfidf_max_features))

    df = save_dataframe(results, classification_dir / "classification_summary.csv")
    plot_classification_results(df, classification_dir / "classification_metrics.png")


def run_clustering(args: argparse.Namespace, bundle) -> None:
    clustering_dir = ensure_dir(Path(args.output_dir) / "clustering")
    results = run_clustering_experiments(
        bundle,
        clustering_dir,
        args.encoder_model,
        args.encoder_batch_size,
        args.encoder_max_length,
        args.tfidf_max_features,
        args.require_cuda,
    )
    df = save_dataframe(results, clustering_dir / "clustering_summary.csv")
    plot_clustering_results(df, clustering_dir / "clustering_metrics.png")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    initialize_runtime(args)
    bundle = load_bbc_dataset(limit_train=args.limit_train, limit_test=args.limit_test)

    if args.task in {"all", "classification"}:
        run_classification(args, bundle)
    if args.task in {"all", "clustering"}:
        run_clustering(args, bundle)


if __name__ == "__main__":
    main()
