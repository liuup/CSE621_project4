from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from datasets import DatasetDict, load_dataset


TEXT_CANDIDATES = ("text", "content", "article", "sentence")
LABEL_CANDIDATES = ("label", "labels", "target", "category")
LABEL_NAME_CANDIDATES = ("label_text", "label_name", "category_name")


@dataclass(slots=True)
class DatasetBundle:
    train_texts: list[str]
    train_labels: np.ndarray
    test_texts: list[str]
    test_labels: np.ndarray
    label_names: list[str]
    text_column: str
    label_column: str


def _find_first_present(columns: Iterable[str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find any of {candidates} in dataset columns {list(columns)}")


def _extract_label_names(dataset: DatasetDict, label_column: str) -> list[str]:
    feature = dataset["train"].features[label_column]
    if getattr(feature, "names", None):
        return [str(name) for name in feature.names]

    train_columns = set(dataset["train"].column_names)
    for candidate in LABEL_NAME_CANDIDATES:
        if candidate in train_columns:
            values = dataset["train"][candidate]
            if values:
                pairs = sorted(
                    {
                        int(label): str(name)
                        for label, name in zip(dataset["train"][label_column], values, strict=False)
                    }.items()
                )
                return [name for _, name in pairs]

    unique_labels = sorted(set(int(label) for label in dataset["train"][label_column]))
    return [str(label) for label in unique_labels]


def _subset_if_needed(split, limit: int | None):
    if limit is None or limit >= len(split):
        return split
    return split.select(range(limit))


def load_bbc_dataset(limit_train: int | None = None, limit_test: int | None = None) -> DatasetBundle:
    dataset = load_dataset("SetFit/bbc-news")
    train_columns = dataset["train"].column_names
    text_column = _find_first_present(train_columns, TEXT_CANDIDATES)
    label_column = _find_first_present(train_columns, LABEL_CANDIDATES)
    label_names = _extract_label_names(dataset, label_column)

    train_split = _subset_if_needed(dataset["train"], limit_train)
    test_split = _subset_if_needed(dataset["test"], limit_test)

    return DatasetBundle(
        train_texts=[str(text) for text in train_split[text_column]],
        train_labels=np.asarray(train_split[label_column], dtype=np.int64),
        test_texts=[str(text) for text in test_split[text_column]],
        test_labels=np.asarray(test_split[label_column], dtype=np.int64),
        label_names=label_names,
        text_column=text_column,
        label_column=label_column,
    )


def sample_few_shot_examples(
    texts: list[str],
    labels: np.ndarray,
    label_names: list[str],
    shots_per_class: int = 1,
    seed: int = 42,
) -> list[tuple[str, str]]:
    rng = np.random.default_rng(seed)
    examples: list[tuple[str, str]] = []

    for label_id, label_name in enumerate(label_names):
        indices = np.where(labels == label_id)[0]
        if len(indices) == 0:
            continue
        sample_size = min(shots_per_class, len(indices))
        chosen = rng.choice(indices, size=sample_size, replace=False)
        for idx in chosen:
            examples.append((texts[int(idx)], label_name))

    return examples
