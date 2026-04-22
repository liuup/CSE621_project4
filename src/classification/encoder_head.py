from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from ..data import DatasetBundle
from ..eval import classification_metrics
from ..utils import ensure_dir, resolve_torch_device, timer


class TokenizedTextDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


class FrozenEncoderClassifier(nn.Module):
    def __init__(self, encoder_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, **batch):
        labels = batch.pop("labels", None)
        outputs = self.encoder(**batch)
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int
    max_length: int
    epochs: int
    learning_rate: float
    weight_decay: float
    validation_size: float


def _tokenize_texts(tokenizer, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def _evaluate(model, dataloader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    gold: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].numpy()
            gold.append(labels)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            pred = outputs["logits"].argmax(dim=1).cpu().numpy()
            preds.append(pred)
    return np.concatenate(gold), np.concatenate(preds)


def run_encoder_frozen_head(
    bundle: DatasetBundle,
    output_dir: str | Path,
    model_name: str,
    config: TrainingConfig,
    require_cuda: bool = True,
) -> dict:
    ensure_dir(output_dir)
    device = resolve_torch_device(require_cuda=require_cuda)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    indices = np.arange(len(bundle.train_texts))
    min_val_size = len(bundle.label_names)
    requested_val_size = max(int(round(len(indices) * config.validation_size)), 1)
    use_validation = len(indices) - requested_val_size >= min_val_size and requested_val_size >= min_val_size

    if use_validation:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=requested_val_size,
            random_state=42,
            stratify=bundle.train_labels,
        )
    else:
        train_idx = indices
        val_idx = indices

    train_encodings = _tokenize_texts(tokenizer, [bundle.train_texts[i] for i in train_idx], config.max_length)
    val_encodings = _tokenize_texts(tokenizer, [bundle.train_texts[i] for i in val_idx], config.max_length)
    test_encodings = _tokenize_texts(tokenizer, bundle.test_texts, config.max_length)

    train_dataset = TokenizedTextDataset(train_encodings, bundle.train_labels[train_idx])
    val_dataset = TokenizedTextDataset(val_encodings, bundle.train_labels[val_idx])
    test_dataset = TokenizedTextDataset(test_encodings, bundle.test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = FrozenEncoderClassifier(model_name, len(bundle.label_names)).to(device)
    optimizer = AdamW(model.classifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0

    with timer() as timing:
        for _ in range(config.epochs):
            model.train()
            for batch in tqdm(train_loader, desc="Training encoder head", leave=False):
                optimizer.zero_grad()
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                outputs["loss"].backward()
                optimizer.step()

            val_true, val_pred = _evaluate(model, val_loader, device)
            val_metrics = classification_metrics(val_true, val_pred, runtime_sec=0.0)
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        test_true, test_pred = _evaluate(model, test_loader, device)

    metrics = classification_metrics(test_true, test_pred, timing.elapsed)
    result = {"model": "encoder_frozen_head", "best_val_f1": float(best_val_f1), **metrics}

    pd.DataFrame(
        {
            "text": bundle.test_texts,
            "true_label": [bundle.label_names[int(label)] for label in test_true],
            "predicted_label": [bundle.label_names[int(label)] for label in test_pred],
        }
    ).to_csv(Path(output_dir) / "encoder_frozen_head_predictions.csv", index=False)

    return result
