from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from ..data import DatasetBundle
from ..eval import classification_metrics
from ..utils import ensure_dir, resolve_torch_device, timer


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _encode_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    model.eval()
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.cpu().numpy())
    return np.concatenate(vectors, axis=0)


def run_encoder_zero_shot(
    bundle: DatasetBundle,
    output_dir: str | Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    require_cuda: bool = True,
) -> dict:
    ensure_dir(output_dir)
    device = resolve_torch_device(require_cuda=require_cuda)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    label_prompts = [f"This news article is about {label}." for label in bundle.label_names]

    with timer() as timing:
        label_vectors = _encode_texts(label_prompts, tokenizer, model, device, batch_size, max_length)
        text_vectors = _encode_texts(bundle.test_texts, tokenizer, model, device, batch_size, max_length)
        scores = text_vectors @ label_vectors.T
        predictions = scores.argmax(axis=1)

    metrics = classification_metrics(bundle.test_labels, predictions, timing.elapsed)
    result = {"model": "encoder_zero_shot_similarity", **metrics}

    pd.DataFrame(
        {
            "text": bundle.test_texts,
            "true_label": [bundle.label_names[int(label)] for label in bundle.test_labels],
            "predicted_label": [bundle.label_names[int(label)] for label in predictions],
            "max_similarity": scores.max(axis=1),
        }
    ).to_csv(Path(output_dir) / "encoder_zero_shot_predictions.csv", index=False)

    return result
