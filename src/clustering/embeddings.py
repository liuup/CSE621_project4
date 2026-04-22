from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from ..utils import resolve_torch_device


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def generate_encoder_embeddings(
    texts: list[str],
    model_name: str,
    pooling: str,
    batch_size: int,
    max_length: int,
    require_cuda: bool = True,
) -> np.ndarray:
    device = resolve_torch_device(require_cuda=require_cuda)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    vectors: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f"Embedding {pooling}", leave=False):
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
            if pooling == "cls":
                pooled = outputs.last_hidden_state[:, 0]
            elif pooling == "mean":
                pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            else:
                raise ValueError(f"Unsupported pooling mode: {pooling}")
        vectors.append(pooled.cpu().numpy())

    return np.concatenate(vectors, axis=0)
