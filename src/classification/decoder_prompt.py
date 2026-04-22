from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data import DatasetBundle, sample_few_shot_examples
from ..eval import classification_metrics
from ..utils import ensure_dir, get_torch_dtype, resolve_torch_device, timer


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _build_prompt(label_names: list[str], text: str, examples: list[tuple[str, str]] | None = None) -> str:
    label_line = ", ".join(label_names)
    prompt_lines = [
        "You are classifying BBC news articles.",
        f"Choose exactly one label from: {label_line}.",
        "Return only the label name.",
    ]
    if examples:
        prompt_lines.append("Examples:")
        for example_text, example_label in examples:
            prompt_lines.append(f"Article: {example_text}")
            prompt_lines.append(f"Label: {example_label}")
    prompt_lines.append(f"Article: {text}")
    prompt_lines.append("Label:")
    return "\n".join(prompt_lines)


def _extract_prediction(raw_output: str, label_names: list[str]) -> int:
    normalized = _normalize(raw_output)
    for idx, label in enumerate(label_names):
        if _normalize(label) == normalized:
            return idx

    for idx, label in enumerate(label_names):
        if _normalize(label) in normalized:
            return idx

    return 0


def _classify_texts(
    texts: list[str],
    label_names: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    require_cuda: bool = True,
    prefer_bf16: bool = True,
    examples: list[tuple[str, str]] | None = None,
) -> list[int]:
    device = resolve_torch_device(require_cuda=require_cuda)
    torch_dtype = get_torch_dtype(device, prefer_bf16=prefer_bf16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    predictions: list[int] = []
    model.eval()
    for start in tqdm(range(0, len(texts), batch_size), desc="Decoder classification", leave=False):
        batch_texts = texts[start : start + batch_size]
        prompts = [_build_prompt(label_names, text, examples) for text in batch_texts]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        predictions.extend(_extract_prediction(output, label_names) for output in outputs)

    return predictions


def run_decoder_zero_shot(
    bundle: DatasetBundle,
    output_dir: str | Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    require_cuda: bool = True,
    prefer_bf16: bool = True,
) -> dict:
    ensure_dir(output_dir)

    with timer() as timing:
        predictions = _classify_texts(
            bundle.test_texts,
            bundle.label_names,
            model_name,
            batch_size,
            max_length,
            max_new_tokens,
            require_cuda=require_cuda,
            prefer_bf16=prefer_bf16,
            examples=None,
        )

    metrics = classification_metrics(bundle.test_labels, predictions, timing.elapsed)
    result = {"model": "decoder_zero_shot", **metrics}

    pd.DataFrame(
        {
            "text": bundle.test_texts,
            "true_label": [bundle.label_names[int(label)] for label in bundle.test_labels],
            "predicted_label": [bundle.label_names[int(label)] for label in predictions],
        }
    ).to_csv(Path(output_dir) / "decoder_zero_shot_predictions.csv", index=False)

    return result


def run_decoder_few_shot(
    bundle: DatasetBundle,
    output_dir: str | Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    shots_per_class: int,
    seed: int,
    require_cuda: bool = True,
    prefer_bf16: bool = True,
) -> dict:
    ensure_dir(output_dir)
    few_shot_examples = sample_few_shot_examples(
        bundle.train_texts,
        bundle.train_labels,
        bundle.label_names,
        shots_per_class=shots_per_class,
        seed=seed,
    )

    with timer() as timing:
        predictions = _classify_texts(
            bundle.test_texts,
            bundle.label_names,
            model_name,
            batch_size,
            max_length,
            max_new_tokens,
            require_cuda=require_cuda,
            prefer_bf16=prefer_bf16,
            examples=few_shot_examples,
        )

    metrics = classification_metrics(bundle.test_labels, predictions, timing.elapsed)
    result = {"model": "decoder_few_shot", **metrics}

    pd.DataFrame(
        {
            "text": bundle.test_texts,
            "true_label": [bundle.label_names[int(label)] for label in bundle.test_labels],
            "predicted_label": [bundle.label_names[int(label)] for label in predictions],
        }
    ).to_csv(Path(output_dir) / "decoder_few_shot_predictions.csv", index=False)

    return result
