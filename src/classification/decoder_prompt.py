from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data import DatasetBundle, sample_few_shot_examples
from ..eval import classification_metrics
from ..utils import ensure_dir, get_torch_dtype, resolve_torch_device, timer


def _label_descriptions(label_names: list[str]) -> list[str]:
    descriptions = {
        "tech": "tech - technology, internet, software, gadgets, telecom",
        "business": "business - companies, markets, economy, finance, deals",
        "sport": "sport - sports matches, teams, athletes, tournaments",
        "entertainment": "entertainment - film, music, tv, celebrities, arts",
        "politics": "politics - government, elections, policy, parliament, leaders",
    }
    return [descriptions.get(label, label) for label in label_names]


def _trim_text(text: str, char_limit: int) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= char_limit:
        return cleaned
    return cleaned[: char_limit - 3].rstrip() + "..."


def _build_prompt(label_names: list[str], text: str, examples: list[tuple[str, str]] | None = None) -> str:
    label_line = ", ".join(label_names)
    description_block = "\n".join(_label_descriptions(label_names))
    article_limit = 900 if not examples else 420

    prompt_lines = [
        "You are classifying BBC news articles.",
        f"Choose exactly one label from: {label_line}.",
        "Label meanings:",
        description_block,
        "Respond with the best label.",
    ]
    if examples:
        prompt_lines.append("Examples:")
        for example_text, example_label in examples:
            prompt_lines.append(f"Article: {_trim_text(example_text, 140)}")
            prompt_lines.append(f"Label: {example_label}")
    prompt_lines.append(f"Article: {_trim_text(text, article_limit)}")
    prompt_lines.append("Label:")
    return "\n".join(prompt_lines)


def _format_prompt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": "You are a careful text classification assistant."},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _score_candidate_batch(
    model,
    tokenizer,
    prompts: list[str],
    label_names: list[str],
    device: torch.device,
    max_length: int,
) -> tuple[list[int], list[str]]:
    label_texts = [f" {label}" for label in label_names]
    label_token_lengths = [
        len(tokenizer(label_text, add_special_tokens=False)["input_ids"]) for label_text in label_texts
    ]

    full_texts: list[str] = []
    repeated_label_lengths: list[int] = []
    for prompt in prompts:
        for label_text, label_length in zip(label_texts, label_token_lengths, strict=False):
            full_texts.append(prompt + label_text)
            repeated_label_lengths.append(label_length)

    encoded = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()

    batch_size, padded_length = input_ids.shape
    for row_idx in range(batch_size):
        full_length = int(attention_mask[row_idx].sum().item())
        label_length = repeated_label_lengths[row_idx]
        label_start = padded_length - label_length
        non_padding_start = padded_length - full_length
        label_start = max(label_start, non_padding_start)
        labels[row_idx, :label_start] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size())
    valid = shift_labels.ne(-100)
    sequence_loss = token_losses.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    score_matrix = (-sequence_loss).view(len(prompts), len(label_names))

    predictions = score_matrix.argmax(dim=1).tolist()
    diagnostics = [
        json.dumps(
            {label: float(score_matrix[row_idx, label_idx].item()) for label_idx, label in enumerate(label_names)},
            ensure_ascii=True,
        )
        for row_idx in range(score_matrix.size(0))
    ]
    return predictions, diagnostics


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
) -> tuple[list[int], list[str]]:
    del max_new_tokens
    device = resolve_torch_device(require_cuda=require_cuda)
    torch_dtype = get_torch_dtype(device, prefer_bf16=prefer_bf16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    predictions: list[int] = []
    diagnostics: list[str] = []
    model.eval()
    for start in tqdm(range(0, len(texts), batch_size), desc="Decoder classification", leave=False):
        batch_texts = texts[start : start + batch_size]
        prompts = [_format_prompt(tokenizer, _build_prompt(label_names, text, examples)) for text in batch_texts]
        batch_predictions, batch_diagnostics = _score_candidate_batch(
            model,
            tokenizer,
            prompts,
            label_names,
            device,
            max_length,
        )
        predictions.extend(batch_predictions)
        diagnostics.extend(batch_diagnostics)

    return predictions, diagnostics


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
        predictions, diagnostics = _classify_texts(
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
            "raw_output": diagnostics,
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
        predictions, diagnostics = _classify_texts(
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
            "raw_output": diagnostics,
        }
    ).to_csv(Path(output_dir) / "decoder_few_shot_predictions.csv", index=False)

    return result
