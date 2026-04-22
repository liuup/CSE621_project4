from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_torch_device(require_cuda: bool = True):
    import torch

    has_cuda = torch.cuda.is_available()
    if require_cuda and not has_cuda:
        raise RuntimeError(
            "CUDA is required by AGENTS.md but is not available. "
            "Activate the project environment and verify the NVIDIA driver and torch CUDA install."
        )
    return torch.device("cuda" if has_cuda else "cpu")


def get_torch_dtype(device, prefer_bf16: bool = True):
    import torch

    if device.type != "cuda":
        return torch.float32
    if prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def configure_torch_runtime(device) -> dict:
    import torch

    metadata = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        metadata["device_name"] = torch.cuda.get_device_name(device)
        metadata["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        metadata["tf32_enabled"] = True

    return metadata


@dataclass(slots=True)
class TimerResult:
    elapsed: float = 0.0


@contextmanager
def timer():
    result = TimerResult()
    start = time.perf_counter()
    try:
        yield result
    finally:
        end = time.perf_counter()
        result.elapsed = end - start


def save_json(data: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def save_dataframe(records: list[dict], output_path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(records)
    path = Path(output_path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return df
