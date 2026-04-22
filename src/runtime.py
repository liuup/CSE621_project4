from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from .utils import configure_torch_runtime, ensure_dir, resolve_torch_device, save_json


def initialize_runtime(args: Namespace) -> dict:
    output_dir = ensure_dir(Path(args.output_dir))
    device = resolve_torch_device(require_cuda=args.require_cuda)
    runtime_metadata = configure_torch_runtime(device)
    runtime_metadata["require_cuda"] = bool(args.require_cuda)
    runtime_metadata["prefer_bf16"] = bool(args.prefer_bf16)
    runtime_metadata["output_dir"] = str(output_dir)
    save_json(vars(args), output_dir / "run_config.json")
    save_json(runtime_metadata, output_dir / "runtime.json")
    return runtime_metadata
