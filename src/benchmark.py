"""Dedicated benchmarking: FPS measurement (with/without tiling) and, once
a trained checkpoint exists, ONNX export + fp16 quantization comparisons.

This module is built and runnable, but the full teacher -> student ->
exported/quantized benchmark chain (Phase 4 of the project plan) has not
been executed as of this commit -- it requires a real trained student
checkpoint, which does not exist yet in this environment (no GPU here;
training was handed off). Run `python -m src.benchmark --help` once a
checkpoint is available.
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.config import load_config, resolve_device
from src.evaluate import sharpen_image_inference
from src.models.student import build_student
from src.models.teacher import build_teacher
from src.utils import load_checkpoint


def measure_fps(
    model: nn.Module,
    device,
    num_runs: int = 10,
    num_warmup: int = 5,
    input_size: tuple[int, int] = (512, 512),
    use_tiling: bool = False,
    tiling_patch_size: int = 512,
    tiling_overlap: int = 64,
) -> dict:
    """Runs `model` on a synthetic (black) image `num_warmup` + `num_runs`
    times and reports timing statistics -- mean/median, not a single
    sample, per-run times included for transparency."""
    model.eval()
    model.to(device)

    for _ in range(num_warmup):
        sharpen_image_inference(
            input_image_path=None,
            output_image_path=None,
            model=model,
            device=device,
            model_input_size=input_size,
            use_tiling=use_tiling,
            tiling_patch_size=tiling_patch_size,
            tiling_overlap=tiling_overlap,
        )

    times_ms = []
    for _ in range(num_runs):
        start = time.perf_counter()
        sharpen_image_inference(
            input_image_path=None,
            output_image_path=None,
            model=model,
            device=device,
            model_input_size=input_size,
            use_tiling=use_tiling,
            tiling_patch_size=tiling_patch_size,
            tiling_overlap=tiling_overlap,
        )
        times_ms.append((time.perf_counter() - start) * 1000)

    mean_ms = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)
    return {
        "input_size": input_size,
        "use_tiling": use_tiling,
        "num_runs": num_runs,
        "individual_times_ms": times_ms,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "mean_fps": 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
        "median_fps": 1000.0 / median_ms if median_ms > 0 else float("inf"),
    }


def export_to_onnx(model: nn.Module, output_path: str | Path, input_size: tuple[int, int] = (512, 512), opset: int = 17) -> Path:
    """Exports `model` (expected to return (output, *aux)) to ONNX,
    wrapping it so only the primary image output is exported."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class _OutputOnlyWrapper(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x):
            out, *_ = self.wrapped(x)
            return out

    wrapped = _OutputOnlyWrapper(model).eval()
    dummy = torch.randn(1, 3, input_size[0], input_size[1])
    torch.onnx.export(
        wrapped,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}},
        opset_version=opset,
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Benchmark FPS of the teacher and/or a student checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--which", choices=["teacher", "student"], required=True)
    parser.add_argument("--checkpoint", default=None, help="required if --which student")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--tiling", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, project_root=args.project_root)
    device = resolve_device(config.device)

    if args.which == "teacher":
        model = build_teacher(config, device)
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required when --which student")
        model = build_student(config.model.student).to(device)
        load_checkpoint(model, args.checkpoint, device)

    stats = measure_fps(
        model,
        device,
        num_runs=args.num_runs,
        input_size=tuple(config.data.image_size),
        use_tiling=args.tiling,
        tiling_patch_size=config.inference.tiling_patch_size,
        tiling_overlap=config.inference.tiling_overlap,
    )
    print(f"{args.which}: mean {stats['mean_fps']:.2f} FPS (median {stats['median_fps']:.2f}) over {args.num_runs} runs")


if __name__ == "__main__":
    main()
