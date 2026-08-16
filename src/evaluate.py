"""Quantitative evaluation (PSNR/SSIM over a dataset) and single-image
tiled inference. FPS benchmarking lives in src/benchmark.py.

Usage:
    python -m src.evaluate --config configs/default.yaml --checkpoint model_checkpoints/student_kd_ema_final_epoch_050.pth
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.config import load_config, resolve_device
from src.data.dataset import CustomToTensor, build_dataloader
from src.losses import ssim as calculate_ssim_torch
from src.models.student import build_student
from src.utils import calculate_psnr, load_checkpoint


@torch.no_grad()
def evaluate_model(model: nn.Module, data_loader, device, save_output_images: bool = False, output_dir: str | Path | None = None):
    """Average SSIM/PSNR of `model` over every pair in `data_loader`."""
    model.eval()
    total_ssim, total_psnr, n = 0.0, 0.0, 0

    if len(data_loader) == 0:
        return 0.0, 0.0

    for blurry, sharp, paths in tqdm(data_loader, desc="Evaluating"):
        blurry, sharp = blurry.to(device), sharp.to(device)
        output, *_ = model(blurry)
        output = output.clamp(0, 1)

        for i in range(output.shape[0]):
            total_ssim += float(calculate_ssim_torch(output[i:i+1], sharp[i:i+1]))
            total_psnr += calculate_psnr(output[i], sharp[i], data_range=1.0)
            n += 1

            if save_output_images and output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                name = Path(paths[i]).name
                stem, ext = name.rsplit(".", 1) if "." in name else (name, "png")
                out_path = output_dir / f"{stem}_sharpened.{ext}"
                transforms.ToPILImage()(output[i].cpu()).save(out_path)

    return (total_ssim / n, total_psnr / n) if n else (0.0, 0.0)


@torch.no_grad()
def sharpen_image_inference(
    input_image_path: str | None,
    output_image_path: str | Path | None,
    model: nn.Module,
    device,
    model_input_size: tuple[int, int] = (512, 512),
    target_resolution: tuple[int, int] | None = None,
    use_tiling: bool = False,
    tiling_patch_size: int = 512,
    tiling_overlap: int = 64,
) -> Image.Image:
    """Sharpen one image, optionally tiling for resolutions above the
    model's native training size."""
    model.eval()

    if input_image_path is None:
        input_image = Image.new("RGB", (tiling_patch_size, tiling_patch_size), color="black")
    else:
        input_image = Image.open(input_image_path).convert("RGB")

    original_width, original_height = input_image.size
    to_tensor = CustomToTensor()

    if use_tiling:
        stride = tiling_patch_size - tiling_overlap
        num_h = math.ceil((original_height - tiling_overlap) / stride)
        num_w = math.ceil((original_width - tiling_overlap) / stride)

        accumulator = torch.zeros(3, original_height, original_width, device=device)
        weights = torch.zeros(3, original_height, original_width, device=device)
        preprocess = transforms.Compose(
            [transforms.Resize((tiling_patch_size, tiling_patch_size), transforms.InterpolationMode.BICUBIC)]
        )

        for i in range(num_h):
            for j in range(num_w):
                top, left = i * stride, j * stride
                bottom = min(top + tiling_patch_size, original_height)
                right = min(left + tiling_patch_size, original_width)
                if bottom == original_height:
                    top = max(0, original_height - tiling_patch_size)
                if right == original_width:
                    left = max(0, original_width - tiling_patch_size)

                patch = input_image.crop((left, top, right, bottom))
                patch_tensor = to_tensor(preprocess(patch)).unsqueeze(0).to(device)
                sharpened_patch, *_ = model(patch_tensor)
                sharpened_patch = sharpened_patch.squeeze(0).clamp(0, 1)

                accumulator[:, top:bottom, left:right] += sharpened_patch
                weights[:, top:bottom, left:right] += 1.0

        weights[weights == 0] = 1e-12
        final_tensor = accumulator / weights
        sharpened_image = transforms.ToPILImage()(final_tensor.cpu())
    else:
        preprocess = transforms.Resize(model_input_size, transforms.InterpolationMode.BICUBIC)
        input_tensor = to_tensor(preprocess(input_image)).unsqueeze(0).to(device)
        output, *_ = model(input_tensor)
        sharpened_image = transforms.ToPILImage()(output.squeeze(0).cpu().clamp(0, 1))

    if target_resolution and sharpened_image.size != tuple(target_resolution):
        sharpened_image = sharpened_image.resize(tuple(target_resolution), Image.LANCZOS)
    elif input_image_path is not None and sharpened_image.size != (original_width, original_height):
        sharpened_image = sharpened_image.resize((original_width, original_height), Image.LANCZOS)

    if output_image_path is not None:
        output_image_path = Path(output_image_path)
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        sharpened_image.save(output_image_path)

    return sharpened_image


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained student checkpoint on the test set")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, project_root=args.project_root)
    device = resolve_device(config.device)

    model = build_student(config.model.student).to(device)
    load_checkpoint(model, args.checkpoint, device)

    test_loader = build_dataloader(config, "test", shuffle=False)
    output_dir = config.resolve(config.paths.inference_output_dir) if args.save_images else None
    avg_ssim, avg_psnr = evaluate_model(model, test_loader, device, save_output_images=args.save_images, output_dir=output_dir)
    print(f"Test set -- SSIM: {avg_ssim:.4f}  PSNR: {avg_psnr:.4f} dB  (n={len(test_loader.dataset)})")


if __name__ == "__main__":
    main()
