"""Training entry point: knowledge distillation of the U-Net+SE student
from the frozen Restormer teacher.

Usage:
    python -m src.train --config configs/default.yaml

Importing this module does nothing by itself -- all work happens inside
main(), guarded by `if __name__ == "__main__"`.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from src.config import Config, load_config, resolve_device
from src.data.dataset import build_dataloader
from src.evaluate import evaluate_model
from src.losses import build_loss
from src.models.student import build_student
from src.models.teacher import build_teacher
from src.utils import save_checkpoint, set_seed, update_ema


class MetricsLogger:
    """Appends one row per epoch to both a CSV and a JSON-lines file under
    config.paths.logs_dir, so training curves survive the process exiting
    (unlike the original script, which only printed to stdout).

    Fieldnames are a FIXED superset, known up front (not inferred from the
    first logged row) -- eval-only fields (val_ssim, val_psnr) only appear
    on eval-interval epochs, but every row still gets every column, with
    blanks where a value doesn't apply. This is what the original bug got
    wrong: the header locked in from epoch 1's train-only row, then crashed
    the moment an eval-epoch row tried to add columns that weren't in the
    header."""

    FIELDNAMES = [
        "epoch", "lr",
        "train_recon_l1", "train_perceptual", "train_feature_distillation",
        "train_kl_div", "train_ssim", "train_combined",
        "val_ssim", "val_psnr",
    ]

    def __init__(self, logs_dir: Path, run_name: str):
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = logs_dir / f"{run_name}_metrics.csv"
        self.jsonl_path = logs_dir / f"{run_name}_metrics.jsonl"

    def log(self, row: dict) -> None:
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        write_header = not self.csv_path.exists()
        full_row = {k: row.get(k, "") for k in self.FIELDNAMES}
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(full_row)


def train(config: Config, run_name: str = "student_kd") -> None:
    set_seed(config.training.seed)
    device = resolve_device(config.device)
    print(f"Device: {device}")

    train_loader = build_dataloader(config, "train")
    test_loader = build_dataloader(config, "test", shuffle=False)
    print(f"Train pairs: {len(train_loader.dataset)} | Test pairs: {len(test_loader.dataset)}")

    teacher = build_teacher(config, device)
    teacher.eval()

    student = build_student(config.model.student).to(device)
    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student trainable parameters: {n_params:,}")

    loss_fn = build_loss(config.losses, device)

    optimizer = optim.Adam(student.parameters(), lr=config.training.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.training.scheduler.t_0,
        T_mult=config.training.scheduler.t_mult,
        eta_min=config.training.scheduler.eta_min,
    )
    ema_model = build_student(config.model.student).to(device)
    ema_model.load_state_dict(student.state_dict())
    ema_model.eval()

    logger = MetricsLogger(config.resolve(config.paths.logs_dir), run_name)
    checkpoints_dir = config.resolve(config.paths.checkpoints_dir)
    grad_accum = config.training.gradient_accumulation_steps

    student.train()
    for epoch in range(config.training.num_epochs):
        epoch_totals = {"recon_l1": 0.0, "perceptual": 0.0, "feature_distillation": 0.0, "kl_div": 0.0, "ssim": 0.0, "combined": 0.0}
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{config.training.num_epochs}")

        for batch_idx, (blurry, sharp, _) in pbar:
            blurry, sharp = blurry.to(device), sharp.to(device)

            with torch.no_grad():
                teacher_out, teacher_e2, teacher_e4, teacher_b = teacher(blurry)

            student_out, student_e2, student_e4, student_b = student(blurry)

            def _downsample(feat, target):
                if feat is None:
                    return None
                return torch.nn.functional.interpolate(feat.detach(), size=target.shape[2:], mode="bilinear", align_corners=False)

            teacher_feats = (
                _downsample(teacher_e2, student_e2),
                _downsample(teacher_e4, student_e4),
                _downsample(teacher_b, student_b),
            )
            student_feats = (student_e2, student_e4, student_b)

            combined_loss, components = loss_fn(
                student_output=student_out,
                sharp_images=sharp,
                teacher_output=teacher_out,
                student_feats=student_feats,
                teacher_feats=teacher_feats,
            )

            (combined_loss / grad_accum).backward()

            if (batch_idx + 1) % grad_accum == 0:
                optimizer.step()
                update_ema(ema_model, student, config.training.ema_decay)
                optimizer.zero_grad()

            for key, value in components.items():
                epoch_totals[key] += float(value.item())
            epoch_totals["combined"] += float(combined_loss.item())

            if batch_idx % config.training.log_interval == 0:
                pbar.set_postfix({k: f"{float(v.item()):.4f}" for k, v in components.items()})

        if len(train_loader) % grad_accum != 0:
            optimizer.step()
            update_ema(ema_model, student, config.training.ema_decay)
            optimizer.zero_grad()

        scheduler.step()

        n_batches = len(train_loader)
        row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{k}": v / n_batches for k, v in epoch_totals.items()})

        if (epoch + 1) % config.training.eval_interval == 0 or (epoch + 1) == config.training.num_epochs:
            val_ssim, val_psnr = evaluate_model(ema_model, test_loader, device)
            row["val_ssim"] = val_ssim
            row["val_psnr"] = val_psnr
            student.train()

        logger.log(row)
        print(f"Epoch {epoch + 1} summary: {row}")

        if (epoch + 1) % config.training.eval_interval == 0 or (epoch + 1) == config.training.num_epochs:
            save_checkpoint(ema_model, checkpoints_dir / f"{run_name}_ema_latest.pth")
            save_checkpoint(student, checkpoints_dir / f"{run_name}_raw_latest.pth")
        if (epoch + 1) == config.training.num_epochs:
            save_checkpoint(ema_model, checkpoints_dir / f"{run_name}_ema_final_epoch_{epoch + 1:03d}.pth")
            save_checkpoint(student, checkpoints_dir / f"{run_name}_raw_final_epoch_{epoch + 1:03d}.pth")

    print(f"Training complete. Metrics logged to {logger.csv_path} and {logger.jsonl_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the KD student model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--run-name", default="student_kd")
    args = parser.parse_args()

    config = load_config(args.config, project_root=args.project_root)
    train(config, run_name=args.run_name)


if __name__ == "__main__":
    main()
