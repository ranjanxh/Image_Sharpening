"""Small shared utilities: seeding, PSNR, EMA update, checkpoint I/O."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """PSNR between two (C, H, W) tensors in [0, data_range]."""
    img1 = img1.detach().cpu().float()
    img2 = img2.detach().cpu().float()
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * float(torch.log10(torch.tensor(data_range ** 2) / mse))


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.copy_(decay * ema_param + (1.0 - decay) * param)


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))


def load_checkpoint(model: torch.nn.Module, path: str | Path, device) -> torch.nn.Module:
    state_dict = torch.load(str(path), map_location=device)
    model.load_state_dict(state_dict)
    return model
