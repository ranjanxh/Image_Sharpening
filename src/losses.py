"""All five knowledge-distillation loss terms, each independently
toggleable via config (`losses.enabled.*`) so the Phase 3 ablation study
can turn terms off without touching the training loop.

Disabled terms are skipped entirely (not just zero-weighted) so ablation
runs don't pay for compute (e.g. a VGG forward pass) on a loss that isn't
being used.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import VGG19_Weights, vgg19

from src.config import LossesConfig


def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor([math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def _create_ssim_window(window_size: int, channel: int) -> torch.Tensor:
    _1d = _gaussian_window(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0, window_size: int = 11) -> torch.Tensor:
    """Differentiable SSIM over a batch of (B, C, H, W) tensors."""
    img1, img2 = img1.float(), img2.float()
    if data_range != 1.0:
        img1, img2 = img1 / data_range, img2 / data_range

    channel = img1.size(1)
    window = _create_ssim_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1, C2 = (0.01 * 1.0) ** 2, (0.03 * 1.0) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12
    )
    return ssim_map.mean()


class PerceptualLoss(nn.Module):
    """MSE between VGG19 features of two images (relu1_1..relu5_1 stack,
    truncated at layer 30 i.e. through relu5_1)."""

    def __init__(self, device):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:31])  # through relu5_1
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.mse = nn.MSELoss()

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        feat_gen = self.feature_extractor(self.normalize(generated))
        feat_tgt = self.feature_extractor(self.normalize(target))
        return self.mse(feat_gen, feat_tgt)


class CombinedLoss(nn.Module):
    """Computes the weighted sum of whichever loss terms are enabled.

    forward() returns (total_loss, components: dict[str, float]).
    `components` always contains an entry per known term; disabled terms
    report 0.0 rather than being omitted, so logging code doesn't need to
    special-case ablation configs.
    """

    def __init__(self, losses_config: LossesConfig, device):
        super().__init__()
        self.weights = losses_config.weights
        self.enabled = losses_config.enabled
        self.temperature = losses_config.kl_temperature

        self.l1 = nn.L1Loss() if self.enabled.get("recon_l1", True) else None
        self.feature_l1 = nn.L1Loss() if self.enabled.get("feature_distillation", True) else None
        self.perceptual = PerceptualLoss(device) if self.enabled.get("perceptual", True) else None

    def forward(
        self,
        student_output: torch.Tensor,
        sharp_images: torch.Tensor,
        teacher_output: torch.Tensor | None = None,
        student_feats: tuple[torch.Tensor | None, ...] = (None, None, None),
        teacher_feats: tuple[torch.Tensor | None, ...] = (None, None, None),
    ):
        device = student_output.device
        zero = torch.tensor(0.0, device=device)
        components: dict[str, torch.Tensor] = {}
        total = zero.clone()

        if self.enabled.get("recon_l1", True):
            recon = self.l1(student_output, sharp_images)
            components["recon_l1"] = recon
            total = total + self.weights["recon_l1"] * recon
        else:
            components["recon_l1"] = zero

        student_clamped = student_output.clamp(0, 1)
        teacher_clamped = teacher_output.clamp(0, 1) if teacher_output is not None else None

        if self.enabled.get("perceptual", True) and teacher_clamped is not None:
            perceptual = self.perceptual(student_clamped, teacher_clamped)
            components["perceptual"] = perceptual
            total = total + self.weights["perceptual"] * perceptual
        else:
            components["perceptual"] = zero

        if self.enabled.get("feature_distillation", True):
            pair_losses = [
                self.feature_l1(s_feat, t_feat)
                for s_feat, t_feat in zip(student_feats, teacher_feats)
                if s_feat is not None and t_feat is not None
            ]
            feat_dist = sum(pair_losses) / len(pair_losses) if pair_losses else zero
            components["feature_distillation"] = feat_dist
            total = total + self.weights["feature_distillation"] * feat_dist
        else:
            components["feature_distillation"] = zero

        if self.enabled.get("kl_div", True) and teacher_output is not None:
            T = self.temperature
            kl = F.kl_div(
                F.log_softmax(student_output / T, dim=1),
                F.softmax(teacher_output / T, dim=1),
                reduction="batchmean",
            ) * (T ** 2)
            components["kl_div"] = kl
            total = total + self.weights["kl_div"] * kl
        else:
            components["kl_div"] = zero

        if self.enabled.get("ssim", True):
            ssim_loss = 1 - ssim(student_clamped, sharp_images)
            components["ssim"] = ssim_loss
            total = total + self.weights["ssim"] * ssim_loss
        else:
            components["ssim"] = zero

        return total, components


def build_loss(losses_config: LossesConfig, device) -> CombinedLoss:
    return CombinedLoss(losses_config, device)
