"""Teacher model: a frozen, pretrained Restormer wrapped to also expose
intermediate features for knowledge distillation.

This module has no import-time side effects -- build_teacher() must be
called explicitly to instantiate and load weights.
"""
from __future__ import annotations

from pathlib import Path
from runpy import run_path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config, TeacherConfig


def _load_restormer_class(restormer_repo: Path):
    """Load the Restormer nn.Module class straight from the vendored
    submodule source, without needing `basicsr` installed as a package
    (the arch file only depends on torch/einops)."""
    arch_path = restormer_repo / "basicsr" / "models" / "archs" / "restormer_arch.py"
    if not arch_path.exists():
        raise FileNotFoundError(
            f"Restormer architecture file not found at {arch_path}. "
            "Did you run `git submodule update --init --recursive`?"
        )
    namespace = run_path(str(arch_path))
    return namespace["Restormer"]


class RestormerTeacherWrapper(nn.Module):
    """Wraps a frozen Restormer, padding inputs to a multiple-of-N size and
    exposing three intermediate feature taps (shallow encoder, deep encoder,
    bottleneck) projected to the student's channel widths for feature-level
    distillation.

    NOTE on a corrected bug: the original script hooked a nonexistent
    `encoder_level4` attribute (this Restormer implementation only has
    encoder_level1/2/3 followed directly by the `latent` bottleneck -- there
    is no separate 4th encoder stage). That hook silently never fired, so
    "deep encoder" feature distillation never actually happened in the
    original code. This version hooks `encoder_level3` instead (the actual
    deepest encoder stage, immediately before the bottleneck) and sizes its
    projection conv to that stage's real channel width (dim*4), not the
    incorrect assumed width (dim*8) the original code used.
    """

    def __init__(self, teacher_config: TeacherConfig, weights_path: Path,
                 restormer_repo: Path, student_base_channels: int, device):
        super().__init__()
        self.img_multiple_of = teacher_config.img_multiple_of
        self.device = device

        RestormerClass = _load_restormer_class(restormer_repo)
        model_parameters = dict(
            inp_channels=teacher_config.in_channels,
            out_channels=teacher_config.out_channels,
            dim=teacher_config.dim,
            num_blocks=teacher_config.num_blocks,
            num_refinement_blocks=teacher_config.num_refinement_blocks,
            heads=teacher_config.heads,
            ffn_expansion_factor=teacher_config.ffn_expansion_factor,
            bias=teacher_config.bias,
            LayerNorm_type=teacher_config.layer_norm_type,
            dual_pixel_task=teacher_config.dual_pixel_task,
        )
        self.restormer = RestormerClass(**model_parameters)

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Restormer pretrained weights not found at {weights_path}. "
                "Download from: "
                "https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth"
            )
        state_dict = torch.load(str(weights_path), map_location=device)
        if "params" in state_dict:
            state_dict = state_dict["params"]
        state_dict = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        missing, unexpected = self.restormer.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Restormer checkpoint did not match architecture exactly. "
                f"missing={missing} unexpected={unexpected}"
            )

        self.restormer.eval()
        self.restormer.to(device)
        for p in self.restormer.parameters():
            p.requires_grad = False

        dim = teacher_config.dim
        self._teacher_features = {"e2": None, "e_deep": None, "b": None}

        def make_hook(key):
            def hook(module, inp, out):
                self._teacher_features[key] = out
            return hook

        self.proj_e2 = None
        self.proj_deep = None
        self.proj_b = None

        if hasattr(self.restormer, "encoder_level2"):
            self.restormer.encoder_level2.register_forward_hook(make_hook("e2"))
            self.proj_e2 = nn.Conv2d(dim * 2, student_base_channels * 2, kernel_size=1).to(device)

        if hasattr(self.restormer, "encoder_level3"):
            self.restormer.encoder_level3.register_forward_hook(make_hook("e_deep"))
            self.proj_deep = nn.Conv2d(dim * 4, student_base_channels * 8, kernel_size=1).to(device)

        if hasattr(self.restormer, "latent"):
            self.restormer.latent.register_forward_hook(make_hook("b"))
            self.proj_b = nn.Conv2d(dim * 8, student_base_channels * 16, kernel_size=1).to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        h_orig, w_orig = x.shape[2], x.shape[3]
        H = ((h_orig + self.img_multiple_of - 1) // self.img_multiple_of) * self.img_multiple_of
        W = ((w_orig + self.img_multiple_of - 1) // self.img_multiple_of) * self.img_multiple_of
        x_padded = F.pad(x, (0, W - w_orig, 0, H - h_orig), mode="reflect")

        restored_padded = self.restormer(x_padded)
        restored = restored_padded[:, :, :h_orig, :w_orig]

        e2_raw = self._teacher_features["e2"]
        deep_raw = self._teacher_features["e_deep"]
        b_raw = self._teacher_features["b"]

        feat_e2 = self.proj_e2(e2_raw) if (e2_raw is not None and self.proj_e2 is not None) else None
        feat_deep = self.proj_deep(deep_raw) if (deep_raw is not None and self.proj_deep is not None) else None
        feat_b = self.proj_b(b_raw) if (b_raw is not None and self.proj_b is not None) else None

        self._teacher_features = {"e2": None, "e_deep": None, "b": None}
        return restored, feat_e2, feat_deep, feat_b


def build_teacher(config: Config, device) -> RestormerTeacherWrapper:
    return RestormerTeacherWrapper(
        teacher_config=config.model.teacher,
        weights_path=config.resolve(config.paths.restormer_weights),
        restormer_repo=config.resolve(config.paths.restormer_repo),
        student_base_channels=config.model.student.base_channels,
        device=device,
    )
