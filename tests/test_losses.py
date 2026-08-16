import pytest
import torch

from src.config import LossesConfig
from src.losses import CombinedLoss, ssim


def _all_enabled_config():
    return LossesConfig(
        weights={"recon_l1": 1.0, "perceptual": 0.01, "feature_distillation": 1.0, "kl_div": 0.01, "ssim": 10.0},
        enabled={"recon_l1": True, "perceptual": True, "feature_distillation": True, "kl_div": True, "ssim": True},
        kl_temperature=4.0,
    )


def _sample_batch():
    torch.manual_seed(0)
    student_out = torch.rand(1, 3, 32, 32)
    sharp = torch.rand(1, 3, 32, 32)
    teacher_out = torch.rand(1, 3, 32, 32)
    student_feats = (torch.rand(1, 4, 16, 16), torch.rand(1, 8, 8, 8), torch.rand(1, 16, 4, 4))
    teacher_feats = (torch.rand(1, 4, 16, 16), torch.rand(1, 8, 8, 8), torch.rand(1, 16, 4, 4))
    return student_out, sharp, teacher_out, student_feats, teacher_feats


def test_ssim_identical_images_is_one():
    img = torch.rand(1, 3, 32, 32)
    assert ssim(img, img).item() == pytest.approx(1.0, abs=1e-4)


def test_all_five_terms_present_and_nonzero_when_enabled():
    device = torch.device("cpu")
    loss_fn = CombinedLoss(_all_enabled_config(), device)
    student_out, sharp, teacher_out, student_feats, teacher_feats = _sample_batch()

    total, components = loss_fn(student_out, sharp, teacher_out, student_feats, teacher_feats)

    assert set(components.keys()) == {"recon_l1", "perceptual", "feature_distillation", "kl_div", "ssim"}
    for name, value in components.items():
        assert value.item() != 0.0, f"{name} unexpectedly zero on random (non-identical) inputs"
    assert total.item() > 0


def test_disabled_term_contributes_zero_and_is_skipped():
    config = _all_enabled_config()
    config.enabled["kl_div"] = False
    device = torch.device("cpu")
    loss_fn = CombinedLoss(config, device)
    student_out, sharp, teacher_out, student_feats, teacher_feats = _sample_batch()

    total, components = loss_fn(student_out, sharp, teacher_out, student_feats, teacher_feats)
    assert components["kl_div"].item() == 0.0


def test_perceptual_disabled_skips_vgg_construction():
    config = _all_enabled_config()
    config.enabled["perceptual"] = False
    device = torch.device("cpu")
    loss_fn = CombinedLoss(config, device)
    assert loss_fn.perceptual is None


def test_recon_l1_matches_manual_l1():
    device = torch.device("cpu")
    loss_fn = CombinedLoss(_all_enabled_config(), device)
    student_out, sharp, teacher_out, student_feats, teacher_feats = _sample_batch()
    _, components = loss_fn(student_out, sharp, teacher_out, student_feats, teacher_feats)
    expected = torch.nn.L1Loss()(student_out, sharp)
    assert abs(components["recon_l1"].item() - expected.item()) < 1e-6


def test_feature_distillation_averages_available_pairs():
    device = torch.device("cpu")
    loss_fn = CombinedLoss(_all_enabled_config(), device)
    student_out, sharp, teacher_out, student_feats, teacher_feats = _sample_batch()
    # drop one teacher feature to simulate a missing hook tap (e.g. e4 in the
    # pre-fix code) -- feature_distillation should average over remaining pairs only
    teacher_feats_missing = (teacher_feats[0], None, teacher_feats[2])
    _, components = loss_fn(student_out, sharp, teacher_out, student_feats, teacher_feats_missing)
    l1 = torch.nn.L1Loss()
    expected = (l1(student_feats[0], teacher_feats[0]) + l1(student_feats[2], teacher_feats[2])) / 2
    assert abs(components["feature_distillation"].item() - expected.item()) < 1e-6
