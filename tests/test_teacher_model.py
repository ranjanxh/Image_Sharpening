
import pytest
import torch

from src.config import REPO_ROOT, load_config, resolve_device
from src.models.teacher import build_teacher

WEIGHTS_PATH = REPO_ROOT / "Restormer" / "Motion_Deblurring" / "pretrained_models" / "motion_deblurring.pth"
SUBMODULE_ARCH_PATH = REPO_ROOT / "Restormer" / "basicsr" / "models" / "archs" / "restormer_arch.py"

pytestmark = pytest.mark.skipif(
    not (WEIGHTS_PATH.exists() and SUBMODULE_ARCH_PATH.exists()),
    reason="Restormer submodule and/or motion_deblurring.pth weights not present locally "
    "(run `git submodule update --init --recursive` and download the weights; "
    "see README for the exact URL).",
)


@pytest.fixture(scope="module")
def teacher():
    config = load_config("configs/default.yaml")
    device = resolve_device("cpu")
    return build_teacher(config, device), device


def test_teacher_loads_real_weights_with_exact_match(teacher):
    model, _ = teacher
    # build_teacher() itself raises RuntimeError on any missing/unexpected
    # key -- reaching this point already proves an exact state_dict match.
    assert sum(p.numel() for p in model.restormer.parameters()) > 0


def test_teacher_forward_shapes(teacher):
    model, device = teacher
    x = torch.randn(1, 3, 64, 64, device=device)
    with torch.no_grad():
        out, feat_e2, feat_deep, feat_b = model(x)

    assert out.shape == (1, 3, 64, 64)
    # feature taps are projected to student_base_channels multiples (256 in default.yaml)
    assert feat_e2.shape[0] == 1 and feat_e2.shape[1] == 256 * 2
    assert feat_deep.shape[0] == 1 and feat_deep.shape[1] == 256 * 8
    assert feat_b.shape[0] == 1 and feat_b.shape[1] == 256 * 16


def test_teacher_is_frozen(teacher):
    model, _ = teacher
    assert all(not p.requires_grad for p in model.restormer.parameters())
