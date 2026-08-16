import torch

from src.models.student import StudentModel
from src.utils import load_checkpoint, save_checkpoint


def test_checkpoint_save_load_roundtrip(tmp_path):
    device = torch.device("cpu")
    model = StudentModel(in_channels=3, out_channels=3, base_channels=8)
    ckpt_path = tmp_path / "student.pth"

    save_checkpoint(model, ckpt_path)
    assert ckpt_path.exists()

    x = torch.randn(1, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        original_out, *_ = model(x)

    reloaded = StudentModel(in_channels=3, out_channels=3, base_channels=8)
    load_checkpoint(reloaded, ckpt_path, device)
    reloaded.eval()
    with torch.no_grad():
        reloaded_out, *_ = reloaded(x)

    assert torch.allclose(original_out, reloaded_out, atol=1e-6)


def test_checkpoint_creates_parent_dirs(tmp_path):
    model = StudentModel(in_channels=3, out_channels=3, base_channels=8)
    nested_path = tmp_path / "nested" / "dir" / "student.pth"
    save_checkpoint(model, nested_path)
    assert nested_path.exists()
