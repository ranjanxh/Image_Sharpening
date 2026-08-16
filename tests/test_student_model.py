import torch

from src.models.student import StudentModel


def test_student_forward_shapes():
    model = StudentModel(in_channels=3, out_channels=3, base_channels=8)
    x = torch.randn(1, 3, 64, 64)
    out, e2, e4, b = model(x)

    assert out.shape == (1, 3, 64, 64)
    assert e2.shape == (1, 16, 32, 32)
    assert e4.shape == (1, 64, 8, 8)
    assert b.shape == (1, 128, 4, 4)


def test_student_forward_nonsquare_input():
    model = StudentModel(in_channels=3, out_channels=3, base_channels=8)
    x = torch.randn(1, 3, 48, 80)
    out, *_ = model(x)
    assert out.shape == (1, 3, 48, 80)


def test_student_has_trainable_params():
    model = StudentModel(in_channels=3, out_channels=3, base_channels=8)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 0
