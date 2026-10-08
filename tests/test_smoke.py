import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from model import build_model


def test_forward_shapes():
    for name in ("cnn", "resnet50"):
        m = build_model(name, 6).eval()
        assert m(torch.randn(2, 3, 128, 128)).shape == (2, 6)
