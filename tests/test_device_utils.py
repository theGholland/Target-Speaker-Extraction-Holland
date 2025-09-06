import sys
from pathlib import Path

import pytest
import torch

# Ensure src directory is on the path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import device_utils


def test_get_device_cuda(monkeypatch, capsys):
    """Return CUDA device when available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Fake GPU")
    device = device_utils.get_device()
    captured = capsys.readouterr().out
    assert device.type == "cuda"
    assert "CUDA" in captured


def test_get_device_no_accelerator_raises(monkeypatch):
    """Raise error when no accelerator is present."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        device_utils.get_device()

