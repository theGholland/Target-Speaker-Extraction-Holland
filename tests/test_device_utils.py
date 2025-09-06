import sys
from pathlib import Path

import torch

# Ensure src directory is on the path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import device_utils


def test_get_device_cpu(monkeypatch, capsys):
    """When no accelerator is available, CPU should be selected."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    device = device_utils.get_device()
    captured = capsys.readouterr().out
    assert device.type == "cpu"
    assert "CPU" in captured
