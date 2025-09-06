"""Tests to ensure OpenVINO Demucs runs on GPU."""

from pathlib import Path

import pytest
import torch


def _dummy_download(**kwargs):
    """Return a dummy path for hf_hub_download calls."""
    filename = kwargs.get("filename", "model.xml")
    local_dir = Path(kwargs.get("local_dir", "."))
    local_dir.mkdir(parents=True, exist_ok=True)
    path = local_dir / filename
    path.touch()
    return path


class _DummyCompiledModel:
    def output(self, idx: int):
        return idx


class _DummyCore:
    """Minimal OpenVINO Core stub capturing compile target."""

    available_devices = ["GPU"]

    def __init__(self):
        self.last_device = None

    def read_model(self, *_args, **_kwargs):  # pragma: no cover - trivially exercised
        return "ov_model"

    def compile_model(self, model, device):  # pragma: no cover - exercised by test
        self.last_device = device
        return _DummyCompiledModel()


def test_load_sep_model_uses_gpu(monkeypatch):
    """load_sep_model should compile OpenVINO models on GPU."""

    from eval_tse_on_voices import load_sep_model

    dummy_core = _DummyCore()
    monkeypatch.setattr("openvino.runtime.Core", lambda: dummy_core)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _dummy_download)

    load_sep_model("demucs", torch.device("cpu"))
    assert dummy_core.last_device == "GPU"


def test_load_sep_model_errors_without_gpu(monkeypatch):
    """load_sep_model should raise when OpenVINO GPU is unavailable."""

    from eval_tse_on_voices import load_sep_model

    class NoGpuCore(_DummyCore):
        available_devices = ["CPU"]

    monkeypatch.setattr("openvino.runtime.Core", lambda: NoGpuCore())
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _dummy_download)

    with pytest.raises(RuntimeError):
        load_sep_model("demucs", torch.device("cpu"))

