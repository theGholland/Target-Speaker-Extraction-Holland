import sys
import types
import torch
import subprocess
import builtins

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from eval_tse_on_voices import load_sep_model


class DummyModel(torch.nn.Module):
    pass


def test_loads_existing_demucs(monkeypatch):
    """When demucs is available, load_sep_model should return it without pip."""
    dummy = DummyModel()
    dummy_pretrained = types.SimpleNamespace(get_model=lambda name: dummy)
    dummy_demucs = types.SimpleNamespace(pretrained=dummy_pretrained)
    monkeypatch.setitem(sys.modules, "demucs", dummy_demucs)
    monkeypatch.setitem(sys.modules, "demucs.pretrained", dummy_pretrained)
    called = False

    def fake_check_call(cmd):
        nonlocal called
        called = True

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)
    model = load_sep_model("demucs", torch.device("cpu"))
    assert model is dummy
    assert not called


def test_installs_demucs_when_missing(monkeypatch):
    """If demucs is missing, load_sep_model should install it via pip."""
    monkeypatch.delitem(sys.modules, "demucs", raising=False)
    monkeypatch.delitem(sys.modules, "demucs.pretrained", raising=False)
    dummy = DummyModel()
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("demucs"):
            raise ModuleNotFoundError
        return real_import(name, *args, **kwargs)

    def fake_check_call(cmd):
        assert cmd[-1] == "demucs"
        dummy_pretrained = types.SimpleNamespace(get_model=lambda name: dummy)
        dummy_demucs = types.SimpleNamespace(pretrained=dummy_pretrained)
        sys.modules["demucs"] = dummy_demucs
        sys.modules["demucs.pretrained"] = dummy_pretrained
        monkeypatch.setattr(builtins, "__import__", real_import)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(subprocess, "check_call", fake_check_call)
    model = load_sep_model("demucs", torch.device("cpu"))
    assert model is dummy
