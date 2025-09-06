import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import tse_select
import eval_tse_on_voices

def _test_mix_fn(mix_fn):
    target = torch.randn(16000)
    noise = torch.zeros_like(target)
    mixed = mix_fn(target, noise, 0.0)
    assert torch.allclose(mixed, target)
    assert torch.isfinite(mixed).all()

def test_mix_at_snr_select():
    _test_mix_fn(tse_select.mix_at_snr)

def test_mix_at_snr_eval():
    _test_mix_fn(eval_tse_on_voices.mix_at_snr)
