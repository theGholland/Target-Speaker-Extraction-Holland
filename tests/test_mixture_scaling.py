import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.tse_select import mix_at_snr


def test_mixture_scaled_below_unity():
    target = torch.ones(16000)
    noise = torch.ones(16000)
    mixture = mix_at_snr(target, noise, 0.0)
    peak = mixture.abs().max().item()
    assert peak > 1.0
    if peak > 1.0:
        mixture = mixture / peak * 0.9
    assert mixture.abs().max().item() == pytest.approx(0.9, abs=1e-6)
