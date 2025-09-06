import sys
from pathlib import Path

import torch
import torchaudio
import pytest

# Add src directory to sys.path for importing tse_select
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from tse_select import mix_at_snr, compute_si_sdr


def test_mix_long_target_short_noise(tmp_path):
    sr = 16000
    target = torch.randn(sr * 2)  # 2 seconds
    noise = torch.randn(sr // 2)  # 0.5 seconds

    mixture = mix_at_snr(target, noise, 0.0)
    assert mixture.shape[-1] == sr // 2

    # Slice target to mixture length as done in tse_select
    target = target[: mixture.shape[-1]]

    mix_path = tmp_path / "mix.wav"
    tgt_path = tmp_path / "target.wav"

    torchaudio.save(str(mix_path), mixture.unsqueeze(0), sr)
    torchaudio.save(str(tgt_path), target.unsqueeze(0), sr)

    loaded_mix, _ = torchaudio.load(str(mix_path))
    loaded_tgt, _ = torchaudio.load(str(tgt_path))

    # Ensure saved files have matching lengths
    assert loaded_mix.shape[-1] == loaded_tgt.shape[-1]

    # compute_si_sdr should operate without size mismatch
    val = compute_si_sdr(loaded_mix.squeeze(0), loaded_tgt.squeeze(0))
    float(val)  # should be convertible to float without error


def test_compute_si_sdr_trims_and_warns():
    est = torch.randn(1000)
    ref = torch.randn(800)

    with pytest.warns(UserWarning):
        val = compute_si_sdr(est, ref)

    expected = compute_si_sdr(est[:800], ref)
    assert torch.isclose(val, expected)
