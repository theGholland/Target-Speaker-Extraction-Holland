import sys
from pathlib import Path

import torch
import torchaudio

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from tse_select import sample_musan_noise


def test_sample_musan_noise(tmp_path):
    sr = 16000
    musan_root = tmp_path / "musan"
    noise_dir = musan_root / "noise"
    noise_dir.mkdir(parents=True)

    wav = torch.randn(sr // 2)
    torchaudio.save(str(noise_dir / "n.wav"), wav.unsqueeze(0), sr)

    out = sample_musan_noise(musan_root, "noise", sr, sr)
    assert out.shape[-1] == sr
    assert torch.isfinite(out).all()

