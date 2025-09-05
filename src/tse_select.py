import argparse
import os
import time
from typing import Optional, Iterable

import torch
import torchaudio
from pathlib import Path


def load_audio(path: str, target_sr: Optional[int] = None) -> tuple[torch.Tensor, int]:
    """Load audio file, convert to mono, optionally resample."""
    wav, sr = torchaudio.load(path)
    if wav.ndim > 1:
        wav = wav.mean(dim=0)
    if target_sr is not None and sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav, sr


def mix_at_snr(target: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Mix target and noise at a given SNR."""
    # Ensure same length
    L = min(target.shape[-1], noise.shape[-1])
    target = target[..., :L]
    noise = noise[..., :L]
    target_power = target.pow(2).mean()
    noise_power = noise.pow(2).mean()
    scale = torch.sqrt(target_power / noise_power) * (10 ** (-snr_db / 20))
    noise = noise * scale
    return target + noise

def compose_babble(wavs: Iterable[torch.Tensor], length: int) -> torch.Tensor:
    """Create uniform babble by looping and level-equalizing input waves."""
    wavs = list(wavs)
    if not wavs:
        return torch.zeros(length)
    processed = []
    for w in wavs:
        if w.shape[-1] < length:
            reps = (length + w.shape[-1] - 1) // w.shape[-1]
            w = w.repeat(reps)[:length]
        else:
            w = w[:length]
        w = w - w.mean()
        rms = w.pow(2).mean().sqrt()
        if rms > 0:
            w = w / rms
        processed.append(w)
    babble = torch.stack(processed).mean(dim=0)
    return babble


def ecapa_embedding(model, wav: torch.Tensor, sr: int, device: torch.device) -> torch.Tensor:
    """Compute a speaker embedding using NeMo ECAPA."""
    with torch.no_grad():
        emb = model.get_embedding(wav.to(device), sr=sr)
        emb = torch.nn.functional.normalize(emb.squeeze(0), dim=-1)
    return emb


def compute_si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """Compute SI-SDR between estimate and reference."""
    def zero_mean(x):
        return x - x.mean()

    estimate = zero_mean(estimate)
    reference = zero_mean(reference)
    s = torch.dot(estimate, reference) / torch.dot(reference, reference) * reference
    e = estimate - s
    return 10 * torch.log10(torch.dot(s, s) / torch.dot(e, e))


def main():
    parser = argparse.ArgumentParser(description="Target speaker extraction by selection")
    parser.add_argument("--target", type=str, required=True, help="Path to clean target/enrollment audio")
    parser.add_argument("--noise", type=str, help="Path to noise/interferer audio")
    parser.add_argument("--create_noise", action="store_true", help="Create babble noise from other voices")
    parser.add_argument("--snr_db", type=float, default=0.0, help="SNR for mixing target and noise")
    parser.add_argument(
        "--model",
        type=str,
        choices=["dprnn", "convtasnet", "demucs"],
        default="dprnn",
        help="Separation model to use",
    )
    args = parser.parse_args()

    # Determine device
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        raise RuntimeError("No CUDA or MPS accelerator available")

    # Load enrollment/clean target
    target_wav, sr = load_audio(args.target)

    # If noise provided or to be created, create mixture
    if args.create_noise:
        target_path = Path(args.target)
        voices_dir = target_path.parent.parent
        speaker_dir = target_path.parent.name
        babble_wavs = []
        for spk in voices_dir.iterdir():
            if not spk.is_dir() or spk.name == speaker_dir:
                continue
            wav, sr = load_audio(str(spk / "target.wav"), sr)
            babble_wavs.append(wav)
        if not babble_wavs:
            raise RuntimeError("No other speakers found for babble noise")
        noise_wav = compose_babble(babble_wavs, target_wav.shape[-1])
        mixture = mix_at_snr(target_wav, noise_wav, args.snr_db)
    elif args.noise:
        noise_wav, sr = load_audio(args.noise, sr)
        mixture = mix_at_snr(target_wav, noise_wav, args.snr_db)
    else:
        mixture = target_wav

    audio_duration = mixture.shape[-1] / sr

    # Load separation model
    if args.model == "dprnn":
        from asteroid.models import DPRNNTasNet

        model = DPRNNTasNet.from_pretrained("mpariente/DPRNNTasNet_WHAM!_sepclean_16k").to(device)
        model.eval()
        start = time.time()
        with torch.no_grad():
            est_sources = model(mixture.unsqueeze(0).to(device))
    elif args.model == "convtasnet":
        from asteroid.models import ConvTasNet

        model = ConvTasNet.from_pretrained("mpariente/ConvTasNet_WHAM!_sepclean_16k").to(device)
        model.eval()
        start = time.time()
        with torch.no_grad():
            est_sources = model(mixture.unsqueeze(0).to(device))
    else:  # demucs
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        model = get_model("htdemucs").to(device)
        model.eval()
        start = time.time()
        with torch.no_grad():
            est_sources = apply_model(
                model, mixture.unsqueeze(0).unsqueeze(0).to(device), split=True, progress=False
            )[0]

    processing_time = time.time() - start
    est_sources = est_sources.squeeze(0).cpu()

    # Compute embeddings using NeMo ECAPA
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    spk_model = EncDecSpeakerLabelModel.from_pretrained("speakerverification_ecapa").to(device)
    spk_model.eval()

    enroll_emb = ecapa_embedding(spk_model, target_wav, sr, device)
    est_embs = [ecapa_embedding(spk_model, src, sr, device) for src in est_sources]
    scores = [torch.nn.functional.cosine_similarity(enroll_emb, emb, dim=0).item() for emb in est_embs]
    chosen_idx = int(torch.argmax(torch.tensor(scores)))
    tse_result = est_sources[chosen_idx]

    # Write output WAVs
    out_dir = os.path.dirname(args.target)
    torchaudio.save(os.path.join(out_dir, "sep_source0.wav"), est_sources[0].unsqueeze(0), sr)
    torchaudio.save(os.path.join(out_dir, "sep_source1.wav"), est_sources[1].unsqueeze(0), sr)
    torchaudio.save(os.path.join(out_dir, "tse_result.wav"), tse_result.unsqueeze(0), sr)

    rtf = processing_time / audio_duration if audio_duration > 0 else float('inf')

    print(f"Similarity scores: {scores}")
    print(f"Chosen source: {chosen_idx}")
    if args.target:
        si_sdr_value = compute_si_sdr(tse_result, target_wav)
        print(f"SI-SDR: {si_sdr_value:.2f} dB")
    print(f"RTF: {rtf:.3f}")
    print(f"Processing time: {processing_time:.2f} s for {audio_duration:.2f} s of audio")


if __name__ == "__main__":
    main()
