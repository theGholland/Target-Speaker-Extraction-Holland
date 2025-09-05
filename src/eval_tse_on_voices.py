#!/usr/bin/env python3
"""Batch evaluation of target speaker extraction with babble sweeps."""

import argparse
import csv
import itertools
import random
import time
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    """CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Batch evaluation with uniform babble and parameter sweeps",
    )
    parser.add_argument(
        "--voices_dir",
        type=Path,
        default=Path("data/voices"),
        help="Directory containing speaker subfolders with enroll.wav and target.wav",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("out_eval"),
        help="Directory for evaluation outputs",
    )
    parser.add_argument("--snr_db", type=float, help="SNR value for a single run")
    parser.add_argument(
        "--num_babble_voices", type=int, help="Number of babble voices for a single run",
    )
    parser.add_argument(
        "--snr_list",
        type=str,
        help="Comma-separated list of SNR values for sweeping",
    )
    parser.add_argument(
        "--babble_list",
        type=str,
        help="Comma-separated list of babble counts for sweeping",
    )
    parser.add_argument(
        "--sep_models",
        type=str,
        default="dprnn",
        help="Comma-separated list of separation models to evaluate (dprnn, convtasnet, demucs)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on number of speakers to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed controlling selection order",
    )
    return parser.parse_args()


def load_audio(path: Path, target_sr: int | None = None):
    """Load mono audio and optionally resample."""
    import torchaudio

    wav, sr = torchaudio.load(path)
    if wav.ndim > 1:
        wav = wav.mean(dim=0)
    if target_sr is not None and sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.squeeze(0), sr


def mix_at_snr(target, noise, snr_db: float):
    """Mix target and noise at a given SNR."""
    import torch

    L = min(target.shape[-1], noise.shape[-1])
    target = target[..., :L]
    noise = noise[..., :L]
    target_power = target.pow(2).mean()
    noise_power = noise.pow(2).mean()
    scale = torch.sqrt(target_power / noise_power) * (10 ** (-snr_db / 20))
    noise = noise * scale
    return target + noise


def ecapa_embedding(model, wav, sr: int, device):
    """Compute speaker embedding using a pre-loaded model."""
    import torch

    with torch.no_grad():
        emb = model.get_embedding(wav.to(device), sr=sr)
        emb = torch.nn.functional.normalize(emb.squeeze(0), dim=-1)
    return emb


def compute_si_sdr(estimate, reference) -> float:
    """Compute SI-SDR between estimate and reference."""
    import torch

    def zero_mean(x):
        return x - x.mean()

    estimate = zero_mean(estimate)
    reference = zero_mean(reference)
    s = torch.dot(estimate, reference) / torch.dot(reference, reference) * reference
    e = estimate - s
    return 10 * torch.log10(torch.dot(s, s) / torch.dot(e, e))


def compose_babble(wavs: Iterable, length: int):
    """Create uniform babble by looping and level-equalizing input waves."""
    import torch

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


def main() -> None:
    args = parse_args()

    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        raise RuntimeError("No CUDA or MPS accelerator available")

    # Determine evaluation combinations
    if args.snr_db is not None and args.num_babble_voices is not None:
        combos = [(args.snr_db, args.num_babble_voices)]
    elif args.snr_list and args.babble_list:
        snr_values = [float(s) for s in args.snr_list.split(",")]
        babble_values = [int(b) for b in args.babble_list.split(",")]
        combos = list(itertools.product(snr_values, babble_values))
    else:
        raise ValueError(
            "Specify either --snr_db and --num_babble_voices or --snr_list and --babble_list"
        )

    sep_models = [m.strip().lower() for m in args.sep_models.split(",")]

    # Gather speakers
    speakers = [p for p in args.voices_dir.iterdir() if p.is_dir()]
    rng = random.Random(args.seed)
    rng.shuffle(speakers)
    if args.limit:
        speakers = speakers[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    spk_model = EncDecSpeakerLabelModel.from_pretrained(
        "ecapa_tdnn",
    ).to(device)
    spk_model.eval()

    for model_name in sep_models:
        if model_name == "dprnn":
            from asteroid.models import DPRNNTasNet

            sep_model = DPRNNTasNet.from_pretrained(
                "mpariente/DPRNNTasNet_WHAM!_sepclean_16k"
            ).to(device)
        elif model_name == "convtasnet":
            from asteroid.models import ConvTasNet

            sep_model = ConvTasNet.from_pretrained(
                "mpariente/ConvTasNet_WHAM!_sepclean_16k"
            ).to(device)
        elif model_name == "demucs":
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            sep_model = get_model("htdemucs").to(device)
        else:
            raise ValueError(f"Unknown separation model: {model_name}")
        sep_model.eval()

        for snr_db, num_babble in combos:
            for idx, spk_dir in enumerate(speakers):
                enroll_wav, sr = load_audio(spk_dir / "enroll.wav")
                target_wav, sr = load_audio(spk_dir / "target.wav", sr)

                # Select babble speakers deterministically
                babbler_dirs = [
                    speakers[(idx + 1 + j) % len(speakers)] for j in range(num_babble)
                ]
                babble_wavs = [
                    load_audio(b / "target.wav", sr)[0] for b in babbler_dirs
                ]
                babble = compose_babble(babble_wavs, target_wav.shape[-1])

                mixture = mix_at_snr(target_wav, babble, snr_db)
                peak = mixture.abs().max().item()
                if peak > 1.0:
                    mixture = mixture / peak * 0.9  # apply headroom

                audio_duration = mixture.shape[-1] / sr

                start = time.time()
                if model_name == "demucs":
                    with torch.no_grad():
                        est_sources = apply_model(
                            sep_model,
                            mixture.unsqueeze(0).unsqueeze(0).to(device),
                            split=True,
                            progress=False,
                        )[0]
                else:
                    with torch.no_grad():
                        est_sources = sep_model(mixture.unsqueeze(0).to(device)).squeeze(0)
                processing_time = time.time() - start
                est_sources = est_sources.cpu()

                enroll_emb = ecapa_embedding(spk_model, enroll_wav, sr, device)
                est_embs = [ecapa_embedding(spk_model, src, sr, device) for src in est_sources]
                scores = [
                    torch.nn.functional.cosine_similarity(enroll_emb, emb, dim=0).item()
                    for emb in est_embs
                ]
                chosen_idx = int(torch.argmax(torch.tensor(scores)))
                tse_result = est_sources[chosen_idx]

                rtf = processing_time / audio_duration if audio_duration > 0 else float("inf")
                if rtf > 0.5:
                    print(
                        f"Rejected {spk_dir.name} model={model_name} snr={snr_db} "
                        f"babble={num_babble} RTF={rtf:.3f}"
                    )
                    continue

                si_sdr = compute_si_sdr(tse_result, target_wav).item()
                results.append(
                    [spk_dir.name, model_name, snr_db, num_babble, si_sdr, rtf]
                )
                print(
                    f"speaker={spk_dir.name} model={model_name} snr={snr_db} babble={num_babble} "
                    f"si_sdr={si_sdr:.2f} rtf={rtf:.3f}"
                )

    # Write results
    if results:
        csv_path = args.out_dir / "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["speaker", "model", "snr_db", "num_babble", "si_sdr", "rtf"])
            writer.writerows(results)


if __name__ == "__main__":
    main()
