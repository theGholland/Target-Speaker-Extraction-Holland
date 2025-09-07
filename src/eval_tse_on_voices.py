#!/usr/bin/env python3
"""Batch evaluation of target speaker extraction with babble sweeps."""

import argparse
import csv
import random
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
import torchaudio

from device_utils import get_device

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
        help="Comma-separated SNR values; zipped with babble_list and sep_models",
    )
    parser.add_argument(
        "--babble_list",
        type=str,
        help="Comma-separated babbler counts; zipped with snr_list and sep_models",
    )
    parser.add_argument(
        "--sep_models",
        type=str,
        default="dprnn",
        help=(
            "Comma-separated separation models (dprnn, convtasnet, demucs); "
            "zipped with snr_list/babble_list when lists are used"
        ),
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
    parser.add_argument(
        "--musan_dir",
        type=Path,
        help="Root of MUSAN dataset for background noise",
    )
    parser.add_argument(
        "--musan_category",
        type=str,
        default="noise",
        choices=["noise", "music"],
        help="MUSAN subset to sample from when --musan_dir is provided",
    )
    parser.add_argument(
        "--eval_text",
        action="store_true",
        help="Evaluate ASR accuracy when target transcripts are available",
    )

    args = parser.parse_args()
    if args.musan_dir and (args.num_babble_voices is not None or args.babble_list):
        parser.error("--musan_dir cannot be combined with babble arguments")
    return args


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
    if noise_power == 0:
        return target
    scale = torch.sqrt(target_power / noise_power) * (10 ** (-snr_db / 20))
    noise = noise * scale
    return target + noise


def ecapa_embedding(model, wav, device):
    """Compute speaker embedding using a pre-loaded model."""
    import torch

    with torch.no_grad():
        wav = wav.unsqueeze(0).to(device)
        length = torch.tensor([wav.shape[-1]], device=device)
        _, emb = model.forward(input_signal=wav, input_signal_length=length)
        emb = torch.nn.functional.normalize(emb.squeeze(0), dim=-1)
    return emb


def compute_si_sdr(estimate, reference) -> float:
    """Compute SI-SDR between estimate and reference."""
    import torch
    import warnings

    if estimate.shape[-1] != reference.shape[-1]:
        min_len = min(estimate.shape[-1], reference.shape[-1])
        warnings.warn(
            f"Trimming inputs for SI-SDR: {estimate.shape[-1]} and {reference.shape[-1]} -> {min_len}",
            stacklevel=2,
        )
        estimate = estimate[..., :min_len]
        reference = reference[..., :min_len]

    def zero_mean(x):
        return x - x.mean()

    estimate = zero_mean(estimate)
    reference = zero_mean(reference)
    s = torch.dot(estimate, reference) / torch.dot(reference, reference) * reference
    e = estimate - s
    return 10 * torch.log10(torch.dot(s, s) / torch.dot(e, e))


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute the word error rate between reference and hypothesis texts."""

    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    # Initialize distance matrix
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # deletion
                d[i][j - 1] + 1,  # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )

    return d[-1][-1] / max(len(ref_words), 1)



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


def sample_musan_noise(musan_root: Path, category: str, sr: int, length: int):
    """Sample a noise clip from the MUSAN dataset.

    Parameters
    ----------
    musan_root: Path
        Root directory of the MUSAN dataset containing ``noise``/``music`` folders.
    category: str
        Which subfolder of MUSAN to sample from (e.g., ``"noise"`` or ``"music"``).
    sr: int
        Target sample rate for the returned noise.
    length: int
        Desired length in samples. The clip is trimmed or looped to match.
    """

    import torchaudio  # local import to keep CLI help lightweight

    files = list((musan_root / category).rglob("*.wav"))
    if not files:
        raise RuntimeError(f"No audio files found in {musan_root / category}")
    noise_path = random.choice(files)
    noise, noise_sr = torchaudio.load(noise_path)
    if noise.ndim > 1:
        noise = noise.mean(dim=0)
    if noise_sr != sr:
        noise = torchaudio.functional.resample(noise, noise_sr, sr)
    if noise.shape[-1] < length:
        reps = (length + noise.shape[-1] - 1) // noise.shape[-1]
        noise = noise.repeat(reps)[:length]
    else:
        noise = noise[:length]
    return noise


def select_babblers(speakers: list, idx: int, num_babble: int):
    """Select babble speaker directories excluding the current speaker.

    Parameters
    ----------
    speakers: list
        All available speaker directories.
    idx: int
        Index of the current target speaker in ``speakers``.
    num_babble: int
        Number of babble speakers to select.

    Raises
    ------
    ValueError
        If ``num_babble`` exceeds the number of other speakers.

    Returns
    -------
    list
        List of directories corresponding to babble speakers.
    """

    other_speakers = [s for i, s in enumerate(speakers) if i != idx]
    if num_babble > len(other_speakers):
        raise ValueError(
            f"Requested {num_babble} babble voices but only "
            f"{len(other_speakers)} other speakers available"
        )
    return other_speakers[:num_babble]


def load_sep_model(model_name: str, device):
    """Instantiate a separation model by name."""
    if model_name == "dprnn":
        from asteroid.models import DPRNNTasNet

        model = DPRNNTasNet.from_pretrained(
            "julien-c/DPRNNTasNet-ks16_WHAM_sepclean"
        ).to(device)
    elif model_name == "convtasnet":
        from asteroid.models import ConvTasNet

        model = ConvTasNet.from_pretrained(
            "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k"
        ).to(device)
    elif model_name == "demucs":
        try:
            from demucs.pretrained import get_model
        except ModuleNotFoundError:
            import subprocess, sys

            subprocess.check_call([sys.executable, "-m", "pip", "install", "demucs"])
            from demucs.pretrained import get_model

        model = get_model(name="htdemucs_speech_ft").to(device)
    else:
        raise ValueError(f"Unknown separation model: {model_name}")

    if hasattr(model, "eval"):
        model.eval()
    return model


def main() -> None:
    args = parse_args()

    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    # Determine device
    device = get_device()

    # Determine evaluation combinations

    sep_models = [m.strip().lower() for m in args.sep_models.split(",")]
    if args.musan_dir:
        if args.snr_db is not None:
            combos = [(args.snr_db, 0, sep_models[0])]
        elif args.snr_list:
            snr_values = [float(s) for s in args.snr_list.split(",")]
            if len(sep_models) not in (1, len(snr_values)):
                raise ValueError("sep_models must have length 1 or match snr_list length")
            models = sep_models * len(snr_values) if len(sep_models) == 1 else sep_models
            combos = [(snr, 0, model) for snr, model in zip(snr_values, models)]
        else:
            raise ValueError("Provide --snr_db or --snr_list when using --musan_dir")
    elif args.snr_db is not None and args.num_babble_voices is not None:
        combos = [(args.snr_db, args.num_babble_voices, sep_models[0])]
    elif args.snr_list and args.babble_list:
        snr_values = [float(s) for s in args.snr_list.split(",")]
        babble_values = [int(b) for b in args.babble_list.split(",")]
        if len(snr_values) != len(babble_values):
            raise ValueError("snr_list and babble_list must have the same length")
        if len(sep_models) == 1:
            models = sep_models * len(snr_values)
        elif len(sep_models) == len(snr_values):
            models = sep_models
        else:
            raise ValueError(
                "sep_models must have length 1 or match snr_list/babble_list length",
            )
        combos = list(zip(snr_values, babble_values, models))
    else:
        raise ValueError(
            "Specify either --snr_db and --num_babble_voices or --snr_list and --babble_list",
        )

    # Gather speakers and pick one at random
    all_speakers = [p for p in args.voices_dir.iterdir() if p.is_dir()]
    rng = random.Random(args.seed)
    rng.shuffle(all_speakers)
    if args.eval_text:
        candidates = [s for s in all_speakers if (s / "target.txt").exists()]
        if not candidates:
            raise RuntimeError("No speakers with target.txt available for text evaluation")
        chosen_speaker = candidates[0]
        speakers = [chosen_speaker] + [s for s in all_speakers if s != chosen_speaker]
    else:
        chosen_speaker = all_speakers[0]
        speakers = [chosen_speaker] + [s for s in all_speakers[1:]]

    timestamp_dir = (
        args.out_dir
        / datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
    )  # millisecond precision
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    results = []

    spk_model = EncDecSpeakerLabelModel.from_pretrained(
        "ecapa_tdnn",
    ).to(device)
    spk_model.eval()

    if args.eval_text:
        asr_bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        asr_model = asr_bundle.get_model().to(device)
        asr_model.eval()
        asr_labels = asr_bundle.get_labels()

        def transcribe_text(wav, sr):
            """Transcribe waveform using the loaded ASR model."""
            if sr != asr_bundle.sample_rate:
                wav = torchaudio.functional.resample(
                    wav, sr, asr_bundle.sample_rate
                )
                sr = asr_bundle.sample_rate
            with torch.no_grad():
                emissions, _ = asr_model(wav.unsqueeze(0).to(device))
            tokens = torch.argmax(emissions[0], dim=-1)
            tokens = torch.unique_consecutive(tokens)
            transcript = "".join(
                asr_labels[t] for t in tokens if asr_labels[t] != "-"
            )
            return transcript.replace("|", " ").lower().strip()
    else:
        asr_bundle = None
        asr_model = None
        asr_labels = None

    loaded_models: dict[str, object] = {}

    for run_idx, (snr_db, num_babble, model_name) in enumerate(combos):
        if model_name not in loaded_models:
            loaded_models[model_name] = load_sep_model(model_name, device)
        sep_model = loaded_models[model_name]

        enroll_wav, sr = load_audio(chosen_speaker / "enroll.wav")
        target_wav, sr = load_audio(chosen_speaker / "target.wav", sr)

        if args.musan_dir:
            noise = sample_musan_noise(
                args.musan_dir, args.musan_category, sr, target_wav.shape[-1]
            )
        else:
            babbler_dirs = select_babblers(speakers, 0, num_babble)
            babble_wavs = [load_audio(b / "target.wav", sr)[0] for b in babbler_dirs]
            noise = compose_babble(babble_wavs, target_wav.shape[-1])

        mixture = mix_at_snr(target_wav, noise, snr_db)
        peak = mixture.abs().max().item()
        if peak > 1.0:
            mixture = mixture / peak * 0.9  # apply headroom

        audio_duration = mixture.shape[-1] / sr

        start = time.time()
        if model_name == "demucs":
            from demucs.apply import apply_model

            with torch.no_grad():
                # Demucs expects a stereo input. The current mixture is mono,
                # so duplicate it across two channels before passing it to the
                # model to avoid a channel mismatch error.
                mix = torch.stack([mixture, mixture]).unsqueeze(0).to(device)
                est_sources = apply_model(sep_model, mix, device=device)[0]
                est_sources = est_sources.mean(dim=1)
        else:
            with torch.no_grad():
                est_sources = sep_model(mixture.unsqueeze(0).to(device)).squeeze(0)
        processing_time = time.time() - start
        est_sources = est_sources.cpu()

        enroll_emb = ecapa_embedding(spk_model, enroll_wav, device)
        est_embs = [ecapa_embedding(spk_model, src, device) for src in est_sources]
        scores = [
            torch.nn.functional.cosine_similarity(enroll_emb, emb, dim=0).item()
            for emb in est_embs
        ]
        chosen_idx = int(torch.argmax(torch.tensor(scores)))
        tse_result = est_sources[chosen_idx]

        rtf = processing_time / audio_duration if audio_duration > 0 else float("inf")
        if rtf > 0.5:
            print(
                f"High RTF detected for {chosen_speaker.name} model={model_name} "
                f"snr={snr_db} babble={num_babble} RTF={rtf:.3f}"
            )
            keep = input("Keep this run despite high RTF? [y/N]: ").strip().lower()
            if keep not in {"y", "yes"}:
                print("Run skipped due to high RTF.")
                continue

        si_sdr = compute_si_sdr(tse_result, target_wav).item()

        out_dir = timestamp_dir / f"run_{run_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)

        torchaudio.save(out_dir / "mixture.wav", mixture.unsqueeze(0), sr)
        torchaudio.save(out_dir / "tse_result.wav", tse_result.unsqueeze(0), sr)
        torchaudio.save(out_dir / "noise.wav", noise.unsqueeze(0), sr)

        def save_spectrogram_png(wav, path):
            import matplotlib.pyplot as plt

            plt.figure(figsize=(2, 1))
            plt.specgram(wav.numpy(), Fs=sr, NFFT=512, noverlap=256, cmap="magma")
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(path, dpi=100, bbox_inches="tight", pad_inches=0)
            plt.close()

        save_spectrogram_png(mixture, out_dir / "mixture.png")
        save_spectrogram_png(tse_result, out_dir / "tse_result.png")

        gt_text = ""
        mixture_text = ""
        post_text = ""
        mix_ratio = float("nan")
        post_ratio = float("nan")
        if args.eval_text:
            gt_path = chosen_speaker / "target.txt"
            gt_text = gt_path.read_text().strip().lower()
            shutil.copy(gt_path, out_dir / "target.txt")

            mixture_text = transcribe_text(mixture, sr)
            post_text = transcribe_text(tse_result, sr)
            mix_ratio = 1.0 - word_error_rate(gt_text, mixture_text)
            post_ratio = 1.0 - word_error_rate(gt_text, post_text)

        mixture_path = str(out_dir / "mixture.wav")
        result_path = str(out_dir / "tse_result.wav")
        similarity0 = scores[0] if len(scores) > 0 else float("nan")
        similarity1 = scores[1] if len(scores) > 1 else float("nan")

        results.append(
            [
                chosen_speaker.name,
                model_name,
                snr_db,
                num_babble,
                similarity0,
                similarity1,
                chosen_idx,
                si_sdr,
                rtf,
                mixture_path,
                result_path,
                mixture_text,
                post_text,
                gt_text,
                mix_ratio,
                post_ratio,
            ]
        )
        if args.eval_text:
            print(
                f"speaker={chosen_speaker.name} model={model_name} snr={snr_db} babble={num_babble} "
                f"si_sdr={si_sdr:.2f} rtf={rtf:.3f} mix/gt_acc={mix_ratio:.3f} post/gt_acc={post_ratio:.3f}"
            )
            print(
                f"mixture/GT accuracy: {mix_ratio:.3f}, post-processing/GT accuracy: {post_ratio:.3f}"
            )
        else:
            print(
                f"speaker={chosen_speaker.name} model={model_name} snr={snr_db} babble={num_babble} "
                f"si_sdr={si_sdr:.2f} rtf={rtf:.3f}"
            )

    # Write results
    if results:
        csv_path = timestamp_dir / "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "speaker_id",
                    "model",
                    "snr_db",
                    "num_babble",
                    "similarity0",
                    "similarity1",
                    "picked",
                    "si_sdr_target",
                    "rtf",
                    "mixture_path",
                    "result_path",
                    "mixture_text",
                    "post_text",
                    "gt_text",
                    "mixture_gt_ratio",
                    "post_gt_ratio",
                ]
            )
            writer.writerows(results)


if __name__ == "__main__":
    main()
