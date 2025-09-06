#!/usr/bin/env python3
"""Batch evaluation of target speaker extraction with babble sweeps."""

import argparse
import csv
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable


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


def demucs_openvino_separate(sep_model, wav, sr):
    """Run Demucs OpenVINO model on a mono waveform.

    The OpenVINO export of Demucs expects a fixed-size complex STFT
    representation of stereo audio with shape ``[1, 4, 2048, 336]`` where the
    second dimension stacks real and imaginary parts for each stereo channel.

    This helper performs the required pre- and post-processing so callers can
    simply provide a 1D waveform tensor. It mirrors the preprocessing used when
    exporting the model and returns time-domain estimates for each separated
    source as a tensor of shape ``(num_sources, num_samples)``.
    """

    import torch
    import torchaudio

    orig_sr = sr
    orig_len = wav.shape[-1]

    target_sr = 44100
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    n_fft = 4096
    hop = 1024
    n_frames = 336

    # Demucs OpenVINO model was exported with 336 STFT frames. With
    # ``center=True`` below, ``torch.stft`` adds ``n_fft // 2`` padding on
    # both sides, yielding one extra frame unless the waveform length is
    # ``(n_frames - 1) * hop``.
    seg_length = (n_frames - 1) * hop
    if wav.shape[-1] < seg_length:
        wav = torch.nn.functional.pad(wav, (0, seg_length - wav.shape[-1]))
    else:
        wav = wav[:seg_length]

    # create stereo by duplicating the mono track
    stereo = wav.repeat(2, 1)

    window = torch.hann_window(n_fft)
    stft = torch.stft(
        stereo,
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        return_complex=True,
        center=True,
    )
    # Drop the Nyquist bin to match expected 2048 frequency bins
    stft = stft[:, :-1, :]
    spec = torch.cat([stft.real, stft.imag], dim=0).unsqueeze(0)

    ov_output = sep_model([spec.cpu().numpy()])[sep_model.output(0)]
    ov_output = torch.from_numpy(ov_output).squeeze(0)

    # The model outputs real and imaginary parts concatenated on the channel
    # dimension. Reconstruct the complex STFT for each estimated source.
    half = ov_output.shape[0] // 2
    real, imag = ov_output[:half], ov_output[half:]
    complex_spec = torch.complex(real, imag)
    # Restore the dropped Nyquist frequency bin for iSTFT
    complex_spec = torch.cat(
        [complex_spec, torch.zeros_like(complex_spec[:, :1, :])], dim=1
    )

    est = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        length=seg_length,
    )

    est = est[..., : int(orig_len * target_sr / orig_sr)]
    if orig_sr != target_sr:
        est = torchaudio.functional.resample(est, target_sr, orig_sr)
    est = est[..., :orig_len]

    return est


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
        from huggingface_hub import hf_hub_download
        try:
            from openvino.runtime import Core
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Running the 'demucs' separation model requires the 'openvino' "
                "package. Install it with `pip install openvino`."
            ) from exc

        core = Core()
        repo_id = "Intel/demucs-openvino"
        variant = "htdemucs_v4"
        local_dir = "models/demucs_openvino"
        xml_path = hf_hub_download(
            repo_id=repo_id,
            filename="htdemucs_fwd.xml",
            subfolder=variant,
            local_dir=local_dir,
        )
        hf_hub_download(
            repo_id=repo_id,
            filename="htdemucs_fwd.bin",
            subfolder=variant,
            local_dir=local_dir,
        )
        ov_model = core.read_model(xml_path)
        model = core.compile_model(ov_model, "CPU")
    else:
        raise ValueError(f"Unknown separation model: {model_name}")

    if hasattr(model, "eval"):
        model.eval()
    return model


def main() -> None:
    args = parse_args()

    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    # Determine device
    device = get_device()

    # Determine evaluation combinations
    sep_models = [m.strip().lower() for m in args.sep_models.split(",")]
    if args.snr_db is not None and args.num_babble_voices is not None:
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
                "sep_models must have length 1 or match snr_list/babble_list length"
            )
        combos = list(zip(snr_values, babble_values, models))
    else:
        raise ValueError(
            "Specify either --snr_db and --num_babble_voices or --snr_list and --babble_list"
        )

    # Gather speakers and pick one at random
    speakers = [p for p in args.voices_dir.iterdir() if p.is_dir()]
    rng = random.Random(args.seed)
    rng.shuffle(speakers)
    chosen_speaker = speakers[0]
    speakers = [chosen_speaker] + [s for s in speakers[1:]]  # keep list for babblers

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

    loaded_models: dict[str, object] = {}

    for run_idx, (snr_db, num_babble, model_name) in enumerate(combos):
        if model_name not in loaded_models:
            loaded_models[model_name] = load_sep_model(model_name, device)
        sep_model = loaded_models[model_name]

        enroll_wav, sr = load_audio(chosen_speaker / "enroll.wav")
        target_wav, sr = load_audio(chosen_speaker / "target.wav", sr)

        # Select babble speakers deterministically after shuffle
        babbler_dirs = select_babblers(speakers, 0, num_babble)
        babble_wavs = [load_audio(b / "target.wav", sr)[0] for b in babbler_dirs]
        babble = compose_babble(babble_wavs, target_wav.shape[-1])

        mixture = mix_at_snr(target_wav, babble, snr_db)
        peak = mixture.abs().max().item()
        if peak > 1.0:
            mixture = mixture / peak * 0.9  # apply headroom

        audio_duration = mixture.shape[-1] / sr

        start = time.time()
        if model_name == "demucs":
            est_sources = demucs_openvino_separate(sep_model, mixture, sr)
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
                f"Rejected {chosen_speaker.name} model={model_name} snr={snr_db} "
                f"babble={num_babble} RTF={rtf:.3f}"
            )
            continue

        si_sdr = compute_si_sdr(tse_result, target_wav).item()

        out_dir = timestamp_dir / f"run_{run_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)

        import torchaudio
        torchaudio.save(out_dir / "mixture.wav", mixture.unsqueeze(0), sr)
        torchaudio.save(out_dir / "tse_result.wav", tse_result.unsqueeze(0), sr)
        torchaudio.save(out_dir / "babble.wav", babble.unsqueeze(0), sr)

        def save_waveform_png(wav, path):
            import matplotlib.pyplot as plt
            import numpy as np

            plt.figure(figsize=(2, 1))
            t = np.linspace(0, wav.shape[-1] / sr, wav.shape[-1])
            plt.plot(t, wav.numpy())
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(path, dpi=100)
            plt.close()

        save_waveform_png(mixture, out_dir / "mixture.png")
        save_waveform_png(tse_result, out_dir / "tse_result.png")

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
            ]
        )
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
                ]
            )
            writer.writerows(results)


if __name__ == "__main__":
    main()
