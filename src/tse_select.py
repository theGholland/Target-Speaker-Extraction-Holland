import argparse
import os
import time
import random
from typing import Optional, Iterable

import torch
import torchaudio
from pathlib import Path

import shutil

from device_utils import get_device
from eval_tse_on_voices import demucs_openvino_separate


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
    if noise_power == 0:
        return target
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


def sample_musan_noise(
    musan_root: Path, category: str, sr: int, length: int
) -> torch.Tensor:
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


def ecapa_embedding(model, wav: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute a speaker embedding using NeMo ECAPA."""
    with torch.no_grad():
        wav = wav.unsqueeze(0).to(device)
        length = torch.tensor([wav.shape[-1]], device=device)
        _, emb = model.forward(input_signal=wav, input_signal_length=length)
        emb = torch.nn.functional.normalize(emb.squeeze(0), dim=-1)
    return emb


def compute_si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """Compute SI-SDR between estimate and reference."""
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
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost,
            )
    return d[-1][-1] / max(len(ref_words), 1)


def main():
    parser = argparse.ArgumentParser(description="Target speaker extraction by selection")
    parser.add_argument("--target", type=str, required=True, help="Path to clean target/enrollment audio")
    parser.add_argument("--noise", type=str, help="Path to noise/interferer audio")
    parser.add_argument("--create_noise", action="store_true", help="Create babble noise from other voices")
    parser.add_argument("--musan_dir", type=str, help="Path to MUSAN dataset root for noise")
    parser.add_argument(
        "--musan_category",
        type=str,
        default="noise",
        choices=["noise", "music"],
        help="MUSAN subset to sample from when --musan_dir is provided",
    )
    parser.add_argument("--snr_db", type=float, default=0.0, help="SNR for mixing target and noise")
    parser.add_argument(
        "--model",
        type=str,
        choices=["dprnn", "convtasnet", "demucs"],
        default="dprnn",
        help="Separation model to use",
    )
    parser.add_argument(
        "--gt_text",
        type=str,
        help="Path to ground truth transcript (defaults to target with .txt)",
    )
    parser.add_argument(
        "--eval_text",
        action="store_true",
        help="Evaluate ASR accuracy when ground truth transcript is available",
    )
    args = parser.parse_args()

    if sum([bool(args.noise), bool(args.create_noise), bool(args.musan_dir)]) > 1:
        parser.error("--noise, --create_noise, and --musan_dir are mutually exclusive")

    # Determine device
    device = get_device()

    # Load enrollment/clean target
    target_wav, sr = load_audio(args.target)

    # If noise provided or to be created, create mixture
    if args.musan_dir:
        noise_wav = sample_musan_noise(Path(args.musan_dir), args.musan_category, sr, target_wav.shape[-1])
        mixture = mix_at_snr(target_wav, noise_wav, args.snr_db)
    elif args.create_noise:
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

    # Ensure the reference target matches the mixture length
    target_wav = target_wav[..., : mixture.shape[-1]]
    peak = mixture.abs().max().item()
    if peak > 1.0:
        mixture = mixture / peak * 0.9

    audio_duration = mixture.shape[-1] / sr

    # Load separation model
    if args.model == "dprnn":
        from asteroid.models import DPRNNTasNet

        model = DPRNNTasNet.from_pretrained(
            "julien-c/DPRNNTasNet-ks16_WHAM_sepclean"
        ).to(device)
        model.eval()
        start = time.time()
        with torch.no_grad():
            est_sources = model(mixture.unsqueeze(0).to(device))
    elif args.model == "convtasnet":
        from asteroid.models import ConvTasNet

        model = ConvTasNet.from_pretrained(
            "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k"
        ).to(device)
        model.eval()
        start = time.time()
        with torch.no_grad():
            est_sources = model(mixture.unsqueeze(0).to(device))
    else:  # demucs
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
        if "GPU" not in core.available_devices:
            raise RuntimeError("OpenVINO GPU device is required but not available")
        ov_model = core.read_model(xml_path)
        model = core.compile_model(ov_model, "GPU")
        start = time.time()
        est_sources = demucs_openvino_separate(model, mixture, sr)

    processing_time = time.time() - start
    est_sources = est_sources.squeeze(0).cpu()

    # Compute embeddings using NeMo ECAPA
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    spk_model = EncDecSpeakerLabelModel.from_pretrained("speakerverification_ecapa").to(device)
    spk_model.eval()

    enroll_emb = ecapa_embedding(spk_model, target_wav, device)
    est_embs = [ecapa_embedding(spk_model, src, device) for src in est_sources]
    scores = [torch.nn.functional.cosine_similarity(enroll_emb, emb, dim=0).item() for emb in est_embs]
    chosen_idx = int(torch.argmax(torch.tensor(scores)))
    tse_result = est_sources[chosen_idx]

    # Write output WAVs
    out_dir = os.path.dirname(args.target)
    torchaudio.save(
        os.path.join(out_dir, "sep_source0.wav"),
        est_sources[0].unsqueeze(0),
        sr,
    )
    torchaudio.save(
        os.path.join(out_dir, "sep_source1.wav"),
        est_sources[1].unsqueeze(0),
        sr,
    )
    torchaudio.save(
        os.path.join(out_dir, "tse_result.wav"),
        tse_result.unsqueeze(0),
        sr,
    )
    rtf = processing_time / audio_duration if audio_duration > 0 else float("inf")

    gt_path = Path(args.gt_text) if args.gt_text else Path(args.target).with_suffix(".txt")
    if args.eval_text:
        if gt_path.exists():
            asr_bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            asr_model = asr_bundle.get_model().to(device)
            asr_model.eval()

            def transcribe(wav, sr):
                if sr != asr_bundle.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, asr_bundle.sample_rate)
                with torch.no_grad():
                    emissions, _ = asr_model(wav.unsqueeze(0).to(device))
                return asr_bundle.decode(emissions.argmax(dim=-1))[0].lower().strip()

            gt_text = gt_path.read_text().strip().lower()
            mixture_text = transcribe(mixture, sr)
            result_text = transcribe(tse_result, sr)
            mix_ratio = 1.0 - word_error_rate(gt_text, mixture_text)
            post_ratio = 1.0 - word_error_rate(gt_text, result_text)

            out_gt = Path(out_dir) / "target.txt"
            if not out_gt.exists() or gt_path.resolve() != out_gt.resolve():
                shutil.copy(gt_path, out_gt)
            Path(out_dir, "mixture.txt").write_text(mixture_text + "\n")
            Path(out_dir, "tse_result.txt").write_text(result_text + "\n")

            print(f"mixture/GT accuracy: {mix_ratio:.3f}")
            print(f"post-processing/GT accuracy: {post_ratio:.3f}")
        else:
            print(f"Ground truth text not found at {gt_path}; skipping ASR evaluation.")
    else:
        print("Text evaluation disabled.")

    print(f"Similarity scores: {scores}")
    print(f"Chosen source: {chosen_idx}")
    if args.target:
        si_sdr_value = compute_si_sdr(tse_result, target_wav)
        print(f"SI-SDR: {si_sdr_value:.2f} dB")
    print(f"RTF: {rtf:.3f}")
    print(
        f"Processing time: {processing_time:.2f} s for {audio_duration:.2f} s of audio"
    )


if __name__ == "__main__":
    main()
