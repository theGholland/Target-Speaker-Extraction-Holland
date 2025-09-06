import argparse
from pathlib import Path
import random

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local voice bank using the LibriSpeech dataset."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/voices"),
        help="Directory where the prepared voices will be stored.",
    )
    parser.add_argument(
        "--num_speakers",
        type=int,
        required=True,
        help="Number of speakers to include in the voice bank.",
    )
    parser.add_argument(
        "--min_enroll_sec",
        type=float,
        default=10.0,
        help="Minimum duration in seconds for the enrollment audio.",
    )
    parser.add_argument(
        "--max_enroll_sec",
        type=float,
        default=20.0,
        help="Maximum duration in seconds for the enrollment audio.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=1.0,
        help=(
            "Fraction of the dataset to use. Useful for quick experiments. "
            "Value must be in (0, 1]."
        ),
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="train-clean-100",
        help="LibriSpeech subset to download/use.",
    )
    args = parser.parse_args()
    if not (0 < args.limit <= 1):
        raise ValueError(f"--limit must be in (0, 1], got {args.limit}")
    return args


def build_voice_bank(args: argparse.Namespace) -> None:
    # Import heavy dependencies lazily so that --help works without them.
    import torch
    import torchaudio

    Path("data/LibriSpeech").mkdir(parents=True, exist_ok=True)

    dataset = torchaudio.datasets.LIBRISPEECH(
        root="data/LibriSpeech", url=args.subset, download=True
    )

    total_items = len(dataset)
    usable_items = int(total_items * args.limit)
    indices = list(range(usable_items))

    by_speaker = {}
    for i in indices:
        waveform, sample_rate, _transcript, speaker_id, _chapter_id, _utterance_id = dataset[i]
        by_speaker.setdefault(str(speaker_id), []).append((waveform, sample_rate))

    chosen_speakers = random.sample(list(by_speaker.keys()), k=min(args.num_speakers, len(by_speaker)))

    args.out.mkdir(parents=True, exist_ok=True)

    for speaker_id in chosen_speakers:
        utterances = by_speaker[speaker_id]
        if len(utterances) < 2:
            # Need at least two utterances for enroll and target
            continue

        random.shuffle(utterances)
        target_waveform, sr = utterances.pop()
        enroll_waveforms = []
        total_seconds = 0.0

        while utterances and total_seconds < args.min_enroll_sec:
            w, sr = utterances.pop()
            enroll_waveforms.append(w)
            total_seconds += w.shape[1] / sr

        if total_seconds < args.min_enroll_sec:
            # Not enough audio for enrollment
            continue

        enroll_wave = torch.cat(enroll_waveforms, dim=1)
        max_samples = int(args.max_enroll_sec * sr)
        if enroll_wave.shape[1] > max_samples:
            enroll_wave = enroll_wave[:, :max_samples]

        speaker_dir = args.out / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        torchaudio.save(speaker_dir / "enroll.wav", enroll_wave, sr)
        torchaudio.save(speaker_dir / "target.wav", target_waveform, sr)


if __name__ == "__main__":
    args = parse_args()
    build_voice_bank(args)
