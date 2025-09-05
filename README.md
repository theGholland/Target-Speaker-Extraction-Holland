# Target-Speaker-Extraction-Holland

This repository contains demonstration scripts for target speaker extraction and evaluation.
It supports creating a local voice bank from LibriSpeech, running single-file target
speaker extraction, and performing batch evaluations with babble noise.

## Installation

The scripts require Python 3.10+ and an environment with either a CUDA GPU or Apple M‑series
GPU (MPS).  Installing in a virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preparing a Voice Bank

Use `prepare_voices.py` to build a directory of speaker voices from the LibriSpeech dataset.
The script downloads the specified subset if it is not already present and creates a voice
bank with enrollment and target audio for each speaker.

```bash
python src/prepare_voices.py --num_speakers 10 --subset train-clean-100
```

Resulting structure:

```
data/voices/
  ├── <speaker_id>/
  │   ├── enroll.wav
  │   └── target.wav
  └── ...
```

## Target Speaker Extraction on a Single File

`tse_select.py` separates a mixture into sources using a chosen separation model
(DPRNN, ConvTasNet, or Demucs) and selects the source most similar to the enrollment
speaker using NeMo ECAPA embeddings.

```bash
python src/tse_select.py --target path/to/clean.wav --noise path/to/noise.wav \
       --snr_db 0 --model dprnn
```

Separated sources (`sep_source0.wav`, `sep_source1.wav`) and the selected result
(`tse_result.wav`) are written alongside the target file.

## Batch Evaluation of Target Speaker Extraction

`eval_tse_on_voices.py` mixes each target speaker with uniform babble noise at specified
signal-to-noise ratios and evaluates the extraction quality.  Results are written to
`out_eval/results.csv` with columns for speaker, separation model, SNR, babble count,
SI-SDR, and RTF.

Evaluate a single condition:

```bash
python src/eval_tse_on_voices.py --snr_db 0 --num_babble_voices 3
```

Sweep over multiple SNRs and babble counts:

```bash
python src/eval_tse_on_voices.py --snr_list "-5,0,5" --babble_list "1,2,3"
```

Sweep over different separation models:

```bash
python src/eval_tse_on_voices.py --snr_db 0 --num_babble_voices 3 --sep_models "dprnn,convtasnet,demucs"
```

## Repository Layout

```
README.md            # This file
requirements.txt     # Python dependencies
src/                 # Project scripts
├── prepare_voices.py
├── eval_tse_on_voices.py
└── tse_select.py
```

## License

This project is provided for demonstration purposes and has no explicit license.
