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
python src/prepare_voices.py --num_speakers 10 --subset test-clean
```

The `--subset` argument accepts any of the official LibriSpeech subsets:

- `dev-clean`
- `dev-other`
- `test-clean`
- `test-other`
- `train-clean-100`
- `train-clean-360`
- `train-other-500`

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

To test robustness to non-speech noise, provide a path to the [MUSAN dataset](https://www.openslr.org/17)
and optionally choose a category such as `noise` or `music`:

```bash
python src/tse_select.py --target path/to/clean.wav \
       --musan_dir /path/to/musan --musan_category music --snr_db 5
```

## Batch Evaluation of Target Speaker Extraction

`eval_tse_on_voices.py` mixes each target speaker with either uniform babble noise from
other speakers or a random clip from the MUSAN dataset and then evaluates the extraction
quality. Each
invocation creates a timestamped directory under `out_eval` named with the
current datetime down to milliseconds. A `results.csv` summarising the runs is
written inside this directory along with one subfolder per run containing the
audio waveforms and plots. Columns in `results.csv` include speaker,
separation model, SNR, babble count, SI-SDR and RTF.

Evaluate a single condition:

```bash
python src/eval_tse_on_voices.py --snr_db 0 --num_babble_voices 3
```

Sweep over multiple SNRs and babble counts:

```bash
python src/eval_tse_on_voices.py --snr_list "-5,0,5" --babble_list "1,2,3"
```

Use MUSAN noise instead of babble voices:

```bash
python src/eval_tse_on_voices.py --snr_db 0 --musan_dir /path/to/musan --musan_category noise
```

Or sweep SNRs with MUSAN noise:

```bash
python src/eval_tse_on_voices.py --snr_list "-5,0,5" --musan_dir /path/to/musan
```

Sweep over different separation models:

```bash
python src/eval_tse_on_voices.py --snr_db 0 --num_babble_voices 3 --sep_models "dprnn,convtasnet,demucs"
```

The `demucs` separation model uses the OpenVINO export from the `Intel/demucs-openvino`
repository. Ensure the `openvino` package is installed and note that the script downloads
the `htdemucs_v4` variant on first use.

## Makefile Targets

Common workflows are wrapped in a Makefile.  Variables may be overridden on the command
line or via the environment.

Build a voice bank:

```bash
make prepare NUM_SPEAKERS=10 SUBSET=test-clean
```

Run a single evaluation:

```bash
make eval SNR_DB=0 NUM_BABBLE=3
```

Sweep over SNR values:

```bash
make sweep_snr SNR_LIST="-5,0,5" NUM_BABBLE=3
```

Sweep over babble voice counts:

```bash
make sweep_babble SNR_DB=0 BABBLE_LIST="1,2,3"
```

Run a full SNR × babble grid:

```bash
make grid SNR_LIST="-5,0,5" BABBLE_LIST="1,2,3"
```

Remove evaluation outputs:

```bash
make clean
```

## Repository Layout

```
README.md            # This file
requirements.txt     # Python dependencies
src/                 # Project scripts
├── prepare_voices.py
├── eval_tse_on_voices.py
└── tse_select.py
tests/               # Unit tests
```

## Tests

The `tests` directory contains unit tests that validate the project's core utilities and
command-line scripts. They cover device selection, audio mixing behavior, mixture length
alignment, amplitude scaling, validation of `prepare_voices.py` arguments, and the
selection of babbler speakers. Running these tests helps catch regressions in the data
preparation and evaluation workflows.

## License

This project is provided for demonstration purposes and has no explicit license.
