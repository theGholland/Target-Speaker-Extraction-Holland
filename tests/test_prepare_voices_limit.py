import subprocess
import sys
import pytest

@pytest.mark.parametrize("limit", ["0", "-0.5", "1.5"])
def test_prepare_voices_invalid_limit(limit):
    result = subprocess.run(
        [sys.executable, "-m", "src.prepare_voices", "--num_speakers", "1", "--limit", limit],
        capture_output=True,
    )
    assert result.returncode != 0
    assert b"--limit must be in (0, 1]" in result.stderr


@pytest.mark.parametrize("subset", ["train-clean-10", "unknown", "dev-clean-100"])
def test_prepare_voices_invalid_subset(subset):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.prepare_voices",
            "--num_speakers",
            "1",
            "--subset",
            subset,
        ],
        capture_output=True,
    )
    assert result.returncode != 0
    assert b"Invalid --subset" in result.stderr
