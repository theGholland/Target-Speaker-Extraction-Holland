from pathlib import Path
import sys

import pytest

# Add src directory to import path for direct access to script functions
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from eval_tse_on_voices import select_babblers


@pytest.mark.parametrize(
    "num_speakers, requested, expected",
    [
        (1, 1, 0),
        (2, 2, 1),
    ],
)
def test_insufficient_speaker_pool_warns(num_speakers, requested, expected, capsys):
    speakers = [Path(f"spk{i}") for i in range(num_speakers)]
    babblers = select_babblers(speakers, idx=0, num_babble=requested)
    captured = capsys.readouterr().out.lower()
    assert len(babblers) == expected
    assert "reducing num_babble" in captured
    assert speakers[0] not in babblers
    assert len(set(babblers)) == len(babblers)
