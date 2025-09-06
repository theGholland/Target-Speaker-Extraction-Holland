import sys
from pathlib import Path
import pytest

# Ensure the src directory is on the path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from eval_tse_on_voices import select_babblers


def test_select_babblers_skips_current_and_no_wrap():
    speakers = ['a', 'b', 'c']
    # For idx=1 and num_babble=2, should return remaining speakers in order
    assert select_babblers(speakers, 1, 2) == ['a', 'c']


def test_select_babblers_raises_when_num_babble_exceeds_speakers():
    speakers = ['a', 'b', 'c']
    with pytest.raises(ValueError, match='babble voices'):
        select_babblers(speakers, 0, len(speakers))
