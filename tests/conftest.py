"""Shared pytest configuration for the notamusicbot test suite."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

FIXTURES = Path(__file__).parent / "fixtures"
SINE_FIXTURE = FIXTURES / "sine_2s.wav"


@pytest.fixture(scope="session", autouse=True)
def _ensure_audio_fixtures() -> None:
    """Create the tracked-in-source-but-untracked-on-disk audio fixture.

    ``.wav`` files are not committed (see .gitignore), so a fresh checkout has
    no ``tests/fixtures/sine_2s.wav``; the profile/listen/benchmark tests only
    need a valid 2 s mono WAV at 24 kHz.
    """
    if SINE_FIXTURE.is_file():
        return
    FIXTURES.mkdir(parents=True, exist_ok=True)
    rate = 24000
    t = np.arange(rate * 2) / rate
    sf.write(SINE_FIXTURE, 0.3 * np.sin(2 * np.pi * 220.0 * t), rate, subtype="PCM_16")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: end-to-end tests over real assets")
    config.addinivalue_line("markers", "gpu: requires a CUDA GPU")
