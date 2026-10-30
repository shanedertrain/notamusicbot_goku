"""Listen-mode domain models: captured audio, utterance events, transcripts
and per-turn metrics.

User audio stays in memory: :class:`AudioChunk` carries samples, never a path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Everything downstream of ingress (VAD, endpointing, ASR) runs at 16 kHz mono.
LISTEN_SAMPLE_RATE_HZ = 16000


@dataclass(frozen=True, slots=True, eq=False)
class AudioChunk:
    """In-memory float32 mono audio (16 kHz unless stated otherwise)."""

    samples: np.ndarray
    sample_rate_hz: int = LISTEN_SAMPLE_RATE_HZ

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate_hz


@dataclass(frozen=True, slots=True, eq=False)
class UtteranceEvent:
    """One endpointed utterance from one user.

    ``t_start`` / ``t_eos`` are monotonic-clock seconds of speech onset and
    end of speech; ``audio`` includes the pre-roll. ``t_emit`` is when the
    endpointer produced the event (same clock; diagnostics only, ``None`` when
    not recorded).
    """

    user_id: int
    audio: AudioChunk
    t_start: float
    t_eos: float
    t_emit: float | None = None

    @property
    def duration_s(self) -> float:
        return self.t_eos - self.t_start


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    """Speech onset for one user (drives barge-in before the utterance ends)."""

    user_id: int
    t_start: float


@dataclass(frozen=True, slots=True)
class Transcript:
    """Recognized text for one :class:`AudioChunk`."""

    text: str
    language: str | None = None
    elapsed_s: float = 0.0


@dataclass(frozen=True, slots=True)
class ListenTurnMetrics:
    """Stage deltas for one listen turn, logged as one INFO line."""

    turn_id: str
    user_id: int
    eos_to_asr_s: float = 0.0
    asr_to_llm_s: float = 0.0
    llm_to_tts1_s: float = 0.0
    tts1_to_audio_s: float = 0.0
    total_s: float = 0.0
