"""Listen-mode domain models: captured audio, utterance events, transcripts,
per-turn metrics and the per-session consent set.

User audio stays in memory: :class:`AudioChunk` carries samples, never a path.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
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


class ConsentState:
    """Thread-safe set of users who opted in for this listen session.

    Read on the receive thread (:meth:`allowed`, per packet) and written from
    the event loop (commands, member leave). ``bot_user_id`` and ``None`` are
    never allowed. Clear hooks run synchronously on opt-out and :meth:`clear`,
    after the user is already removed (so no new audio is admitted) and
    outside the lock (so a hook may take its own buffer lock).
    """

    def __init__(self, bot_user_id: int | None = None) -> None:
        self._bot_user_id = bot_user_id
        self._lock = threading.Lock()
        self._opted_in: set[int] = set()
        self._clear_hooks: list[Callable[[int], None]] = []

    def add_clear_hook(self, hook: Callable[[int], None]) -> None:
        """Register ``hook(user_id)``, called when a user's audio must be dropped."""
        with self._lock:
            self._clear_hooks.append(hook)

    def opt_in(self, user_id: int) -> bool:
        """Opt ``user_id`` in. Returns False for the bot itself."""
        if user_id == self._bot_user_id:
            return False
        with self._lock:
            self._opted_in.add(user_id)
        return True

    def opt_out(self, user_id: int) -> bool:
        """Opt ``user_id`` out and run the clear hooks before returning.

        Returns whether the user had been opted in. Hooks run either way, so
        a stray buffer can never outlive an explicit ``off``.
        """
        with self._lock:
            was_in = user_id in self._opted_in
            self._opted_in.discard(user_id)
            hooks = list(self._clear_hooks)
        for hook in hooks:
            hook(user_id)
        return was_in

    def allowed(self, user_id: int | None) -> bool:
        """True only for an opted-in user who is not the bot."""
        if user_id is None or user_id == self._bot_user_id:
            return False
        with self._lock:
            return user_id in self._opted_in

    def opted_in(self) -> frozenset[int]:
        """Snapshot of the opted-in user ids."""
        with self._lock:
            return frozenset(self._opted_in)

    def clear(self) -> None:
        """Opt everyone out (session stop) and run the hooks for each user."""
        with self._lock:
            users = sorted(self._opted_in)
            self._opted_in.clear()
            hooks = list(self._clear_hooks)
        for user_id in users:
            for hook in hooks:
                hook(user_id)
