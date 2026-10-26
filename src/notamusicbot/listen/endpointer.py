"""Per-user endpointing: VAD probabilities in, utterance events out.

:class:`Endpointer` is a plain synchronous state machine (testable with a fake
clock). :class:`EndpointerThread` runs it on one dedicated thread, off the
event loop and off the GPU.
"""

from __future__ import annotations

import inspect
import logging
import math
import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from notamusicbot.domain.listen import (
    LISTEN_SAMPLE_RATE_HZ,
    AudioChunk,
    SpeechStarted,
    UtteranceEvent,
)
from notamusicbot.speech.vad.base import FRAME_MS, FRAME_SAMPLES, VoiceActivityDetector

if TYPE_CHECKING:
    from notamusicbot.config import ListenVadSettings

logger = logging.getLogger(__name__)

_FRAME_S = FRAME_MS / 1000.0

ListenEvent = SpeechStarted | UtteranceEvent


def _frames(ms: float) -> int:
    """Whole frames needed to cover ``ms`` (rounded up)."""
    return math.ceil(ms / FRAME_MS)


@dataclass(slots=True)
class _UserState:
    vad: VoiceActivityDetector
    remainder: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    pre_roll: deque[np.ndarray] = field(default_factory=deque)
    # Voiced frames seen before min_speech is reached: (frame, frame_end_time).
    pending: list[tuple[np.ndarray, float]] = field(default_factory=list)
    # Active utterance: audio up to the last voiced frame, plus a silent tail
    # that is kept only if speech resumes (so the final audio is trimmed).
    audio: list[np.ndarray] = field(default_factory=list)
    tail: list[np.ndarray] = field(default_factory=list)
    in_speech: bool = False
    t_start: float = 0.0
    t_last_voiced: float = 0.0
    last_input: float = 0.0
    n_frames: int = 0  # frames in the active utterance, for the length cap


class IngressSource(Protocol):
    """What :class:`EndpointerThread` needs from ``listen.ingress``."""

    def drain(self, user_id: int) -> np.ndarray:
        """Return and remove the user's pending float32 16 kHz samples."""
        ...


class Endpointer:
    """Turns per-user 16 kHz float32 audio into SpeechStarted / UtteranceEvent.

    ``vad`` is either a factory (one detector per user, required for stateful
    detectors such as silero) or a single detector instance shared by all
    users (fine for the stateless energy VAD). Timestamps come from ``clock``
    (monotonic seconds): a frame's end time is the clock reading when it is
    fed, so a batch is back-dated by one frame period per earlier frame.
    The emitted ``audio`` is pre-roll + speech, trimmed to the last voiced
    frame (the trailing silence that triggered the end is not included).
    """

    def __init__(
        self,
        vad: VoiceActivityDetector | Callable[[], VoiceActivityDetector],
        settings: ListenVadSettings,
        clock: Callable[[], float],
        emit: Callable[[ListenEvent], None],
    ) -> None:
        self._vad_source = vad
        self._settings = settings
        self._clock = clock
        self._emit = emit
        self._users: dict[int, _UserState] = {}
        self._pre_roll_frames = _frames(settings.pre_roll_ms)
        self._min_speech_frames = _frames(settings.min_speech_ms)
        self._end_silence_frames = _frames(settings.end_silence_ms)
        self._max_frames = _frames(settings.max_utterance_s * 1000.0)

    def _state(self, user_id: int) -> _UserState:
        state = self._users.get(user_id)
        if state is None:
            src = self._vad_source
            is_factory = inspect.isclass(src) or not hasattr(src, "process")
            vad: VoiceActivityDetector = src() if is_factory else src  # type: ignore[operator,assignment]
            vad.reset()
            state = _UserState(vad=vad)
            state.pre_roll = deque(maxlen=max(self._pre_roll_frames, 1))
            state.last_input = self._clock()
            self._users[user_id] = state
        return state

    def feed(self, user_id: int, samples: np.ndarray) -> None:
        """Consume any amount of audio for ``user_id`` (split into 32 ms frames)."""
        state = self._state(user_id)
        now = self._clock()
        state.last_input = now
        data = np.concatenate([state.remainder, samples.astype(np.float32, copy=False)])
        n = len(data) // FRAME_SAMPLES
        state.remainder = data[n * FRAME_SAMPLES :].copy()
        for i in range(n):
            frame = data[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES].copy()
            self._frame(user_id, state, frame, now - (n - 1 - i) * _FRAME_S)

    def _frame(self, user_id: int, st: _UserState, frame: np.ndarray, t_end: float) -> None:
        voiced = st.vad.process(frame) >= self._settings.threshold
        if st.in_speech:
            st.n_frames += 1
            if voiced:
                st.audio.extend(st.tail)
                st.tail.clear()
                st.audio.append(frame)
                st.t_last_voiced = t_end
            else:
                st.tail.append(frame)
            if len(st.tail) >= self._end_silence_frames or st.n_frames >= self._max_frames:
                self._finish(user_id, st)
            return
        if voiced:
            st.pending.append((frame, t_end))
            if len(st.pending) >= self._min_speech_frames:
                self._start(user_id, st)
            return
        # Silence: a too-short blip goes back into the pre-roll ring.
        for f, _ in st.pending:
            st.pre_roll.append(f)
        st.pending.clear()
        st.pre_roll.append(frame)

    def _start(self, user_id: int, st: _UserState) -> None:
        st.in_speech = True
        st.t_start = st.pending[0][1] - _FRAME_S
        st.t_last_voiced = st.pending[-1][1]
        st.audio = list(st.pre_roll) + [f for f, _ in st.pending]
        st.n_frames = len(st.pending)
        st.pre_roll.clear()
        st.pending.clear()
        st.tail = []
        self._emit(SpeechStarted(user_id=user_id, t_start=st.t_start))

    def _finish(self, user_id: int, st: _UserState) -> None:
        audio = np.concatenate(st.audio) if st.audio else np.zeros(0, dtype=np.float32)
        event = UtteranceEvent(
            user_id=user_id,
            audio=AudioChunk(samples=audio, sample_rate_hz=LISTEN_SAMPLE_RATE_HZ),
            t_start=st.t_start,
            t_eos=st.t_last_voiced,
            t_emit=self._clock(),
        )
        st.in_speech = False
        st.audio = []
        st.tail = []
        st.n_frames = 0
        st.pre_roll.clear()
        st.vad.reset()
        self._emit(event)

    def tick(self) -> None:
        """End utterances whose user stopped sending audio for ``end_silence_ms``.

        Discord may send nothing at all during silence, so the frame-driven
        end condition alone could hang an utterance open.
        """
        now = self._clock()
        limit = self._settings.end_silence_ms / 1000.0
        for user_id, st in list(self._users.items()):
            if st.in_speech and now - st.last_input >= limit:
                self._finish(user_id, st)

    def discard(self, user_id: int) -> None:
        """Forget a user (opt-out): drop buffered audio and emit nothing."""
        self._users.pop(user_id, None)


class EndpointerThread:
    """Polls ``ingress.drain`` for opted-in users on one dedicated thread.

    ``on_event`` is called ON THIS THREAD. The caller must hop to the asyncio
    loop itself, e.g. ``lambda ev: loop.call_soon_threadsafe(handle, ev)``;
    never touch loop objects directly from it. ``users`` returns the current
    opted-in ids (e.g. ``consent.opted_in``); users that drop out of it are
    discarded from the endpointer so no buffered audio outlives opt-out.
    """

    def __init__(
        self,
        ingress: IngressSource,
        endpointer: Endpointer,
        users: Callable[[], Iterable[int]],
        poll_interval_s: float = 0.01,
    ) -> None:
        self._ingress = ingress
        self._endpointer = endpointer
        self._users = users
        self._poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._known: set[int] = set()

    def poll_once(self) -> int:
        """Drain every opted-in user once; returns the number of samples fed."""
        current = set(self._users())
        for gone in self._known - current:
            self._endpointer.discard(gone)
        self._known = current
        fed = 0
        for user_id in sorted(current):
            samples = self._ingress.drain(user_id)
            if len(samples):
                self._endpointer.feed(user_id, samples)
                fed += len(samples)
        self._endpointer.tick()
        return fed

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                fed = self.poll_once()
            except Exception:
                logger.exception("endpointer poll failed")
                fed = 0
            if not fed:
                self._stop.wait(self._poll_interval_s)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="listen-endpointer", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            self._thread = None
