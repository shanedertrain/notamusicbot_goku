"""Adaptive chunk sizing for streamed speech.

Chunk N+1 only has to be synthesized before chunk N stops playing. Synthesis
runs faster than playback (RTF well below 1), so once something is playing the
next chunk can be much larger than the first one, and bigger chunks carry one
steady delivery instead of a fresh sampling roll per sentence.

:class:`ChunkPacer` turns that into a word budget for the next chunk from three
live measurements: how much audio is still queued ahead of the playhead, how
fast synthesis has recently run (RTF, wall time per second of audio, GPU
contention included) and how fast the voice speaks (words per second).
"""

from __future__ import annotations

# Starting estimates until real chunks have been measured (conservative: the
# measured clone RTF is about 0.29 and the voice speaks about 2.6 words/s).
INITIAL_RTF = 0.35
INITIAL_WORDS_PER_S = 2.6
# Exponential-average weight of the newest measurement.
EMA_ALPHA = 0.3
# Synthesis must beat playback by this factor, to absorb jitter.
SAFETY_FACTOR = 1.3
# Audio that must remain ahead of the playhead when the next chunk lands.
SAFETY_LEAD_S = 0.3
# An opener shorter than this has no lead to hide the next chunk's synthesis behind,
# so it is merged with the following sentence(s) instead of going out alone.
MIN_OPENER_S = 1.5
# Longest the opener waits for the LLM to deliver the sentence it is merging with.
OPENER_WAIT_S = 0.8
# Hard ceiling on one merged chunk, however fast the pipeline is.
MAX_CHUNK_WORDS = 70
# Keep the estimates inside sane bounds so one odd sample cannot run away.
_MIN_RTF, _MAX_RTF = 0.05, 3.0
_MIN_WPS, _MAX_WPS = 1.0, 6.0


class ChunkPacer:
    """Word budget for the next chunk; the averages persist across turns."""

    def __init__(
        self, *, max_chunk_words: int = MAX_CHUNK_WORDS, min_opener_s: float = MIN_OPENER_S
    ) -> None:
        self._max_words = max(1, max_chunk_words)
        self._min_opener_s = max(0.0, min_opener_s)
        self.rtf = INITIAL_RTF
        self.words_per_s = INITIAL_WORDS_PER_S
        self._delivered_s = 0.0
        self._play_t0: float | None = None

    def begin_turn(self) -> None:
        """Forget the previous reply's queue; keep the speed estimates."""
        self._delivered_s = 0.0
        self._play_t0 = None

    def opener_target_words(self) -> int:
        """Words the first chunk needs so it plays long enough; 0 once audio exists."""
        if self._delivered_s > 0.0:
            return 0
        if self._min_opener_s <= 0.0:
            return 0
        return min(self._max_words, int(self._min_opener_s * self.words_per_s) + 1)

    def note_delivered(self, audio_s: float) -> None:
        """A synthesized chunk of ``audio_s`` seconds joined the playback queue."""
        self._delivered_s += max(0.0, audio_s)

    def note_playback_started(self, now: float) -> None:
        """The first chunk began playing at clock time ``now``."""
        if self._play_t0 is None:
            self._play_t0 = now

    def note_synthesis(self, words: int, audio_s: float, took_s: float) -> None:
        """Fold one finished chunk into the speed estimates."""
        if audio_s <= 0.0 or words <= 0:
            return
        rtf = min(_MAX_RTF, max(_MIN_RTF, took_s / audio_s))
        wps = min(_MAX_WPS, max(_MIN_WPS, words / audio_s))
        self.rtf += EMA_ALPHA * (rtf - self.rtf)
        self.words_per_s += EMA_ALPHA * (wps - self.words_per_s)

    def lead_s(self, now: float) -> float:
        """Seconds of audio still queued ahead of the playhead at ``now``."""
        if self._play_t0 is None:
            return self._delivered_s
        return max(0.0, self._delivered_s - max(0.0, now - self._play_t0))

    def budget_words(self, now: float) -> int:
        """Most words the next chunk may hold and still finish before the lead runs out.

        Zero when nothing is queued yet (the first chunk is never padded: it sets
        the time to first audio).
        """
        usable_s = self.lead_s(now) - SAFETY_LEAD_S
        if usable_s <= 0.0:
            return 0
        audio_s = usable_s / (self.rtf * SAFETY_FACTOR)
        return min(self._max_words, int(audio_s * self.words_per_s))
