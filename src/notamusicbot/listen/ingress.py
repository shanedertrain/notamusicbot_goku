"""Voice ingress: the only place user audio is decoded.

``accept_packet`` is called from the receive library's reader thread. It drops
``None`` and the bot's own id (self-hearing) BEFORE opus decode, then downmixes
48 kHz stereo int16 to 16 kHz mono float32 and appends it to a bounded per-user
ring. Audio stays in memory; nothing here touches disk.

The endpointer polls :meth:`ListenIngress.known_user_ids`; users the session
forgets here (a member leaving the channel) are discarded downstream as well,
so no buffered audio outlives their stay.

This module imports no discord; the opus decoder is injected.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from typing import Protocol

import numpy as np

from notamusicbot.domain.listen import LISTEN_SAMPLE_RATE_HZ

# Discord delivers 48 kHz stereo; 48 kHz -> 16 kHz is an exact factor of 3.
_SOURCE_CHANNELS = 2
_DECIMATION = 3


class OpusDecoder(Protocol):
    """Subset of ``discord.opus.Decoder`` used here (one instance per user)."""

    def decode(self, data: bytes) -> bytes:
        """Return 48 kHz stereo int16 little-endian PCM for one opus packet."""
        ...


class _UserBuffer:
    """One user's decoder, resample remainder and bounded sample ring."""

    def __init__(self) -> None:
        # Reentrant: ``on_pcm`` runs under it and may call back into the ingress.
        self.lock = threading.RLock()
        self.decoder: OpusDecoder | None = None
        self.chunks: deque[np.ndarray] = deque()
        self.count = 0
        # Mono 48 kHz samples left over when a packet is not a multiple of 3.
        self.remainder = np.zeros(0, dtype=np.float32)

    def clear(self) -> None:
        self.chunks.clear()
        self.count = 0
        self.remainder = np.zeros(0, dtype=np.float32)
        # Fresh decoder next time: drop opus state from before the clear.
        self.decoder = None


class ListenIngress:
    """Thread-safe decoder and per-user 16 kHz ring buffer."""

    def __init__(
        self,
        bot_user_id: int,
        decoder_factory: Callable[[], OpusDecoder],
        ring_buffer_s: float,
        on_pcm: Callable[[int, np.ndarray], None] | None = None,
    ) -> None:
        self._bot_user_id = bot_user_id
        self._decoder_factory = decoder_factory
        self._max_samples = int(ring_buffer_s * LISTEN_SAMPLE_RATE_HZ)
        self._on_pcm = on_pcm
        self._users_lock = threading.Lock()
        self._users: dict[int, _UserBuffer] = {}

    def accept_packet(self, user_id: int | None, opus: bytes) -> None:
        """Filter, decode and buffer one packet. Safe from any thread; never raises.

        Only ``None`` and the bot's own id are dropped (self-hearing).

        ``on_pcm(user_id, samples)`` runs on the calling thread under the
        user's buffer lock, so it must be quick and must not block.
        """
        if user_id is None or user_id == self._bot_user_id:
            return
        buf = self._buffer(user_id)
        with buf.lock:
            try:
                if buf.decoder is None:
                    buf.decoder = self._decoder_factory()
                pcm = buf.decoder.decode(opus)
            except Exception:  # corrupt packet: drop it; no per-packet logging
                return
            samples = self._to_16k_mono(buf, pcm)
            if len(samples) == 0:
                return
            buf.chunks.append(samples)
            buf.count += len(samples)
            self._trim(buf)
            if self._on_pcm is not None:
                self._on_pcm(user_id, samples)

    def known_user_ids(self) -> frozenset[int]:
        """Snapshot of the user ids that currently have a buffer here."""
        with self._users_lock:
            return frozenset(self._users)

    def pending_samples(self, user_id: int) -> int:
        """Number of buffered 16 kHz samples for ``user_id``."""
        buf = self._peek(user_id)
        if buf is None:
            return 0
        with buf.lock:
            return buf.count

    def drain(self, user_id: int) -> np.ndarray:
        """Take and return all buffered samples (16 kHz mono float32)."""
        buf = self._peek(user_id)
        if buf is None:
            return np.zeros(0, dtype=np.float32)
        with buf.lock:
            if not buf.chunks:
                return np.zeros(0, dtype=np.float32)
            out = np.concatenate(list(buf.chunks))
            buf.chunks.clear()
            buf.count = 0
            return out

    def remove_user(self, user_id: int) -> None:
        """Forget ``user_id`` entirely: buffer, decoder and resample state."""
        with self._users_lock:
            buf = self._users.pop(user_id, None)
        if buf is not None:
            with buf.lock:
                buf.clear()

    def _buffer(self, user_id: int) -> _UserBuffer:
        with self._users_lock:
            buf = self._users.get(user_id)
            if buf is None:
                buf = self._users[user_id] = _UserBuffer()
            return buf

    def _peek(self, user_id: int) -> _UserBuffer | None:
        with self._users_lock:
            return self._users.get(user_id)

    @staticmethod
    def _to_16k_mono(buf: _UserBuffer, pcm: bytes) -> np.ndarray:
        """48 kHz stereo int16 bytes -> 16 kHz mono float32 (mean of each 3-sample group)."""
        usable = len(pcm) - len(pcm) % (2 * _SOURCE_CHANNELS)
        if usable <= 0:
            return np.zeros(0, dtype=np.float32)
        stereo = np.frombuffer(pcm[:usable], dtype="<i2").reshape(-1, _SOURCE_CHANNELS)
        mono = stereo.astype(np.float32).mean(axis=1) / 32768.0
        if len(buf.remainder):
            mono = np.concatenate((buf.remainder, mono))
        whole = len(mono) - len(mono) % _DECIMATION
        buf.remainder = mono[whole:].copy()
        return mono[:whole].reshape(-1, _DECIMATION).mean(axis=1).astype(np.float32)

    def _trim(self, buf: _UserBuffer) -> None:
        """Drop the oldest samples beyond the ring capacity."""
        excess = buf.count - self._max_samples
        while excess > 0 and buf.chunks:
            head = buf.chunks[0]
            if len(head) <= excess:
                buf.chunks.popleft()
                buf.count -= len(head)
                excess -= len(head)
            else:
                buf.chunks[0] = head[excess:]
                buf.count -= excess
                excess = 0
