"""Counting fake opus decoder for the ingress tests."""

from __future__ import annotations

import threading

import numpy as np

FRAME_48K = 960  # 20 ms at 48 kHz per channel


def stereo_bytes(value: int = 3000, frames: int = FRAME_48K) -> bytes:
    """Constant 48 kHz stereo int16 PCM (both channels equal ``value``)."""
    return np.full((frames, 2), value, dtype="<i2").tobytes()


class FakeOpusDecoder:
    """Decoder whose calls are counted on a shared :class:`DecoderLog`."""

    def __init__(self, log: DecoderLog) -> None:
        self._log = log

    def decode(self, data: bytes) -> bytes:
        self._log.record(data)
        # The packet payload encodes the sample value: b"<int>".
        return stereo_bytes(int(data.decode()) if data else 0)


class DecoderLog:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.payloads: list[bytes] = []
        self.created = 0

    def record(self, data: bytes) -> None:
        with self._lock:
            self.payloads.append(data)

    @property
    def calls(self) -> int:
        with self._lock:
            return len(self.payloads)

    def factory(self):
        def make() -> FakeOpusDecoder:
            with self._lock:
                self.created += 1
            return FakeOpusDecoder(self)

        return make
