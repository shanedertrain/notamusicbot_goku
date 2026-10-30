"""ListenIngress: None/bot dropped before decode, 16 kHz mono, bounded ring, known users."""

from __future__ import annotations

import threading

import numpy as np
from _listen_ingress_fakes import DecoderLog

from notamusicbot.listen.ingress import ListenIngress

A, B, BOT = 111, 222, 999
PKT_A, PKT_B, PKT_BOT = b"1000", b"2000", b"3000"


def make(ring_s: float = 30.0, on_pcm=None):
    log = DecoderLog()
    ingress = ListenIngress(BOT, log.factory(), ring_s, on_pcm)
    return log, ingress


def test_none_and_bot_dropped_before_decode():
    log, ingress = make()
    for _ in range(100):
        ingress.accept_packet(None, PKT_A)
        ingress.accept_packet(BOT, PKT_BOT)
    assert log.calls == 0
    assert log.created == 0  # no decoder was even built
    assert ingress.pending_samples(BOT) == 0
    assert ingress.known_user_ids() == frozenset()


def test_every_human_is_decoded():
    log, ingress = make()
    for _ in range(100):
        ingress.accept_packet(A, PKT_A)
        ingress.accept_packet(B, PKT_B)
    assert log.calls == 200
    assert set(log.payloads) == {PKT_A, PKT_B}
    assert log.created == 2  # one decoder per human, no consent gate in between
    # 100 packets x 960 frames at 48 kHz -> 320 samples each at 16 kHz.
    assert ingress.pending_samples(A) == 32000
    assert ingress.pending_samples(B) == 32000


def test_known_user_ids_tracks_buffered_users_and_remove_user_forgets():
    _, ingress = make()
    ingress.accept_packet(A, PKT_A)
    ingress.accept_packet(B, PKT_B)
    assert ingress.known_user_ids() == frozenset({A, B})
    ingress.remove_user(A)
    assert ingress.known_user_ids() == frozenset({B})
    assert ingress.pending_samples(A) == 0


def test_ring_buffer_caps_and_keeps_newest():
    _, ingress = make(ring_s=1.0)
    # 150 packets x 20 ms = 3 s; sample value ramps so the newest is identifiable.
    for i in range(150):
        ingress.accept_packet(A, str(i * 100).encode())
    assert ingress.pending_samples(A) == 16000
    out = ingress.drain(A)
    assert len(out) == 16000
    assert out.dtype == np.float32
    # Last packet value 149*100 -> 14900/32768; first retained packet is #100.
    assert np.isclose(out[-1], 14900 / 32768)
    assert np.isclose(out[0], 10000 / 32768)
    assert ingress.pending_samples(A) == 0


def test_downmix_values_and_drain():
    _, ingress = make()
    ingress.accept_packet(A, b"3000")
    out = ingress.drain(A)
    assert out.shape == (320,)
    assert np.allclose(out, 3000 / 32768)
    assert ingress.drain(A).shape == (0,)


def test_non_multiple_of_three_keeps_remainder():
    class Odd:
        def decode(self, data: bytes) -> bytes:
            return np.full((4, 2), 600, dtype="<i2").tobytes()

    ingress = ListenIngress(BOT, Odd, 30.0)
    for _ in range(3):
        ingress.accept_packet(A, b"x")
    # 12 mono samples at 48 kHz -> exactly 4 at 16 kHz, none lost to rounding.
    assert ingress.pending_samples(A) == 4


def test_on_pcm_called_per_human_packet_only():
    seen: list[tuple[int, int]] = []
    _, ingress = make(on_pcm=lambda u, s: seen.append((u, len(s))))
    ingress.accept_packet(A, PKT_A)
    ingress.accept_packet(B, PKT_B)
    ingress.accept_packet(None, PKT_A)
    ingress.accept_packet(BOT, PKT_BOT)
    assert seen == [(A, 320), (B, 320)]


def test_decoder_error_drops_packet():
    class Bad:
        def decode(self, data: bytes) -> bytes:
            raise ValueError("corrupt")

    ingress = ListenIngress(BOT, Bad, 30.0)
    ingress.accept_packet(A, b"x")
    assert ingress.pending_samples(A) == 0


def test_concurrent_accepts_for_different_users_are_all_buffered():
    _, ingress = make()
    users = (A, B, 333)

    def hammer(user_id: int) -> None:
        for _ in range(200):
            ingress.accept_packet(user_id, str(user_id).encode())

    threads = [threading.Thread(target=hammer, args=(user_id,)) for user_id in users]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    assert ingress.known_user_ids() == frozenset(users)
    assert [ingress.pending_samples(user_id) for user_id in users] == [200 * 320] * 3


def test_remove_user_forgets_buffer():
    _, ingress = make()
    ingress.accept_packet(A, PKT_A)
    ingress.remove_user(A)
    assert ingress.pending_samples(A) == 0
    assert A not in ingress.known_user_ids()
