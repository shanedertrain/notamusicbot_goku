"""ListenIngress: consent before decode, 16 kHz mono output, bounded ring, opt-out race."""

from __future__ import annotations

import threading

import numpy as np
from _listen_ingress_fakes import DecoderLog

from notamusicbot.domain.listen import ConsentState
from notamusicbot.listen.ingress import ListenIngress

A, B, BOT = 111, 222, 999
PKT_A, PKT_B, PKT_BOT = b"1000", b"2000", b"3000"


def make(ring_s: float = 30.0, on_pcm=None, opted=(A,)):
    consent = ConsentState(bot_user_id=BOT)
    for u in opted:
        consent.opt_in(u)
    log = DecoderLog()
    ingress = ListenIngress(consent, BOT, log.factory(), ring_s, on_pcm)
    return consent, log, ingress


def test_only_opted_in_user_is_decoded():
    _, log, ingress = make()
    for _ in range(100):
        ingress.accept_packet(A, PKT_A)
        ingress.accept_packet(B, PKT_B)
        ingress.accept_packet(BOT, PKT_BOT)
    assert log.calls == 100
    assert set(log.payloads) == {PKT_A}
    assert log.created == 1
    # 100 packets x 960 frames at 48 kHz -> 320 samples each at 16 kHz.
    assert ingress.pending_samples(A) == 32000
    assert ingress.pending_samples(B) == 0
    assert ingress.pending_samples(BOT) == 0


def test_none_user_dropped_before_decode():
    _, log, ingress = make()
    ingress.accept_packet(None, PKT_A)
    assert log.calls == 0


def test_opt_out_clears_immediately():
    consent, log, ingress = make()
    for _ in range(10):
        ingress.accept_packet(A, PKT_A)
    assert ingress.pending_samples(A) == 3200
    consent.opt_out(A)
    assert ingress.pending_samples(A) == 0
    ingress.accept_packet(A, PKT_A)
    assert ingress.pending_samples(A) == 0
    assert log.calls == 10


def test_ring_buffer_caps_and_keeps_newest():
    _, _, ingress = make(ring_s=1.0)
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
    _, _, ingress = make()
    ingress.accept_packet(A, b"3000")
    out = ingress.drain(A)
    assert out.shape == (320,)
    assert np.allclose(out, 3000 / 32768)
    assert ingress.drain(A).shape == (0,)


def test_non_multiple_of_three_keeps_remainder():
    consent = ConsentState(bot_user_id=BOT)
    consent.opt_in(A)

    class Odd:
        def decode(self, data: bytes) -> bytes:
            return np.full((4, 2), 600, dtype="<i2").tobytes()

    ingress = ListenIngress(consent, BOT, Odd, 30.0)
    for _ in range(3):
        ingress.accept_packet(A, b"x")
    # 12 mono samples at 48 kHz -> exactly 4 at 16 kHz, none lost to rounding.
    assert ingress.pending_samples(A) == 4


def test_on_pcm_called_per_packet():
    seen: list[tuple[int, int]] = []
    _, _, ingress = make(on_pcm=lambda u, s: seen.append((u, len(s))))
    ingress.accept_packet(A, PKT_A)
    ingress.accept_packet(B, PKT_B)
    assert seen == [(A, 320)]


def test_decoder_error_drops_packet():
    consent = ConsentState(bot_user_id=BOT)
    consent.opt_in(A)

    class Bad:
        def decode(self, data: bytes) -> bytes:
            raise ValueError("corrupt")

    ingress = ListenIngress(consent, BOT, Bad, 30.0)
    ingress.accept_packet(A, b"x")
    assert ingress.pending_samples(A) == 0


def test_remove_user_forgets_buffer():
    _, _, ingress = make()
    ingress.accept_packet(A, PKT_A)
    ingress.remove_user(A)
    assert ingress.pending_samples(A) == 0


def test_consent_clear_empties_all_users():
    consent, _, ingress = make(opted=(A, B))
    ingress.accept_packet(A, PKT_A)
    ingress.accept_packet(B, PKT_B)
    consent.clear()
    assert ingress.pending_samples(A) == 0
    assert ingress.pending_samples(B) == 0


def test_opt_out_between_first_check_and_append_drops_packet():
    """Deterministic form of the race: opt_out (and its clear hook) completes
    after the first ``allowed`` check but before the append. Only the re-check
    under the buffer lock stops this packet; the threaded test below rarely
    hits this window (it passed 50/50 with the re-check removed)."""
    consent = ConsentState(bot_user_id=BOT)
    consent.opt_in(A)
    calls = 0

    class OptsOutMidDecode:
        def decode(self, data: bytes) -> bytes:
            nonlocal calls
            calls += 1
            # Same thread, RLock: the clear hook runs to completion right here.
            consent.opt_out(A)
            return np.full((960, 2), 3000, dtype="<i2").tobytes()

    seen: list[int] = []
    ingress = ListenIngress(
        consent, BOT, OptsOutMidDecode, 30.0, on_pcm=lambda u, s: seen.append(u)
    )
    ingress.accept_packet(A, b"x")
    assert calls == 1
    assert ingress.pending_samples(A) == 0
    assert seen == []


def test_concurrent_accept_during_opt_out_leaves_nothing():
    for _ in range(20):
        consent, _, ingress = make()
        started = threading.Event()
        stop = threading.Event()

        def hammer(ingress=ingress, started=started, stop=stop):
            while not stop.is_set():
                ingress.accept_packet(A, PKT_A)
                started.set()

        t = threading.Thread(target=hammer)
        t.start()
        assert started.wait(5)
        consent.opt_out(A)
        # Stays 0 while the thread keeps hammering after opt_out returned.
        for _ in range(200):
            assert ingress.pending_samples(A) == 0
        stop.set()
        t.join(5)
        assert not t.is_alive()
        assert ingress.pending_samples(A) == 0
