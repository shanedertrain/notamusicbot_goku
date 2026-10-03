"""Review Round-3 P2: interstitial last-fire timing state is guild-scoped.

A cooldown fired in guild A must never suppress an interstitial in guild B.
"""

from __future__ import annotations

import random

from notamusicbot.content.interstitials import InterstitialManager, TopicSource


class _Settings:
    enabled = True
    probability = 1.0  # always fire when the cooldown allows
    min_interval_s = 60.0


def _manager() -> InterstitialManager:
    rng = random.Random(0)
    return InterstitialManager(_Settings(), TopicSource(None, None, rng=rng))


def test_two_guilds_do_not_share_cooldown() -> None:
    mgr = _manager()
    assert mgr.should_fire(guild_id=1, now_monotonic=100.0) is True
    # Guild 2 has never fired: its cooldown bucket is independent.
    assert mgr.should_fire(guild_id=2, now_monotonic=100.0) is True


def test_same_guild_respects_cooldown() -> None:
    mgr = _manager()
    assert mgr.should_fire(guild_id=7, now_monotonic=100.0) is True
    assert mgr.should_fire(guild_id=7, now_monotonic=100.0 + 30) is False
    assert mgr.should_fire(guild_id=7, now_monotonic=100.0 + 60.1) is True


def test_default_caller_uses_shared_global_bucket() -> None:
    mgr = _manager()
    assert mgr.should_fire(now_monotonic=10.0) is True
    # Default (no guild id) bucket is one bucket: cooldown applies within it.
    assert mgr.should_fire(now_monotonic=20.0) is False
    assert mgr.should_fire(now_monotonic=70.1) is True


def test_global_bucket_does_not_block_guilds() -> None:
    mgr = _manager()
    assert mgr.should_fire(now_monotonic=10.0) is True  # global bucket fires
    assert mgr.should_fire(guild_id=5, now_monotonic=10.0) is True  # guild unaffected
