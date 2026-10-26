"""Listen-turn metrics: one INFO log line per turn, and the parser that reads it back.

Pure string formatting; no discord.py, no I/O. ``tools/listen_bench.py`` uses
:func:`parse_turn_line` to compute P50/P90 over real logs.
"""

from __future__ import annotations

import re

from notamusicbot.domain.listen import ListenTurnMetrics

NA = "n/a"

# (field on ListenTurnMetrics, key in the log line), in pipeline order.
STAGES: tuple[tuple[str, str], ...] = (
    ("eos_to_asr_s", "eos->asr"),
    ("asr_to_llm_s", "asr->llm"),
    ("llm_to_tts1_s", "llm->tts1"),
    ("tts1_to_audio_s", "tts1->audio"),
)
STAGE_KEYS: tuple[str, ...] = tuple(key for _, key in STAGES)
FIELD_KEYS: tuple[str, ...] = (*STAGE_KEYS, "total")

_VALUE = r"(\d+(?:\.\d+)?|n/a)"
_TURN_RE = re.compile(
    r"listen turn guild=(?P<guild>\S+) user=(?P<user>\S+)"
    + "".join(rf" {re.escape(key)}={_VALUE}" for key in FIELD_KEYS)
)
_GROUP_NAMES = ("eos_asr", "asr_llm", "llm_tts1", "tts1_audio", "total")


def _fmt(seconds: float | None) -> str:
    # A stage that did not happen stays at its 0.0 default (or None): "n/a".
    if seconds is None or seconds <= 0.0:
        return NA
    return f"{seconds:.2f}"


def format_turn_metrics(m: ListenTurnMetrics, guild_id: int | str, user_id: int | str) -> str:
    """Format ``m`` as the one INFO line logged per listen turn.

    Stages at 0.0 did not happen and print ``n/a``. ``total`` uses
    ``m.total_s`` when set, else the sum of the stages that happened.
    """
    values = [getattr(m, field) for field, _ in STAGES]
    total = m.total_s if m.total_s > 0.0 else sum(v for v in values if v and v > 0.0)
    parts = [f"{key}={_fmt(v)}" for key, v in zip(STAGE_KEYS, values, strict=True)]
    parts.append(f"total={_fmt(total)}")
    return f"listen turn guild={guild_id} user={user_id} " + " ".join(parts)


def parse_turn_line(line: str) -> dict | None:
    """Parse a line from :func:`format_turn_metrics`; ``None`` if it is not one.

    Returns ``{"guild": str, "user": str, "eos->asr": float | None, ...,
    "total": float | None}``; ``n/a`` values become ``None``. The line may carry
    a log prefix (timestamp, logger name).
    """
    match = _TURN_RE.search(line)
    if match is None:
        return None
    out: dict = {"guild": match.group("guild"), "user": match.group("user")}
    for key, raw in zip(FIELD_KEYS, match.groups()[2:], strict=True):
        out[key] = None if raw == NA else float(raw)
    return out
