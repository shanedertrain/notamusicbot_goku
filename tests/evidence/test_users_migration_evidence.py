"""Live legacy migration evidence (review Round-4 section 7).

Runs the real migration against the actual legacy ``input/users.json``
(29 OBAMA / 1 OBAMA_TRANS) shipped in this checkout. Skips gracefully when
the legacy file is absent. Moved here from ``tests/unit/test_user_migration.py``
— this is a live-asset proof, not a portable unit test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from notamusicbot.tools.migrate_users import migrate_users_to_profiles

pytestmark = [pytest.mark.integration, pytest.mark.evidence]

REPO = Path(__file__).resolve().parents[2]


def test_real_legacy_file_migrates_identity() -> None:
    """Against the real input/users.json: identity survives, no goku fallback."""
    source = REPO / "input" / "users.json"
    if not source.is_file():  # legacy file not shipped in every checkout
        pytest.skip("input/users.json not present")
    dest = REPO / "config" / "users.yaml"
    entries = migrate_users_to_profiles(source, dest, REPO / "config" / "voices")
    counts: dict[str, int] = {}
    for entry in entries.values():
        counts[entry["voice_profile"]] = counts.get(entry["voice_profile"], 0) + 1
    assert counts == {"obama": 29, "obama_trans": 1}
    assert not any(e["voice_profile"] == "goku" for e in entries.values())
    # OBAMA_TRANS resolves through an alias (weights shared with Obama) and
    # must be recorded as such, not silently conflated.
    trans = [e for e in entries.values() if e["voice_profile"] == "obama_trans"]
    assert len(trans) == 1
    assert trans[0]["legacy_model_name"] == "OBAMA_TRANS"
    assert trans[0]["voice_profile_source"] == "legacy_alias"
