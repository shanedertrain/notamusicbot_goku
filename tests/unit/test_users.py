"""Unit tests for user migration and lookup (dict and list shapes)."""

from __future__ import annotations

from pathlib import Path

import pytest

from notamusicbot.domain.users import (
    UserRegistry,
    load_users_yaml,
    migrate_users_file,
    parse_users,
    write_users_yaml,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_parse_dict_shape() -> None:
    import json

    raw = json.loads((FIXTURES / "users_dict.json").read_text())
    users = parse_users(raw)
    assert set(users) == {"517156036445077504", "358723048557248512"}
    assert users["517156036445077504"].real_name == "Cameron"
    assert users["517156036445077504"].legacy_model_name == "OBAMA"


def test_parse_list_shape() -> None:
    import json

    raw = json.loads((FIXTURES / "users_list.json").read_text())
    users = parse_users(raw)
    assert set(users) == {"123456789"}
    assert users["123456789"].screen_name == "listuser"


def test_parse_rejects_garbage() -> None:
    assert parse_users(None) == {}
    assert parse_users(42) == {}


def test_migration_round_trip_dict_shape(tmp_path: Path) -> None:
    source = tmp_path / "users.json"
    source.write_text((FIXTURES / "users_dict.json").read_text())
    dest = tmp_path / "config" / "users.yaml"
    users = migrate_users_file(source, dest)
    assert len(users) == 2
    reread = load_users_yaml(dest)
    assert reread == users


def test_migration_round_trip_list_shape(tmp_path: Path) -> None:
    source = tmp_path / "users.json"
    source.write_text((FIXTURES / "users_list.json").read_text())
    dest = tmp_path / "config" / "users.yaml"
    users = migrate_users_file(source, dest)
    assert len(users) == 1
    assert load_users_yaml(dest) == users


def test_migration_does_not_modify_source(tmp_path: Path) -> None:
    source = tmp_path / "users.json"
    original = (FIXTURES / "users_dict.json").read_text()
    source.write_text(original)
    migrate_users_file(source, tmp_path / "users.yaml")
    assert source.read_text() == original


def test_migration_empty_source_raises(tmp_path: Path) -> None:
    source = tmp_path / "empty.json"
    source.write_text("{}")
    with pytest.raises(ValueError, match="No valid user entries"):
        migrate_users_file(source, tmp_path / "users.yaml")


def test_registry_lookup() -> None:
    users = parse_users({"1": {"screen_name": "a", "voice_profile": "goku"}})
    registry = UserRegistry(users)
    assert registry.lookup("1") is not None
    assert registry.voice_profile_id("1") == "goku"
    assert registry.lookup("999") is None
    assert registry.voice_profile_id("999") is None


def test_write_users_yaml_creates_parent_dirs(tmp_path: Path) -> None:
    users = parse_users({"7": {"screen_name": "x"}})
    dest = tmp_path / "deep" / "nested" / "users.yaml"
    write_users_yaml(users, dest)
    assert load_users_yaml(dest)["7"].screen_name == "x"
