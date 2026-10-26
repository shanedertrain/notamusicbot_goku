"""Local-only guard (plan section 13): the listen system never touches the network.

To cover a new listen module, add its directory/file to ``SCAN_TARGETS``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from notamusicbot.config import ListenLlmSettings, ListenSettings
from notamusicbot.listen.offline import assert_offline, enforce_offline_models

SRC = Path(__file__).resolve().parents[2] / "src" / "notamusicbot"

# The single list of scanned paths (directories are walked recursively).
SCAN_TARGETS: tuple[Path, ...] = (
    SRC / "listen",
    SRC / "speech" / "asr",
    SRC / "speech" / "vad",
    SRC / "domain" / "listen.py",
    SRC / "content" / "spark_chat.py",
    SRC / "content" / "fallback.py",
)

# Plan section 17: the ONE named exception. spark_chat.py may import httpx, and
# only because it calls the loopback guard (asserted below).
SPARK_CHAT = SRC / "content" / "spark_chat.py"
HTTPX_EXCEPTIONS = {SPARK_CHAT: {"httpx"}}
LOOPBACK_GUARD = "assert_loopback_url("

# Network-capable modules; matched on the dotted import path or any prefix of it.
BANNED = frozenset(
    {
        "httpx",
        "requests",
        "aiohttp",
        "openai",
        "anthropic",
        "urllib.request",
        "urllib3",
        "http.client",
        "socket",
        "websockets",
        "grpc",
    }
)
# Model loaders that can download; allowed only in speech/asr/*.py, and only
# next to an explicit offline marker.
LOADERS = frozenset({"huggingface_hub", "onnx_asr", "transformers"})
OFFLINE_MARKERS = ("local_files_only=True", "enforce_offline_models", "assert_offline")
ASR_DIR = SRC / "speech" / "asr"


def _scan_files() -> list[Path]:
    files: list[Path] = []
    for target in SCAN_TARGETS:
        if not target.exists():
            pytest.fail(f"scan target missing: {target}")
        if target.is_dir():
            found = sorted(target.rglob("*.py"))
            if not found:
                pytest.fail(f"scan target has no .py files: {target}")
            files.extend(found)
        else:
            files.append(target)
    return files


def _imports(tree: ast.AST) -> set[str]:
    """Every dotted name imported, including ``from a import b`` as ``a.b``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _hits(imports: set[str], wanted: frozenset[str]) -> set[str]:
    return {
        name for name in imports if any(name == bad or name.startswith(bad + ".") for bad in wanted)
    }


def test_scan_targets_exist() -> None:
    assert len(_scan_files()) >= 8


def test_no_network_imports_in_listen_code() -> None:
    violations: list[str] = []
    for path in _scan_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = _imports(tree)
        allowed = HTTPX_EXCEPTIONS.get(path, set())
        for name in sorted(_hits(imports, BANNED) - allowed):
            violations.append(f"{path.relative_to(SRC)}: imports {name}")
        loaders = _hits(imports, LOADERS)
        if not loaders:
            continue
        if path.parent != ASR_DIR:
            violations.append(
                f"{path.relative_to(SRC)}: loader import {sorted(loaders)} outside speech/asr"
            )
        elif not any(marker in path.read_text(encoding="utf-8") for marker in OFFLINE_MARKERS):
            violations.append(
                f"{path.relative_to(SRC)}: loader {sorted(loaders)} without offline marker"
            )
    assert not violations, "\n".join(violations)


def test_only_spark_chat_imports_httpx_and_it_has_the_loopback_guard() -> None:
    importers = [
        path
        for path in _scan_files()
        if "httpx" in _hits(_imports(ast.parse(path.read_text(encoding="utf-8"))), BANNED)
    ]
    assert importers == [SPARK_CHAT]
    source = SPARK_CHAT.read_text(encoding="utf-8")
    assert source.count(LOOPBACK_GUARD) >= 1
    # The guard is invoked from the constructor and at call time.
    assert "_clean_base(settings.base_url)" in source
    assert "def _guard(self)" in source
    assert source.count("self._guard()") >= 2


@pytest.mark.parametrize("name", sorted(BANNED))
def test_scanner_flags_each_banned_module(name: str) -> None:
    for source in (f"import {name}", f"from {name} import x", f"import {name}.sub"):
        assert _hits(_imports(ast.parse(source)), BANNED), source
    parent, _, child = name.partition(".")
    if child:
        assert _hits(_imports(ast.parse(f"from {parent} import {child}")), BANNED)


def test_scanner_loader_rules() -> None:
    assert _hits(_imports(ast.parse("import onnx_asr")), LOADERS) == {"onnx_asr"}
    assert not _hits(_imports(ast.parse("import json")), BANNED | LOADERS)


def test_enforce_offline_forces_variables() -> None:
    environ = {"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}
    changed = enforce_offline_models(environ)
    assert environ == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    assert changed == environ
    assert enforce_offline_models(environ) == {}


def test_assert_offline() -> None:
    with pytest.raises(RuntimeError):
        assert_offline({})
    with pytest.raises(RuntimeError):
        assert_offline({"HF_HUB_OFFLINE": "0"})
    assert_offline({"HF_HUB_OFFLINE": "1"})


def _field_names(model: type, seen: frozenset[type] = frozenset()) -> set[str]:
    """Dotted field paths of a pydantic model, recursing into nested models."""
    names: set[str] = set()
    for field, info in model.model_fields.items():  # type: ignore[attr-defined]
        names.add(field)
        ann = info.annotation
        if isinstance(ann, type) and hasattr(ann, "model_fields") and ann not in seen:
            names.update(f"{field}.{sub}" for sub in _field_names(ann, seen | {ann}))
    return names


def test_listen_settings_llm_is_loopback_only_and_has_no_secret_fields() -> None:
    names = _field_names(ListenSettings)
    assert "asr.model_id" in names  # the walker does recurse into nested models
    offenders = {n for n in names if n.split(".")[-1] in {"api_key", "api_base"}}
    assert not offenders, offenders
    # base_url exists only as listen.llm.base_url, guarded by the loopback validator.
    assert {n for n in names if n.split(".")[-1] == "base_url"} == {"llm.base_url"}
    with pytest.raises(ValueError, match="not loopback"):
        ListenLlmSettings(base_url="http://10.0.0.5:8000/v1")
    assert ListenSettings().llm.base_url == "http://127.0.0.1:8000/v1"
