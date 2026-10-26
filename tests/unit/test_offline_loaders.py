"""Offline model loaders and chat system prompt (P10 finding 5). No real models."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from notamusicbot.content import llm as llm_mod
from notamusicbot.content import prompts
from notamusicbot.content.base import ContentKind, ContentRequest
from notamusicbot.content.llm import LocalQwenLlmProvider, resolve_local_model_path
from notamusicbot.speech.tts.qwen_custom_voice import QwenCustomVoiceSynthesizer

HUB_ID = "Qwen/Qwen3-4B-Instruct-2507"
SNAP = "/cache/snapshots/abc123"


class _FakeModel:
    device = "cpu"

    def to(self, _device: str) -> _FakeModel:
        return self

    def eval(self) -> None:
        return None


def _install_snapshot(monkeypatch: pytest.MonkeyPatch, result: str | Exception) -> list[Any]:
    calls: list[Any] = []

    def fake_snapshot_download(repo_id: str, **kwargs: Any) -> str:
        calls.append((repo_id, kwargs))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    return calls


def test_llm_loader_passes_snapshot_path_not_hub_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    calls = _install_snapshot(monkeypatch, SNAP)
    seen: dict[str, Any] = {}

    def tok(path: str, **_kw: Any) -> str:
        seen["tok"] = path
        return "tokenizer"

    def mdl(path: str, **_kw: Any) -> _FakeModel:
        seen["model"] = path
        return _FakeModel()

    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tok)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", mdl)
    provider = LocalQwenLlmProvider(model_id=HUB_ID, device="cpu", dtype="float32")
    provider.load()
    assert seen == {"tok": SNAP, "model": SNAP}
    assert calls == [(HUB_ID, {"local_files_only": True})]


def test_tts_loader_passes_snapshot_path_not_hub_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _install_snapshot(monkeypatch, SNAP)
    seen: list[Any] = []

    class FakeTts:
        @staticmethod
        def from_pretrained(path: str, **kw: Any) -> str:
            seen.append((path, kw["device_map"]))
            return "model"

    monkeypatch.setitem(sys.modules, "qwen_tts", types.SimpleNamespace(Qwen3TTSModel=FakeTts))
    synth = QwenCustomVoiceSynthesizer.from_settings(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "cpu", "float32", "sdpa"
    )
    assert seen == [(SNAP, "cpu")]
    assert synth._model == "model"


def test_missing_snapshot_error_names_id_and_download(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_snapshot(monkeypatch, FileNotFoundError("not cached"))
    with pytest.raises(RuntimeError) as info:
        resolve_local_model_path(HUB_ID)
    text = str(info.value)
    assert HUB_ID in text
    assert "download" in text
    assert "huggingface-cli download Qwen/Qwen3-4B-Instruct-2507" in text


def test_local_directory_passes_through_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_snapshot(monkeypatch, SNAP)
    assert resolve_local_model_path(str(tmp_path)) == str(tmp_path)
    assert calls == []


def _capture_system_prompt(monkeypatch: pytest.MonkeyPatch, request: ContentRequest) -> str:
    class FakeTokenizer:
        eos_token_id = 0
        captured: list[Any] = []

        def apply_chat_template(self, messages: list[dict[str, str]], **_kw: Any) -> str:
            self.captured.append(messages)
            raise StopIteration  # stop before any tensor work

    provider = LocalQwenLlmProvider(device="cpu")
    tokenizer = FakeTokenizer()
    provider._model = _FakeModel()  # type: ignore[assignment]
    provider._tokenizer = tokenizer
    prompt = prompts.build_prompt(request)
    system = provider._chat_system_prompt() if request.kind is ContentKind.CHAT else None
    with pytest.raises(StopIteration):
        provider._generate_sync(prompt, system)
    assert tokenizer.captured[0][1] == {"role": "user", "content": prompt}
    return tokenizer.captured[0][0]["content"]


def test_chat_request_uses_chat_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    req = ContentRequest(kind=ContentKind.CHAT, transcript_text="hi", requester_name="Sam")
    system = _capture_system_prompt(monkeypatch, req)
    assert system == prompts.CHAT_SYSTEM_PROMPT
    assert "short" in system
    assert "spoken" in system
    assert "radio" not in system.lower()


def test_non_chat_request_keeps_radio_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    req = ContentRequest(kind=ContentKind.SONG, song_name="S", artist_name="A")
    assert _capture_system_prompt(monkeypatch, req) == prompts.RADIO_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_generate_routes_chat_prompt_through_coordinator() -> None:
    class Gpu:
        got: tuple[Any, ...] = ()

        async def run_blocking(self, fn: Any, *args: Any) -> str:
            self.got = args
            return "ok"

    gpu = Gpu()
    provider = LocalQwenLlmProvider(device="cpu", gpu_coordinator=gpu)  # type: ignore[arg-type]
    req = ContentRequest(kind=ContentKind.CHAT, transcript_text="hi")
    assert await provider.generate(req) == "ok"
    assert gpu.got == (prompts.build_prompt(req), prompts.CHAT_SYSTEM_PROMPT)
    gpu.got = ()
    song = ContentRequest(kind=ContentKind.SONG, song_name="S")
    await provider.generate(song)
    assert gpu.got == (prompts.build_prompt(song),)


def test_helper_is_shared_with_tts_loader() -> None:
    assert llm_mod.resolve_local_model_path is resolve_local_model_path
