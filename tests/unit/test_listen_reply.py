"""Listen-mode reply synthesis (fakes only; no GPU, no weights)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from conftest import FailingSynthesizer, FakeConverter, FakeSynthesizer
from notamusicbot.domain.speech import AudioArtifact, VoiceConversionProfile
from notamusicbot.domain.voice_profiles import VoiceProfile
from notamusicbot.listen import reply as reply_module
from notamusicbot.listen.reply import (
    BackchannelCache,
    ReplyAudio,
    ReplySynthesisError,
    ReplySynthesizer,
    trim_backchannel,
    warm_voice_converter,
)
from notamusicbot.speech.pipeline import SpeechPipeline


def _profile(backend: str = "passthrough") -> VoiceProfile:
    return VoiceProfile.model_validate(
        {
            "id": "goku",
            "persona": {"display_name": "Goku"},
            "tts": {"speaker": "Ryan"},
            "voice_conversion": {"backend": backend},
            "audio": {},
        }
    )


class CountingPipeline(SpeechPipeline):
    """Real pipeline over the fakes, counting ``run`` calls."""

    def __init__(self, out: Path, synth: FakeSynthesizer | None = None) -> None:
        super().__init__(synth or FakeSynthesizer(), FakeConverter(), output_dir=out)
        self.run_count = 0

    async def run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.run_count += 1
        return await super().run(*args, **kwargs)


class RaisingPipeline:
    async def run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("rvc unavailable")


def _files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*")] if root.exists() else []


async def test_pcm_length_and_artifact_cleanup(tmp_path: Path) -> None:
    out = tmp_path / "speech"
    pipeline = CountingPipeline(out, FakeSynthesizer(seconds=0.5))
    reply = await ReplySynthesizer(pipeline, _profile(), artifact_root=out).synthesize_sentence(
        "Hello there."
    )
    assert isinstance(reply, ReplyAudio)
    assert reply.duration_s == pytest.approx(0.5, abs=1e-3)
    assert abs(len(reply.pcm) - round(reply.duration_s * 48000) * 2 * 2) <= 4
    assert len(reply.pcm) == 24000 * 4  # 0.5 s * 48000 frames * 2 ch * 2 bytes
    assert _files(out) == []
    assert pipeline.run_count == 1


async def test_pipeline_failure_raises_and_leaves_nothing(tmp_path: Path) -> None:
    out = tmp_path / "speech"
    synth = ReplySynthesizer(RaisingPipeline(), _profile(), artifact_root=out)
    with pytest.raises(ReplySynthesisError):
        await synth.synthesize_sentence("Hello.")
    assert _files(out) == []


async def test_real_pipeline_failure_removes_failed_job_dir(tmp_path: Path) -> None:
    out = tmp_path / "speech"
    pipeline = CountingPipeline(out, FailingSynthesizer())
    synth = ReplySynthesizer(pipeline, _profile(), artifact_root=out)
    with pytest.raises(ReplySynthesisError):
        await synth.synthesize_sentence("Hello.")
    assert pipeline.run_count == 1
    assert _files(out) == []


async def test_backchannel_second_prerender_hits_cache(tmp_path: Path) -> None:
    out = tmp_path / "speech"
    cache_dir = tmp_path / "bc"
    pipeline = CountingPipeline(out)
    synth = ReplySynthesizer(pipeline, _profile(), artifact_root=out)
    texts = ["Hm?", "Oh yeah?", "Let me think."]

    cache = BackchannelCache(cache_dir, texts, synth)
    assert await cache.prerender() == 3
    assert pipeline.run_count == 3
    assert len(list(cache_dir.glob("*.pcm"))) == 3

    assert await cache.prerender() == 0
    assert pipeline.run_count == 3  # unchanged: served from cache

    # A fresh cache object (new process) also hits the files on disk.
    again = BackchannelCache(cache_dir, texts, synth)
    assert await again.prerender() == 0
    assert pipeline.run_count == 3

    clip = again.get_random(random.Random(1))
    # The 0.5 s tone starts at sample 0 (silent), which trimming removes: 23999 frames.
    assert len(clip.pcm) == 23999 * 4
    assert clip.duration_s == pytest.approx(23999 / 48000)
    assert _files(out) == []


async def test_backchannel_key_depends_on_profile_id(tmp_path: Path) -> None:
    out = tmp_path / "speech"
    pipeline = CountingPipeline(out)
    other = _profile().model_copy(update={"id": "vegeta"})
    a = BackchannelCache(tmp_path / "bc", ["Hm?"], ReplySynthesizer(pipeline, _profile()))
    b = BackchannelCache(tmp_path / "bc", ["Hm?"], ReplySynthesizer(pipeline, other))
    await a.prerender()
    await b.prerender()
    assert pipeline.run_count == 2
    assert len(list((tmp_path / "bc").glob("*.pcm"))) == 2


async def test_get_random_empty_raises(tmp_path: Path) -> None:
    cache = BackchannelCache(tmp_path, ["Hm?"], ReplySynthesizer(RaisingPipeline(), _profile()))
    assert await cache.prerender() == 0  # failure skipped, not fatal
    with pytest.raises(LookupError):
        cache.get_random()


async def test_warm_runs_one_conversion_on_pipeline_and_router(tmp_path: Path) -> None:
    pipeline = CountingPipeline(tmp_path / "speech")
    assert await warm_voice_converter(pipeline, _profile("rvc")) is True
    conv = pipeline._converter  # noqa: SLF001
    assert len(conv.calls) == 1
    source, profile = conv.calls[0]
    assert isinstance(source, AudioArtifact)
    assert profile.backend == "rvc"
    assert pipeline.run_count == 0  # no TTS, no pipeline job

    direct = FakeConverter()
    assert await warm_voice_converter(direct, VoiceConversionProfile(backend="rvc")) is True
    assert len(direct.calls) == 1


async def test_warm_skips_passthrough_and_survives_failure() -> None:
    direct = FakeConverter()
    assert await warm_voice_converter(direct, _profile("passthrough")) is False
    assert direct.calls == []

    class Broken:
        async def convert(self, *a, **k):  # type: ignore[no-untyped-def]
            raise RuntimeError("no worker")

    assert await warm_voice_converter(Broken(), _profile("rvc")) is False


RATE = 48000


def _clip(*segments: tuple[float, float]) -> ReplyAudio:
    """Stereo clip from (seconds, amplitude) segments; non-zero amplitude is a 440 Hz tone."""
    parts = []
    for seconds, amp in segments:
        n = round(seconds * RATE)
        mono = amp * 32767 * np.sin(2 * np.pi * 440 * (np.arange(n) + 7) / RATE)
        parts.append(np.repeat(mono[:, None], 2, axis=1))
    data = np.concatenate(parts).astype("<i2")
    return ReplyAudio(pcm=data.tobytes(), duration_s=data.shape[0] / RATE)


def _frames(audio: ReplyAudio) -> np.ndarray:
    return np.frombuffer(audio.pcm, dtype="<i2").reshape(-1, 2)


def test_trim_cuts_trailing_silence() -> None:
    out = trim_backchannel(_clip((0.5, 0.5), (3.0, 0.0)))
    assert 0.45 <= out.duration_s <= 0.5 + 0.15
    assert len(out.pcm) == _frames(out).shape[0] * 4
    assert out.duration_s == pytest.approx(_frames(out).shape[0] / RATE)


def test_trim_removes_leading_silence() -> None:
    out = trim_backchannel(_clip((0.4, 0.0), (0.5, 0.5)))
    assert out.duration_s == pytest.approx(0.5, abs=0.01)
    assert np.abs(_frames(out)[:200]).max() > 0  # starts at the tone, not at silence


def test_trim_caps_at_1_2s_and_fades_last_80ms() -> None:
    out = trim_backchannel(_clip((3.0, 0.5)))
    frames = _frames(out)
    assert abs(frames.shape[0] - round(1.2 * RATE)) <= 1
    fade = round(0.08 * RATE)
    assert np.abs(frames[-1]).max() <= 8  # near zero (full scale is 32767)
    assert np.abs(frames[-fade - 200 : -fade]).max() > 10000  # untouched before the fade
    assert np.abs(frames[-fade // 2 :]).max() < np.abs(frames[-fade : -fade // 2]).max()


def test_trim_keeps_short_clean_clip_and_all_silent_clip() -> None:
    silent = _clip((0.5, 0.0))
    assert trim_backchannel(silent) is silent
    out = trim_backchannel(_clip((0.3, 0.5)))
    assert out.duration_s == pytest.approx(0.3, abs=0.001)


async def test_cached_clip_is_trimmed(tmp_path: Path) -> None:
    out = tmp_path / "speech"
    synth = ReplySynthesizer(CountingPipeline(out, FakeSynthesizer(seconds=2.0)), _profile())
    cache = BackchannelCache(tmp_path / "bc", ["Hm?"], synth)
    await cache.prerender()
    clip = cache.get_random()
    assert abs(len(clip.pcm) // 4 - round(1.2 * RATE)) <= 1
    stored = (next((tmp_path / "bc").glob("*.pcm"))).read_bytes()
    assert stored == clip.pcm


async def test_backchannel_key_changes_with_trim_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = CountingPipeline(tmp_path / "speech")
    synth = ReplySynthesizer(pipeline, _profile())
    cache = BackchannelCache(tmp_path / "bc", ["Hm?"], synth)
    before = cache._path_for("Hm?")  # noqa: SLF001
    assert before == cache._path_for("Hm?")  # noqa: SLF001  (stable)
    monkeypatch.setattr(reply_module, "BACKCHANNEL_TRIM_VERSION", 2)
    assert cache._path_for("Hm?") != before  # noqa: SLF001
    monkeypatch.setattr(reply_module, "BACKCHANNEL_MAX_S", 0.9)
    assert cache._path_for("Hm?") != before  # noqa: SLF001
