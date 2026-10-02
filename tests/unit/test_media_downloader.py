"""Unit tests for the download cache and explicit temp cleanup (fake yt-dlp)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from notamusicbot.domain.media import MediaKind, PlayableMedia
from notamusicbot.media.downloader import cleanup_file, cleanup_media, download_with_fallback


class FakeYdl:
    """Records outtmpl targets; optionally fails per client."""

    def __init__(self, fail_clients: set[str] | None = None) -> None:
        self.fail_clients = fail_clients or set()
        self.downloads: list[str] = []
        self.current: dict | None = None

    def __call__(self, opts: dict) -> FakeYdl:
        self.current = opts
        return self

    def __enter__(self) -> FakeYdl:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def download(self, urls: list[str]) -> None:
        client = self.current["extractor_args"]["youtube"]["player_client"][0]
        if client in self.fail_clients:
            raise RuntimeError(f"client {client} failed")
        self.downloads.append(self.current["outtmpl"])
        Path(self.current["outtmpl"]).write_bytes(b"audio")


def test_download_uses_fallback_clients_and_writes_cache(tmp_path: Path) -> None:
    ydl = FakeYdl(fail_clients={"default"})
    clients = ("default", "web")
    path = download_audio_via(
        url="https://youtube.com/watch?v=1", clients=clients, cache_dir=tmp_path, ydl=ydl
    )
    assert path.exists()
    assert path.parent == tmp_path
    assert path.suffix == ".m4a"


def download_audio_via(**kwargs):
    from notamusicbot.media.downloader import download_audio

    ydl = kwargs.pop("ydl")
    return asyncio.run(
        download_audio(
            kwargs["url"],
            build_opts=lambda client: {"extractor_args": {"youtube": {"player_client": [client]}}},
            clients=kwargs["clients"],
            cache_dir=kwargs["cache_dir"],
            ydl_factory=lambda opts: ydl(opts),
        )
    )


def test_download_raises_last_error_when_all_clients_fail(tmp_path: Path) -> None:
    ydl = FakeYdl(fail_clients={"default", "web"})
    with pytest.raises(RuntimeError):
        download_audio_via(url="u", clients=("default", "web"), cache_dir=tmp_path, ydl=ydl)
    # failed attempts leave no partial files behind
    assert list(tmp_path.glob("*.m4a")) == []


def test_download_sync_helper_no_clients_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        download_with_fallback(
            "u",
            build_opts=lambda client: {},
            clients=(),
            cache_dir=tmp_path,
            ydl_factory=lambda opts: None,
        )


def test_cleanup_file_removes_and_tolerates_missing(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"x")
    cleanup_file(path)
    assert not path.exists()
    cleanup_file(path)  # missing ok
    cleanup_file(None)  # none ok


def test_cleanup_media_only_deletes_temp_items(tmp_path: Path) -> None:
    temp = tmp_path / "temp.m4a"
    kept = tmp_path / "kept.wav"
    temp.write_bytes(b"a")
    kept.write_bytes(b"a")
    temp_media = PlayableMedia(media_id="t", kind=MediaKind.LOCAL, source_path=temp, temp_file=True)
    local_media = PlayableMedia(media_id="l", kind=MediaKind.LOCAL, source_path=kept)
    cleanup_media(temp_media)
    cleanup_media(local_media)
    assert not temp.exists()
    assert kept.exists()
