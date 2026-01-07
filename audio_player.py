from datetime import timedelta as td
from typing import Dict, Tuple, Union
import os
import asyncio
import uuid

import discord
from discord.ext import commands
import configuration as cfg
from configuration import LOGGER

from types_playlist_items import Video, Audio, SpotifyMedia

import subprocess
import shutil
from pathlib import Path
import yt_dlp

FFMPEG_BEFORE_OPTIONS = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 -nostdin"
FFMPEG_OPTIONS = "-vn -filter:a volume=0.5"

# yt-dlp needs a player client that can derive the signed stream URL. Avoid the
# android client (requires PO token) and try a small set of desktop-friendly
# clients instead.
YTDLP_CLIENTS = ["default", "web", "tv_embedded"]
YTDLP_BASE_OPTS = {
    "quiet": True,
    "format": "bestaudio/best",
    "noplaylist": True,
}

YT_CACHE_FOLDER = cfg.FOLDER_OUTPUT / "yt_cache"
YT_CACHE_FOLDER.mkdir(exist_ok=True, parents=True)
YT_COOKIES_FILE = cfg.FOLDER_INPUT / "youtube_cookies.txt"

# Use a stable desktop UA to reduce PO-token prompts.
DESKTOP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

LOCAL_DENO = Path(__file__).resolve().parent / "tools" / "deno" / ("deno.exe" if os.name == "nt" else "deno")

def _detect_js_runtimes():
    """Find available JS runtimes for yt-dlp (node/deno/quickjs)."""
    runtimes = {}
    for name in ("node", "deno", "qjs", "quickjs"):
        found = shutil.which(name)
        if found:
            runtimes[name] = {"path": found}
    if LOCAL_DENO.exists():
        runtimes["deno"] = {"path": str(LOCAL_DENO)}

    if not runtimes:
        LOGGER.warning(
            "No JavaScript runtime found for yt-dlp. Install Node/Deno/QuickJS or place a runtime in tools/deno/."
        )
    else:
        LOGGER.info("JS runtimes detected for yt-dlp: %s", {k: v.get('path') for k, v in runtimes.items()})
    return runtimes

# Prefer Node for JS runtime, fall back to Deno/QuickJS if available.
YT_JS_RUNTIMES = _detect_js_runtimes()

def build_yt_opts(client: str) -> Dict:
    opts = {
        **YTDLP_BASE_OPTS,
        "extractor_args": {"youtube": {"player_client": [client]}},
        "http_headers": {"User-Agent": DESKTOP_UA},
        "js_runtimes": YT_JS_RUNTIMES,
    }
    if YT_COOKIES_FILE.exists():
        opts["cookiefile"] = str(YT_COOKIES_FILE)
    return opts

class AudioPlayer:
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.playlist = []
        self.voice_client = None

    async def run(self):
        while True:
            await self.play_audio()
            await asyncio.sleep(1)

    async def extract_audio_url(self, url: str) -> Tuple[str, Dict[str, str]]:
        """Use yt-dlp to get the best audio stream URL and its required headers."""

        def _extract() -> Tuple[str, Dict[str, str]]:
            last_err = None
            for client in YTDLP_CLIENTS:
                opts = build_yt_opts(client)
                try:
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                    return info.get("url", ""), info.get("http_headers", {})
                except Exception as e:  # yt-dlp raises generic exceptions here
                    LOGGER.warning(f"yt-dlp client '{client}' failed for {url}: {e}")
                    last_err = e
                    continue
            if last_err:
                raise last_err
            return "", {}

        try:
            return await asyncio.to_thread(_extract)
        except Exception as e:
            LOGGER.error(f"yt-dlp error while extracting {url}: {e}", exc_info=True)
            return "", {}

    async def download_audio(self, url: str) -> Path:
        """Download audio to a temp cache file for resilient playback."""

        def _download() -> Path:
            last_err = None
            for client in YTDLP_CLIENTS:
                out_path = YT_CACHE_FOLDER / f"{uuid.uuid4()}.m4a"
                opts = {**build_yt_opts(client), "outtmpl": str(out_path)}
                try:
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        ydl.download([url])
                    return out_path
                except Exception as e:
                    LOGGER.warning(f"yt-dlp download with client '{client}' failed for {url}: {e}")
                    last_err = e
                    continue
            if last_err:
                raise last_err
            return Path()

        try:
            return await asyncio.to_thread(_download)
        except Exception as e:
            LOGGER.error(f"yt-dlp download error for {url}: {e}", exc_info=True)
            return Path()

    async def play_audio(self):
        try:
            if len(self.playlist) > 0:
                self.voice_client = self.bot.voice_clients[0] if self.bot.voice_clients else None
                if self.voice_client and self.voice_client.is_connected():
                    media = self.playlist.pop(0)
                    LOGGER.debug(f"Popped: {media}")

                    if isinstance(media, Video):
                        audio_url, headers = await self.extract_audio_url(media.video_info.url)
                        local_file = None

                        if not audio_url:
                            LOGGER.warning(f"No streaming URL for {media.video_info.title}; downloading instead.")
                            local_file = await self.download_audio(media.video_info.url)
                        else:
                            # Also prepare a downloaded copy to avoid mid-stream signature/403 failures.
                            local_file = await self.download_audio(media.video_info.url)

                        header_opts = ""
                        if headers:
                            header_blob = "".join(f"{k}: {v}\\r\\n" for k, v in headers.items())
                            header_opts = f' -headers "{header_blob}"'

                        await media.requested_channel.send(
                            f"Now playing: {media.video_info.title} | Duration: {str(td(seconds=media.video_info.duration))} | Requester: {media.requester.screen_name} ({media.requester.real_name})"
                        )
                        if local_file and local_file.exists():
                            self.voice_client.play(discord.FFmpegPCMAudio(str(local_file)))
                        else:
                            self.voice_client.play(
                                discord.FFmpegPCMAudio(
                                    audio_url,
                                    before_options=f"{FFMPEG_BEFORE_OPTIONS}{header_opts}",
                                    options=FFMPEG_OPTIONS,
                                )
                            )

                    elif isinstance(media, Audio):
                        play_source = media.filepath
                        self.voice_client.play(discord.FFmpegPCMAudio(str(play_source)))

                    elif isinstance(media, SpotifyMedia):
                        play_source = media.spotify_info.uri
                        self.voice_client.play(discord.FFmpegPCMAudio(play_source))

                while self.voice_client.is_playing():
                    await asyncio.sleep(1)

        except Exception as e:
            LOGGER.error(e)

    def add_to_playlist(self, media_item: Union[Video, Audio, SpotifyMedia]):
        self.playlist.append(media_item)

    async def clear_playlist(self):
        self.playlist = []
        if self.voice_client and self.voice_client.is_playing():
            self.voice_client.stop()
