"""Media acquisition: YouTube (yt-dlp), download cache, and Spotify metadata."""

from notamusicbot.media.downloader import cleanup_file, download_with_fallback
from notamusicbot.media.spotify import SpotifyResolver, SpotifyTrack
from notamusicbot.media.youtube import YouTubeResolver, build_yt_opts, discover_js_runtimes

__all__ = [
    "SpotifyResolver",
    "SpotifyTrack",
    "YouTubeResolver",
    "build_yt_opts",
    "cleanup_file",
    "discover_js_runtimes",
    "download_with_fallback",
]
