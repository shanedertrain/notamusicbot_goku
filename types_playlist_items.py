from dataclasses import dataclass
from typing import AsyncIterator
from pathlib import Path
import yt_dlp

import discord

from users import User
from spotify_handler import SpotifyTrack

@dataclass
class VideoInfo:
    title: str
    url: str
    duration: int
    uploader: str
    formats: list

@dataclass
class Media:
    id: str

@dataclass
class Video(Media):
    requester: User
    video_info: VideoInfo
    requested_channel: discord.TextChannel

    def __repr__(self) -> str:
        return (f"Video(id={self.id!r}, requester={self.requester!r}, requested_channel={self.requested_channel!r})")

@dataclass
class Audio(Media):
    filepath: Path

@dataclass
class SpotifyMedia(Media):
    spotify_info: SpotifyTrack
    requester: User

async def extract_youtube_video_info(url: str) -> AsyncIterator[VideoInfo]:
    ydl_opts = {
        'quiet': True,
        'format': 'bestaudio',
        'dump_single_json': True,
        'noplaylist': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            yield VideoInfo(
                title=info['title'],
                url=info['webpage_url'],
                duration=info['duration'],
                uploader=info.get('uploader', 'Unknown'),
                formats=info.get('formats', []),
            )
        except yt_dlp.utils.DownloadError as e:
            print(f"Error extracting info for {url}: {e}")
