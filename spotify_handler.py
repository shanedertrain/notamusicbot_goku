from typing import Optional, List
from dataclasses import dataclass
import os

from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

load_dotenv()
client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

@dataclass
class SpotifyTrack:
    name: str
    artists: List[str]
    album: str
    uri: str

class SpotifyHandler:
    def __init__(self, client_id: str=client_id, client_secret: str=client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.sp = self._authenticate_spotify()

    def _authenticate_spotify(self) -> spotipy.Spotify:
        auth_manager = SpotifyClientCredentials(client_id=self.client_id, client_secret=self.client_secret)
        sp = spotipy.Spotify(auth_manager=auth_manager)
        return sp

    def search_track(self, query: str) -> Optional[SpotifyTrack]:
        results = self.sp.search(q=query, limit=1)
        if results['tracks']['items']:
            track_info = results['tracks']['items'][0]
            track = SpotifyTrack(
                name=track_info['name'],
                artists=[artist['name'] for artist in track_info['artists']],
                album=track_info['album']['name'],
                uri=track_info['uri']
            )
            return track
        else:
            return None

    def play_track(self, track: SpotifyTrack) -> None:
        self.sp.start_playback(uris=[track.uri])

if __name__ == "__main__":
    spotify_handler = SpotifyHandler()
    track = spotify_handler.search_track("Total Eclipse of the Heart")
    if track:
        spotify_handler.play_track(track)
