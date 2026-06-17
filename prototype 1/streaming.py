"""Streaming ingestion helpers (Spotify)

Optional: requires `spotipy` and valid Spotify client credentials set via
environment variables `SPOTIPY_CLIENT_ID` and `SPOTIPY_CLIENT_SECRET`.

This module fetches metadata (popularity, preview URL) that can be merged
into the track features pipeline to compute `familiarity`.
"""
from typing import Optional, List, Dict

def _try_import_spotipy():
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
        return spotipy, SpotifyClientCredentials
    except Exception:
        return None, None


class SpotifyIngest:
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        spotipy_mod, cred_cls = _try_import_spotipy()
        if spotipy_mod is None:
            raise RuntimeError('spotipy not installed; pip install spotipy')
        creds = {}
        if client_id and client_secret:
            creds = {'client_id': client_id, 'client_secret': client_secret}
        self._sp = spotipy_mod.Spotify(client_credentials_manager=cred_cls(**creds) if creds else cred_cls())

    def track_meta(self, track_id_or_uri: str) -> Dict:
        """Return metadata for a Spotify track: popularity (0-100), preview_url, artists, name."""
        t = self._sp.track(track_id_or_uri)
        return {
            'spotify_popularity': t.get('popularity'),
            'preview_url': t.get('preview_url'),
            'artists': [a.get('name') for a in t.get('artists', [])],
            'name': t.get('name'),
            'duration_ms': t.get('duration_ms')
        }

    def playlist_tracks(self, playlist_id_or_uri: str) -> List[Dict]:
        items = []
        results = self._sp.playlist_items(playlist_id_or_uri)
        while results:
            for it in results.get('items', []):
                tr = it.get('track')
                if tr:
                    items.append({
                        'id': tr.get('id'),
                        'name': tr.get('name'),
                        'artists': [a.get('name') for a in tr.get('artists', [])],
                        'popularity': tr.get('popularity')
                    })
            if results.get('next'):
                results = self._sp.next(results)
            else:
                break
        return items


__all__ = ['SpotifyIngest']
