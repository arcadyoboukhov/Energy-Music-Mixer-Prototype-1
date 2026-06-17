"""Transition engine recommending transition types between two tracks.

Heuristics-driven, lightweight rules intended to be "never awkward":
 - beatmatch when same/adjacent genre, bpm close and key compatible
 - echo out + drop for genre jumps or dramatic energy increases
 - fade + re-entry for safe smooth transitions
 - hard cut for deliberate hype / big familiarity jump

This module is intentionally simple and conservative.
"""
from typing import Optional, Dict
import math

import musicscan


class TransitionEngine:
    def __init__(self):
        self.keys = musicscan.KEYS
        self.key_to_idx = {k: i for i, k in enumerate(self.keys)}

    def _tonic_idx(self, keyobj: Optional[Dict]) -> Optional[int]:
        try:
            t = (keyobj or {}).get('tonic')
            t = musicscan.normalize_tonic(t)
            return self.key_to_idx.get(t)
        except Exception:
            return None

    def _semitone_diff(self, a: Optional[Dict], b: Optional[Dict]) -> Optional[int]:
        ia = self._tonic_idx(a)
        ib = self._tonic_idx(b)
        if ia is None or ib is None:
            return None
        delta = (ib - ia) % 12
        if delta > 6:
            delta -= 12
        return abs(int(delta))

    def _bpm_diff(self, a: Optional[Dict], b: Optional[Dict]) -> Optional[float]:
        try:
            return abs(float((a or {}).get('bpm') or 0.0) - float((b or {}).get('bpm') or 0.0))
        except Exception:
            return None

    def _mfcc_similar(self, a: Optional[Dict], b: Optional[Dict]) -> Optional[float]:
        try:
            ma = (a or {}).get('mfcc_mean')
            mb = (b or {}).get('mfcc_mean')
            if not ma or not mb or len(ma) != len(mb):
                return None
            # cosine similarity
            dot = 0.0
            na = 0.0
            nb = 0.0
            for x, y in zip(ma, mb):
                dot += float(x) * float(y)
                na += float(x) * float(x)
                nb += float(y) * float(y)
            if na <= 0 or nb <= 0:
                return None
            return dot / (math.sqrt(na) * math.sqrt(nb))
        except Exception:
            return None

    def recommend(self, prev: Optional[Dict], nxt: Dict, state: Optional[Dict] = None, prefer_hype: bool = False) -> Dict:
        """Recommend a transition type and return metadata.

        Returns a dict: {type, confidence, reason, details...}
        """
        if nxt is None:
            return {'type': 'fade_reentry', 'confidence': 0.2, 'reason': 'no next track'}

        if prev is None:
            return {'type': 'fade_reentry', 'confidence': 0.5, 'reason': 'no previous track'}

        # basics
        bpm_diff = self._bpm_diff(prev, nxt) or 999.0
        semis = self._semitone_diff(prev.get('key') if prev else None, nxt.get('key') if nxt else None)
        energy_prev = int((prev.get('energy') or 0))
        energy_next = int((nxt.get('energy') or 0))
        energy_delta = energy_next - energy_prev

        fam_prev = int((prev.get('familiarity') or 50))
        fam_next = int((nxt.get('familiarity') or 50))
        fam_delta = fam_next - fam_prev

        mfcc_cos = self._mfcc_similar(prev, nxt)
        same_genre = False
        if (prev.get('cluster') is not None and prev.get('cluster') == nxt.get('cluster')):
            same_genre = True
        elif mfcc_cos is not None and mfcc_cos >= 0.98:
            same_genre = True

        # heuristic rules
        # big familiarity jump + energy jump -> hard cut (dramatic)
        if abs(fam_delta) >= 40 and abs(energy_delta) >= 30:
            return {
                'type': 'hard_cut',
                'confidence': 0.95,
                'reason': 'large familiarity + energy jump (dramatic)',
                'bpm_diff': bpm_diff,
                'semitone_diff': semis,
                'energy_delta': energy_delta,
                'familiarity_delta': fam_delta
            }

        # same genre -> prefer beatmatch or fade
        if same_genre:
            if bpm_diff <= 4 and (semis is None or semis <= 2):
                return {'type': 'beatmatch', 'confidence': 0.9, 'reason': 'same genre, bpm/key compatible', 'bpm_diff': bpm_diff, 'semitone_diff': semis}
            # close enough for smooth blend
            if bpm_diff <= 10:
                return {'type': 'fade_reentry', 'confidence': 0.75, 'reason': 'same genre, moderate bpm diff', 'bpm_diff': bpm_diff}

        # different genre or big energy change
        if not same_genre:
            # hype/hard transitions for big energy increases
            if energy_delta >= 25 or (prefer_hype and energy_next >= 80):
                return {'type': 'echo_out_drop', 'confidence': 0.85, 'reason': 'genre change + energy jump (dramatic effect)', 'energy_delta': energy_delta}
            # if bpm still similar, a creative echo works
            if bpm_diff <= 8:
                return {'type': 'fade_reentry', 'confidence': 0.6, 'reason': 'genre change but bpm close (safe fade)', 'bpm_diff': bpm_diff}
            # otherwise, safer to apply an effect transition
            return {'type': 'echo_out_drop', 'confidence': 0.6, 'reason': 'genre change & bpm different', 'bpm_diff': bpm_diff}

        # fallback
        return {'type': 'fade_reentry', 'confidence': 0.5, 'reason': 'fallback safe fade', 'bpm_diff': bpm_diff, 'semitone_diff': semis}


__all__ = ['TransitionEngine']
