from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math
import os
import re

import musicscan
import transition_engine
import memory
import time
try:
    from tools.parse_nl_prompt import parse_prompt
except Exception:
    parse_prompt = None


@dataclass
class PartyState:
    energy_level: int = 50               # 0..100
    trajectory: str = "steady"         # rising, falling, steady
    crowd_type: str = "mixed"          # e.g., mixed, party, chill
    time_elapsed: float = 0.0            # seconds since start
    familiarity_bias: float = 0.7        # 0..1, higher => prefer familiar
    current_genre_cluster: Optional[str] = None
    prev_track_path: Optional[str] = None


class EnergyEngine:
    """Generates and adjusts a target energy curve for the party timeline.

    Default target points (seconds -> energy):
      0s:30, 30s:50, 60s:70, 90s:90, 120s:75, 150s:95, 180s:60
    """

    DEFAULT_POINTS: List[Tuple[float, int]] = [
        (0.0, 30), (30.0, 50), (60.0, 70), (90.0, 90), (120.0, 75), (150.0, 95), (180.0, 60)
    ]

    def __init__(self, points: Optional[List[Tuple[float, int]]] = None):
        pts = points or self.DEFAULT_POINTS
        # ensure sorted by time
        self.points = sorted([(float(t), int(e)) for t, e in pts], key=lambda x: x[0])

    def target_at(self, t_seconds: float) -> float:
        """Return linearly interpolated target energy at time t_seconds."""
        if not self.points:
            return 50.0
        if t_seconds <= self.points[0][0]:
            return float(self.points[0][1])
        for i in range(len(self.points) - 1):
            t0, e0 = self.points[i]
            t1, e1 = self.points[i + 1]
            if t0 <= t_seconds <= t1:
                # linear interp
                if t1 == t0:
                    return float(e0)
                frac = (t_seconds - t0) / (t1 - t0)
                return float(e0 + frac * (e1 - e0))
        return float(self.points[-1][1])

    def generate_curve(self, duration_seconds: float = 180.0, step: float = 1.0) -> List[Tuple[float, float]]:
        times = list(self._frange(0.0, duration_seconds, step))
        return [(t, self.target_at(t)) for t in times]

    def adjust_for_events(self, base_target: float, events: Optional[Dict] = None) -> float:
        """Adjust base target energy using lightweight heuristics.

        Supported events keys: 'skip_count', 'user_input' ('more_hype'|'less_hype'), 'upvotes'
        """
        e = events or {}
        delta = 0.0
        skips = int(e.get('skip_count', 0) or 0)
        delta -= min(20.0, skips * 3.0)
        ui = e.get('user_input')
        if ui == 'more_hype':
            delta += 8.0
        elif ui == 'less_hype':
            delta -= 8.0
        up = int(e.get('upvotes', 0) or 0)
        delta += min(10.0, up * 1.5)
        out = max(0.0, min(100.0, base_target + delta))
        return out

    @staticmethod
    def _frange(start: float, stop: float, step: float):
        t = start
        while t <= stop:
            yield float(t)
            t += step


class TrackSelector:
    """Selects the best next track to move party toward target energy state."""

    def __init__(self, engine: EnergyEngine):
        self.engine = engine
        self.keys = musicscan.KEYS
        self.key_to_idx = {k: i for i, k in enumerate(self.keys)}
        # transition engine for recommending transition types
        try:
            self.transitioner = transition_engine.TransitionEngine()
        except Exception:
            self.transitioner = None

    def _normalize_tonic(self, t: Optional[str]) -> Optional[str]:
        return musicscan.normalize_tonic(t)

    def _bpm_score(self, a: float, b: float) -> float:
        try:
            return max(0.0, 1.0 - abs(float(a) - float(b)) / 30.0)
        except Exception:
            return 0.5

    def _key_score(self, ka: Dict, kb: Dict) -> float:
        try:
            ta = self._normalize_tonic(ka.get('tonic')) if ka else None
            tb = self._normalize_tonic(kb.get('tonic')) if kb else None
            ia = self.key_to_idx.get(ta)
            ib = self.key_to_idx.get(tb)
            if ia is None or ib is None:
                return 0.5
            sem = abs(ia - ib)
            sem = min(sem, 12 - sem)
            score = max(0.0, 1.0 - (sem / 6.0))
            if (ka.get('mode') if ka else None) == (kb.get('mode') if kb else None):
                score = min(1.0, score + 0.15)
            return float(score)
        except Exception:
            return 0.5

    def _translation_score(self, ka: Dict, kb: Dict) -> float:
        try:
            ta = self._normalize_tonic(ka.get('tonic')) if ka else None
            tb = self._normalize_tonic(kb.get('tonic')) if kb else None
            ia = self.key_to_idx.get(ta)
            ib = self.key_to_idx.get(tb)
            if ia is None or ib is None:
                return 0.5
            delta = (ib - ia) % 12
            if delta > 6:
                delta -= 12
            abs_shift = abs(int(delta))
            translation = max(0.0, 1.0 - (abs_shift / 3.0))
            if (ka.get('mode') if ka else None) != (kb.get('mode') if kb else None):
                translation *= 0.8
            return float(translation)
        except Exception:
            return 0.5

    def _canonical_title(self, path_or_feat) -> str:
        """Derive a simplified canonical title from a file path or feature dict.

        This strips extensions, parenthetical/remix tags, and common remix/live
        suffixes so different versions of the same song normalize to the same
        token sequence for duplicate-detection.
        """
        s = ''
        try:
            if isinstance(path_or_feat, dict):
                s = path_or_feat.get('path') or ''
            else:
                s = str(path_or_feat or '')
        except Exception:
            s = str(path_or_feat or '')
        # basename without extension
        try:
            base = os.path.splitext(os.path.basename(s))[0]
        except Exception:
            base = s
        t = base.lower()
        # remove common separators and content in parentheses/brackets
        t = re.sub(r"\(.*?\)", ' ', t)
        t = re.sub(r"\[.*?\]", ' ', t)
        # remove common remix/version tags
        t = re.sub(r"\b(remix|rmx|version|edit|live|acoustic|radio|extended|instrumental|dub|mix|feat\.|feat|ft\.|ft)\b", ' ', t)
        # replace non-alphanumeric with space, collapse spaces
        t = re.sub(r"[^a-z0-9]+", ' ', t).strip()
        t = re.sub(r"\s+", ' ', t)
        return t

    def compute_transition_score(self, prev_feat: Optional[Dict], cand: Dict) -> float:
        if not prev_feat:
            return 0.5
        try:
            bpm_a = float(prev_feat.get('bpm') or 0.0)
            bpm_b = float(cand.get('bpm') or 0.0)
        except Exception:
            bpm_a = bpm_b = 0.0
        bpm_score = self._bpm_score(bpm_a, bpm_b)
        key_score = self._key_score(prev_feat.get('key') or {}, cand.get('key') or {})
        translation = self._translation_score(prev_feat.get('key') or {}, cand.get('key') or {})
        # combine with modest weights
        return float(max(0.0, min(1.0, 0.5 * bpm_score + 0.35 * key_score + 0.15 * translation)))

    def select_next_track(self,
                          state: PartyState,
                          candidates: List[Dict],
                          recent_played: Optional[List[str]] = None,
                          prev_track_path: Optional[str] = None,
                          bpm_tolerance: float = 15.0,
                          record_choice: bool = False,
                          advanced: Optional[Dict] = None,
                          target_override: Optional[float] = None) -> Tuple[Optional[Dict], List[Dict]]:
        """Return best candidate and a sorted list of scored candidates.

        candidates: list of feature dicts as produced by `musicscan.compute_features()`
        recent_played: list of paths recently played (novelty penalty)
        prev_track_path: optional path of currently playing track
        """
        recent = recent_played or []
        prev_feat = None
        if prev_track_path:
            for f in candidates:
                if f.get('path') == prev_track_path:
                    prev_feat = f
                    break

        # allow callers to override the target energy (e.g. user-set target)
        target_energy = float(target_override) if target_override is not None else self.engine.target_at(state.time_elapsed)

        # If caller provided an explicit target and candidate energies exist,
        # fall back to the closest available candidate energy (avoid impossible extremes)
        if target_override is not None and candidates:
            try:
                energies = [float(c.get('energy')) for c in candidates if c.get('energy') is not None]
                if energies:
                    nearest = min(energies, key=lambda e: abs(e - float(target_override)))
                    target_energy = float(nearest)
            except Exception:
                # if anything goes wrong, keep the original computed target_energy
                pass

        # first-pass BPM filtering (relative to prev track when available)
        filtered = []
        if prev_feat and prev_feat.get('bpm'):
            prev_bpm = float(prev_feat.get('bpm'))
            for c in candidates:
                try:
                    if c.get('bpm') is None:
                        continue
                    if abs(float(c.get('bpm')) - prev_bpm) <= bpm_tolerance:
                        filtered.append(c)
                except Exception:
                    continue
        else:
            filtered = list(candidates)

        if not filtered:
            filtered = list(candidates)

        # Apply advanced filters (genre filter, explicit toggle, tempo bias)
        adv = advanced or {}
        genre_filter = (adv.get('genre_filter') or '').strip().lower() if adv else ''
        hide_explicit = bool(adv.get('hide_explicit')) if adv and adv.get('hide_explicit') is not None else False
        tempo_bias = int(adv.get('tempo_bias') or 0) if adv else 0
        # Natural-language prompt parsing: prefer LLM parser when available,
        # otherwise fall back to a lightweight heuristic.
        nl_prompt = (adv.get('nl_prompt') or '') if adv else ''
        nl_prompt = (nl_prompt or '').strip()
        preferred_genres = []
        strict_genre = False
        if nl_prompt:
            parsed = None
            try:
                if parse_prompt is not None:
                    parsed = parse_prompt(nl_prompt)
            except Exception:
                parsed = None
            if not parsed:
                # ensure we at least have a minimal structure (vibe removed)
                parsed = {'tempo_bias': 0, 'preferred_genres': [], 'strict_genre': False, 'bpm': None}

            try:
                tempo_bias += int(parsed.get('tempo_bias') or 0)
            except Exception:
                pass
            try:
                preferred_genres = list(parsed.get('preferred_genres') or [])
            except Exception:
                preferred_genres = []
            try:
                strict_genre = bool(parsed.get('strict_genre'))
            except Exception:
                strict_genre = False
            try:
                bp = parsed.get('bpm')
                if bp:
                    try:
                        tempo_bias += int(max(-50, min(50, (int(bp) - 120) // 2)))
                    except Exception:
                        pass
            except Exception:
                pass
        # Vibe system removed: do not derive constraints from human-friendly vibe
        desired_energy = None
        min_dance = None
        max_dance = None
        prefer_tags = []

        # If a vibe defines a desired energy and the caller didn't explicitly
        # override the target, bias the selection target toward the vibe's
        # desired energy so the algorithm prefers tracks at that intensity.
        if desired_energy is not None and target_override is None:
            try:
                orig = float(target_energy)
                diff = abs(float(desired_energy) - orig)
                # If desired energy differs a lot from the engine target, fully
                # honor the vibe target (e.g., Dinner vs mid-party timeline).
                if diff >= 20:
                    target_energy = float(desired_energy)
                else:
                    # otherwise blend but bias toward the vibe
                    target_energy = float((orig * 0.2) + (float(desired_energy) * 0.8))
            except Exception:
                target_energy = float(desired_energy)

        # filter by preferred genres (derived from NL prompt or advanced settings)
        # merge explicit genre_filter into preferred_genres for compatibility
        try:
            if genre_filter:
                preferred_genres.insert(0, genre_filter)
        except Exception:
            pass

        if preferred_genres:
            # perform case-insensitive substring matching against candidate fields
            gfiltered = []
            for c in filtered:
                matched = False
                try:
                    cf = ''
                    if c.get('genre'):
                        cf = str(c.get('genre')).lower()
                    if not matched and c.get('genres'):
                        cf = cf + ' ' + ' '.join([str(x).lower() for x in (c.get('genres') or [])])
                    # ignore 'vibes' tags for matching (vibe system removed)
                    if not matched and c.get('cluster'):
                        cf = cf + ' ' + str(c.get('cluster')).lower()
                    # also inspect filename/title for genre tokens
                    try:
                        cf = cf + ' ' + str(c.get('path') or '').lower()
                    except Exception:
                        pass
                    for pg in preferred_genres:
                        if pg and pg in cf:
                            matched = True
                            break
                except Exception:
                    matched = False
                if matched:
                    gfiltered.append(c)
            # if strict_genre requested and matches exist, use only those matches
            if strict_genre and gfiltered:
                filtered = gfiltered
            else:
                # otherwise, prefer matches by moving them to the front of the list
                if gfiltered:
                    matched_set = set([x.get('path') for x in gfiltered if x.get('path')])
                    filtered = sorted(filtered, key=lambda x: (0 if x.get('path') in matched_set else 1))

        # hide explicit tracks if requested
        if hide_explicit:
            filtered = [c for c in filtered if not (c.get('explicit') or c.get('is_explicit') or c.get('explicitly_labeled'))]

        # Singalong mode removed: no special singalong filtering here.

        # precompute avoid penalties for filtered candidates to avoid per-item DB queries
        try:
            avoid_map = memory.compute_avoid_scores(filtered)
        except Exception:
            avoid_map = {}

        # Avoid recommending the same canonical song/title that was played
        # recently (covers remixes/edits/versions). If recent canonical
        # titles exist, prefer candidates that do not match those canonicals.
        try:
            recent_paths = []
            if isinstance(recent, list):
                recent_paths.extend([p for p in recent if isinstance(p, str)])
            if prev_track_path and isinstance(prev_track_path, str):
                recent_paths.append(prev_track_path)
            recent_canon = set()
            for rp in recent_paths:
                try:
                    recent_canon.add(self._canonical_title(rp))
                except Exception:
                    continue
            if recent_canon:
                nondup = []
                for c in filtered:
                    try:
                        if self._canonical_title(c) not in recent_canon:
                            nondup.append(c)
                    except Exception:
                        nondup.append(c)
                # If doing so would leave us with at least one candidate,
                # prefer the non-duplicates; otherwise keep original set.
                if nondup:
                    filtered = nondup
        except Exception:
            pass

        scored = []
        # directional need: how much energy we need to reach the target from current state
        try:
            energy_needed = float(target_energy) - float(state.energy_level or 0)
        except Exception:
            energy_needed = 0.0

        for c in filtered:
            c_energy = float(c.get('energy') or 0)
            energy_match = max(0.0, 1.0 - abs(c_energy - target_energy) / 30.0)

            fam = float(c.get('familiarity') or 50) / 100.0
            familiarity_fit = state.familiarity_bias * fam + (1.0 - state.familiarity_bias) * (1.0 - fam)

            trans = self.compute_transition_score(prev_feat, c)
            # recommend a transition type (heuristic)
            try:
                if self.transitioner is not None:
                    rec = self.transitioner.recommend(prev_feat, c, state)
                else:
                    rec = {'type': 'fade_reentry', 'confidence': 0.4, 'reason': 'no transition engine'}
            except Exception:
                rec = {'type': 'fade_reentry', 'confidence': 0.2, 'reason': 'error'}

            novelty = 1.0 if (c.get('path') not in recent) else 0.2

            # context relevance: prefer tracks whose metadata matches `crowd_type`
            context = 1.0
            try:
                if state.crowd_type and state.crowd_type != 'mixed':
                    cf_ctx = ''
                    try:
                        cf_ctx += ' ' + str(c.get('genre') or '')
                    except Exception:
                        pass
                    try:
                        cf_ctx += ' ' + ' '.join([str(x) for x in (c.get('genres') or [])])
                    except Exception:
                        pass
                    try:
                        cf_ctx += ' ' + str(c.get('cluster') or '')
                    except Exception:
                        pass
                    try:
                        cf_ctx += ' ' + str(c.get('path') or '')
                    except Exception:
                        pass
                    context = 1.0 if state.crowd_type in cf_ctx.lower() else 0.6
            except Exception:
                context = 1.0

            try:
                dance = float(c.get('danceability') or 0)
            except Exception:
                dance = 0.0

            # NL-derived preferred-genre/tag match check (no 'vibes' dependency)
            prefer_match = False
            try:
                cf = ''
                try:
                    cf = cf + ' ' + str(c.get('genre') or '')
                except Exception:
                    pass
                try:
                    cf = cf + ' ' + ' '.join([str(x) for x in (c.get('genres') or [])])
                except Exception:
                    pass
                try:
                    cf = cf + ' ' + str(c.get('cluster') or '')
                except Exception:
                    pass
                try:
                    cf = cf + ' ' + str(c.get('path') or '')
                except Exception:
                    pass
                cf = cf.lower()
                for pg in preferred_genres:
                    if pg and pg in cf:
                        prefer_match = True
                        break
            except Exception:
                prefer_match = False

            # dynamic weights: prioritize energy match more strongly
            energy_w = 0.5
            fam_w = 0.15
            trans_w = 0.2
            novelty_w = 0.075
            context_w = 0.075
            # if user provided an NL prompt, prefer semantic matching over mere familiarity
            try:
                if nl_prompt:
                    fam_w = max(0.03, fam_w * 0.25)
                    novelty_w = min(0.2, novelty_w * 1.5)
                    # make energy slightly more important when user requested a vibe
                    energy_w += 0.05
            except Exception:
                pass
            # if a vibe defines a desired energy, boost importance of energy match
            if desired_energy is not None:
                energy_w += 0.1
                trans_w = max(0.0, trans_w - 0.05)

            # if no previous track, lower transition importance and boost energy
            if prev_feat is None:
                trans_w = 0.15
                energy_w += 0.05

            # directional bonus: small reward for tracks that move us toward target
            direction_bonus = 0.0
            try:
                if energy_needed > 0 and c_energy > state.energy_level:
                    frac = min(1.0, (c_energy - state.energy_level) / (energy_needed if energy_needed > 0 else 1.0))
                    direction_bonus = frac * 0.1
                elif energy_needed < 0 and c_energy < state.energy_level:
                    frac = min(1.0, (state.energy_level - c_energy) / (-energy_needed))
                    direction_bonus = frac * 0.1
            except Exception:
                direction_bonus = 0.0

            score = (energy_w * energy_match + fam_w * familiarity_fit + trans_w * trans + novelty_w * novelty + context_w * context)
            score = float(max(0.0, min(1.0, score))) + float(direction_bonus)

            # tempo bias: small boost for faster/slower tracks depending on user preference
            tempo_boost = 0.0
            try:
                c_bpm = float(c.get('bpm') or 0.0)
                # amplify tempo preference when NL prompt requested BPM/genre
                tempo_mult = 1.0
                try:
                    if nl_prompt:
                        tempo_mult = 2.0
                        # if strict_genre (e.g., 'rave'), be more aggressive
                        if strict_genre:
                            tempo_mult = 2.5
                except Exception:
                    tempo_mult = 1.0
                if tempo_bias and tempo_bias > 0 and c_bpm >= 120:
                    tempo_boost = 0.05 * (tempo_bias / 100.0) * tempo_mult
                elif tempo_bias and tempo_bias < 0 and c_bpm <= 100:
                    tempo_boost = 0.05 * (abs(tempo_bias) / 100.0) * tempo_mult
            except Exception:
                tempo_boost = 0.0
            score = score + float(tempo_boost)

            # apply explicit prefer-tag/genrish boost (from NL prompt)
            try:
                if prefer_match:
                    score = score + 0.25
            except Exception:
                pass

            # (vibe system removed) no additional 'vibe' influence applied here

            # memory-based adjustments (use precomputed batch results)
            try:
                avoid_penalty = avoid_map.get(c.get('path')) if avoid_map is not None else 0.0
            except Exception:
                avoid_penalty = 0.0
            try:
                pref_boost = memory.get_preference_boost(c, state)
            except Exception:
                pref_boost = 0.0

            # apply: penalize by portion of avoid_penalty, add preference boost
            score = score - (avoid_penalty * 0.6) + float(pref_boost)
            # small extra bump for NL-preferred matches already applied above via score
            score = float(max(0.0, min(1.5, score)))

            # diversity penalty: if we have a previous track, discourage
            # recommending very similar tracks (remixes/edits/etc.). The
            # penalty scales with a similarity metric (0..1) and a
            # configurable `diversify_strength` (default 0.45).
            try:
                if prev_feat is not None:
                    # configurable strength via advanced options
                    try:
                        diversify_strength = float(adv.get('diversify_strength')) if adv and adv.get('diversify_strength') is not None else 0.45
                    except Exception:
                        diversify_strength = 0.45
                    if diversify_strength > 0:
                        try:
                            # immediate exact-path avoid
                            if c.get('path') and prev_track_path and c.get('path') == prev_track_path:
                                score = float(max(0.0, score * 0.05))
                            else:
                                # compute similarity components
                                sim = 0.0
                                try:
                                    # canonical title high-similarity (covers remixes)
                                    if self._canonical_title(prev_feat) == self._canonical_title(c):
                                        sim = max(sim, 0.95)
                                except Exception:
                                    pass
                                try:
                                    prev_e = float(prev_feat.get('energy') or 0)
                                    e_sim = max(0.0, 1.0 - abs(c_energy - prev_e) / 30.0)
                                except Exception:
                                    e_sim = 0.0
                                try:
                                    prev_bpm = float(prev_feat.get('bpm') or 0)
                                    c_bpm = float(c.get('bpm') or 0)
                                    b_sim = max(0.0, 1.0 - abs(c_bpm - prev_bpm) / 30.0)
                                except Exception:
                                    b_sim = 0.0
                                try:
                                    k_sim = self._key_score(prev_feat.get('key') or {}, c.get('key') or {})
                                except Exception:
                                    k_sim = 0.0
                                try:
                                    prev_genres = [str(x).lower() for x in (prev_feat.get('genres') or [])]
                                    c_genres = [str(x).lower() for x in (c.get('genres') or [])]
                                    tag_sim = 1.0 if any(g in prev_genres for g in c_genres) else 0.0
                                except Exception:
                                    tag_sim = 0.0
                                # weighted similarity
                                sim = max(sim, 0.35 * e_sim + 0.30 * b_sim + 0.20 * k_sim + 0.15 * tag_sim)
                                sim = float(max(0.0, min(1.0, sim)))
                                penalty = sim * float(diversify_strength)
                                score = float(max(0.0, score * (1.0 - penalty)))
                        except Exception:
                            pass
            except Exception:
                pass

            scored.append({'track': c, 'score': float(score), 'components': {
                'energy_match': float(energy_match),
                'familiarity_fit': float(familiarity_fit),
                'transition': float(trans),
                'novelty': float(novelty),
                'context': float(context),
                'direction_bonus': float(direction_bonus),
                'transition_choice': rec,
                'memory_avoid': float(avoid_penalty),
                'preference_boost': float(pref_boost),
                'tempo_bias_boost': float(tempo_boost)
            }})

        scored.sort(key=lambda x: x['score'], reverse=True)

        # Pick the highest-scoring candidate that is NOT the exact same
        # file as the currently playing one. This makes it impossible to
        # return the same file path as `prev_track_path`.
        best = None
        if scored:
            for entry in scored:
                try:
                    cand = entry.get('track')
                    if prev_track_path and cand and cand.get('path') == prev_track_path:
                        # skip exact same file
                        continue
                    best = cand
                    break
                except Exception:
                    continue

        # Fallback: ensure we always return something in the queue. If the
        # selector filtered everything out (or scoring failed), pick the
        # candidate closest to the target energy and matching basic vibe
        # constraints where possible. Always exclude the previous exact file.
        if best is None and candidates:
            fallback_best = None
            fallback_score = None
            for c in candidates:
                try:
                    # never consider exact same file as previous
                    if prev_track_path and c.get('path') == prev_track_path:
                        continue
                    ce = float(c.get('energy') or 0)
                except Exception:
                    ce = 0.0
                # energy proximity (higher is better)
                try:
                    energy_prox = 1.0 - (abs(ce - float(target_energy)) / 100.0)
                except Exception:
                    energy_prox = 0.0
                try:
                    dance = float(c.get('danceability') or 0) / 100.0
                except Exception:
                    dance = 0.0
                # prefer tracks that match advanced `prefer_tags` (no 'vibes')
                try:
                    tag_match = 0.0
                    if prefer_tags:
                        cf = ''
                        try:
                            cf += ' ' + str(c.get('genre') or '')
                        except Exception:
                            pass
                        try:
                            cf += ' ' + ' '.join([str(x) for x in (c.get('genres') or [])])
                        except Exception:
                            pass
                        try:
                            cf += ' ' + str(c.get('cluster') or '')
                        except Exception:
                            pass
                        try:
                            cf += ' ' + str(c.get('path') or '')
                        except Exception:
                            pass
                        cf = cf.lower()
                        for pt in prefer_tags:
                            if pt and pt in cf:
                                tag_match = 1.0
                                break
                except Exception:
                    tag_match = 0.0
                try:
                    m = memory.get_preference_boost(c, state)
                except Exception:
                    m = 0.0
                sc = energy_prox * 0.6 + dance * 0.2 + tag_match * 0.15 + float(m) * 0.05
                if fallback_score is None or sc > fallback_score:
                    fallback_best = c
                    fallback_score = sc
            best = fallback_best
            if best is not None:
                scored = [{'track': best, 'score': float(fallback_score or 0.0), 'components': {'fallback': True}}]

        # optionally record that we're about to play the chosen track
        # optionally record that we're about to play the chosen track
        if record_choice and best is not None:
            try:
                mp = best.get('path')
                energy = int(best.get('energy') or 0)
                # derive moment_type from time only (no vibes)
                hour = time.localtime().tm_hour
                moment = 'late-night' if (hour >= 22 or hour <= 4) else 'normal'
                memory.record_play(mp, played_at=time.time(), duration=None, skipped=False, moment_type=moment, artist_cluster=best.get('cluster'), energy=energy)
                # also update preferences if applicable (play event) — do not pass 'vibes'
                memory.update_preferences_on_event({'type': 'play', 'path': mp, 'energy': energy, 'played_at': time.time()})
            except Exception:
                pass

        return best, scored


if __name__ == '__main__':
    # small demo when run directly
    import os
    from pprint import pprint

    os.makedirs('test_wavs', exist_ok=True)
    # try to reuse helper if present
    try:
        from test_scan_runner import make_sine
        p1 = make_sine('test_wavs/party_demo_a.wav', ramp=True, freq=220)
        p2 = make_sine('test_wavs/party_demo_b.wav', drop=True, freq=440)
    except Exception:
        p1 = None
        p2 = None

    if p1 and p2:
        f1 = musicscan.compute_features(p1)
        f2 = musicscan.compute_features(p2)
        engine = EnergyEngine()
        selector = TrackSelector(engine)
        state = PartyState(energy_level=65, trajectory='rising', time_elapsed=90, familiarity_bias=0.7, crowd_type='mixed', prev_track_path=p1)
        best, scored = selector.select_next_track(state, [f1, f2], recent_played=[p1], prev_track_path=p1)
        print('Target energy:', engine.target_at(state.time_elapsed))
        pprint(scored)
        print('Best:', best.get('path'))
