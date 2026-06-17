import os
import sqlite3
import time
import json
from typing import Optional, List, Dict

DB_FILE = os.path.join(os.path.dirname(__file__), 'musicscan.db')


def init_db(db_path: Optional[str] = None):
    path = db_path or DB_FILE
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    try:
        cur.execute('''
        CREATE TABLE IF NOT EXISTS play_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT,
            played_at REAL,
            duration REAL,
            skipped INTEGER,
            moment_type TEXT,
            artist_cluster TEXT,
            energy INTEGER
        )
        ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_ph_path ON play_history(path)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_ph_time ON play_history(played_at)')
    except Exception:
        pass

    try:
        cur.execute('''
        CREATE TABLE IF NOT EXISTS prefs (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        ''')
    except Exception:
        pass
    conn.commit()
    return conn


def record_play(path: str, played_at: Optional[float] = None, duration: Optional[float] = None,
                skipped: bool = False, moment_type: Optional[str] = None,
                artist_cluster: Optional[str] = None, energy: Optional[int] = None,
                db_path: Optional[str] = None):
    conn = init_db(db_path)
    cur = conn.cursor()
    ts = float(played_at or time.time())
    try:
        cur.execute('INSERT INTO play_history(path, played_at, duration, skipped, moment_type, artist_cluster, energy) VALUES (?,?,?,?,?,?,?)',
                    (path, ts, duration or 0.0, 1 if skipped else 0, moment_type, artist_cluster, energy))
        conn.commit()
    except Exception:
        pass


def recent_plays(within_seconds: int = 7200, limit: int = 50, db_path: Optional[str] = None) -> List[Dict]:
    conn = init_db(db_path)
    cur = conn.cursor()
    now = time.time()
    cutoff = now - within_seconds
    try:
        cur.execute('SELECT path, played_at, skipped, moment_type, artist_cluster, energy FROM play_history WHERE played_at >= ? ORDER BY played_at DESC LIMIT ?', (cutoff, limit))
        rows = cur.fetchall()
        return [{'path': r[0], 'played_at': r[1], 'skipped': bool(r[2]), 'moment_type': r[3], 'artist_cluster': r[4], 'energy': r[5]} for r in rows]
    except Exception:
        return []


def was_played_recently(path: str, within_seconds: int = 7200, db_path: Optional[str] = None) -> bool:
    conn = init_db(db_path)
    cur = conn.cursor()
    now = time.time()
    cutoff = now - within_seconds
    try:
        cur.execute('SELECT played_at FROM play_history WHERE path = ? ORDER BY played_at DESC LIMIT 1', (path,))
        r = cur.fetchone()
        if not r:
            return False
        return (now - float(r[0])) <= within_seconds
    except Exception:
        return False


def count_recent_artist(cluster: str, within_seconds: int = 7200, db_path: Optional[str] = None) -> int:
    if not cluster:
        return 0
    conn = init_db(db_path)
    cur = conn.cursor()
    now = time.time()
    cutoff = now - within_seconds
    try:
        cur.execute('SELECT COUNT(*) FROM play_history WHERE artist_cluster = ? AND played_at >= ?', (cluster, cutoff))
        r = cur.fetchone()
        return int(r[0]) if r else 0
    except Exception:
        return 0


def _get_last_play_time(path: str, db_path: Optional[str] = None) -> Optional[float]:
    conn = init_db(db_path)
    cur = conn.cursor()
    try:
        cur.execute('SELECT played_at FROM play_history WHERE path = ? ORDER BY played_at DESC LIMIT 1', (path,))
        r = cur.fetchone()
        return float(r[0]) if r else None
    except Exception:
        return None


def compute_avoid_score(candidate: Dict, state: Optional[Dict] = None, within_seconds: int = 7200, db_path: Optional[str] = None) -> float:
    """Return a penalty in range 0.0..0.9 where higher means avoid this track."""
    try:
        now = time.time()
        path = candidate.get('path')
        penalty = 0.0

        last = _get_last_play_time(path, db_path)
        if last is not None:
            delta = now - last
            if delta <= within_seconds:
                path_pen = 0.6 * (1.0 - (delta / within_seconds))
                penalty += path_pen

        # artist/cluster penalty
        cluster = candidate.get('cluster') or candidate.get('artist_cluster') or None
        if cluster:
            ccount = count_recent_artist(cluster, within_seconds, db_path)
            if ccount >= 2:
                penalty += 0.4
            elif ccount == 1:
                penalty += 0.2

        # moment type penalty: derive moment_type from vibes
        vibes = candidate.get('vibes') or []
        cand_moment = 'late-night' if 'late-night' in vibes else 'normal'
        # count similar moment plays
        recent = recent_plays(within_seconds, 50, db_path)
        same_moment_count = sum(1 for r in recent if (r.get('moment_type') or 'normal') == cand_moment)
        if same_moment_count >= 3:
            penalty += 0.25

        return max(0.0, min(0.9, penalty))
    except Exception:
        return 0.0


def compute_avoid_scores(candidates: List[Dict], within_seconds: int = 7200, db_path: Optional[str] = None) -> Dict[str, float]:
    """Compute avoid penalties for a list of candidates in a single batch.

    Returns dict path -> penalty (0.0..0.9)
    """
    try:
        conn = init_db(db_path)
        cur = conn.cursor()
        now = time.time()
        cutoff = now - within_seconds

        paths = [c.get('path') for c in candidates if c.get('path')]
        path_set = list(dict.fromkeys(paths))
        penalties = {p: 0.0 for p in path_set}

        if path_set:
            # last played time per path (use MAX)
            placeholders = ','.join(['?'] * len(path_set))
            try:
                cur.execute(f"SELECT path, MAX(played_at) FROM play_history WHERE path IN ({placeholders}) GROUP BY path", path_set)
                for row in cur.fetchall():
                    pth, last_at = row[0], row[1]
                    if last_at is None:
                        continue
                    delta = now - float(last_at)
                    if delta <= within_seconds:
                        path_pen = 0.6 * (1.0 - (delta / within_seconds))
                        penalties[pth] += path_pen
            except Exception:
                pass

        # artist cluster penalties (count recent plays per cluster)
        clusters = [c.get('cluster') or c.get('artist_cluster') for c in candidates]
        clusters = [x for x in clusters if x]
        cluster_set = list(dict.fromkeys(clusters))
        cluster_counts = {}
        if cluster_set:
            placeholders = ','.join(['?'] * len(cluster_set))
            try:
                cur.execute(f"SELECT artist_cluster, COUNT(*) FROM play_history WHERE artist_cluster IN ({placeholders}) AND played_at >= ? GROUP BY artist_cluster", cluster_set + [cutoff])
                for row in cur.fetchall():
                    cluster_counts[row[0]] = int(row[1])
            except Exception:
                pass

        # moment_type counts recent
        moment_counts = {}
        try:
            cur.execute('SELECT moment_type, COUNT(*) FROM play_history WHERE played_at >= ? GROUP BY moment_type', (cutoff,))
            for row in cur.fetchall():
                moment_counts[row[0] or 'normal'] = int(row[1])
        except Exception:
            pass

        # compute final penalties for each candidate
        result = {}
        for c in candidates:
            p = c.get('path')
            pen = 0.0
            if p and p in penalties:
                pen += penalties[p]

            cl = c.get('cluster') or c.get('artist_cluster')
            if cl:
                cnt = cluster_counts.get(cl, 0)
                if cnt >= 2:
                    pen += 0.4
                elif cnt == 1:
                    pen += 0.2

            vibes = c.get('vibes') or []
            cand_moment = 'late-night' if 'late-night' in vibes else 'normal'
            same_moment_count = moment_counts.get(cand_moment, 0)
            if same_moment_count >= 3:
                pen += 0.25

            result[p] = max(0.0, min(0.9, pen))

        return result
    except Exception:
        # fallback to per-item compute
        out = {}
        for c in candidates:
            out[c.get('path')] = compute_avoid_score(c, state=None, within_seconds=within_seconds, db_path=db_path)
        return out


def _get_pref(key: str, db_path: Optional[str] = None) -> Optional[Dict]:
    conn = init_db(db_path)
    cur = conn.cursor()
    try:
        cur.execute('SELECT value FROM prefs WHERE key = ?', (key,))
        r = cur.fetchone()
        if not r:
            return None
        return json.loads(r[0])
    except Exception:
        return None


def _set_pref(key: str, obj: Dict, db_path: Optional[str] = None):
    conn = init_db(db_path)
    cur = conn.cursor()
    try:
        cur.execute('REPLACE INTO prefs(key, value) VALUES (?,?)', (key, json.dumps(obj)))
        conn.commit()
    except Exception:
        pass


def update_preferences_on_event(event: Dict, db_path: Optional[str] = None):
    """Event example: {'type':'skip'|'play', 'path':..., 'vibes':[], 'energy':int, 'played_at':ts}
    """
    try:
        etype = event.get('type')
        vibes = event.get('vibes') or []
        energy = int(event.get('energy') or 0)
        ts = float(event.get('played_at') or time.time())
        hour = time.localtime(ts).tm_hour
        late = (hour >= 22 or hour <= 4)

        if etype == 'skip' and late and energy < 40:
            key = 'skip_low_energy_late'
            cur = _get_pref(key, db_path) or {'count': 0}
            cur['count'] = cur.get('count', 0) + 1
            _set_pref(key, cur, db_path)

        if etype == 'play' and late and 'nostalgic' in vibes:
            key = 'like_nostalgia_late'
            cur = _get_pref(key, db_path) or {'count': 0}
            cur['count'] = cur.get('count', 0) + 1
            _set_pref(key, cur, db_path)
    except Exception:
        pass


def get_preference_boost(candidate: Dict, state: Optional[Dict] = None, db_path: Optional[str] = None) -> float:
    """Return a small positive boost (0..0.15) or negative (penalty) based on learned prefs."""
    try:
        ts = time.time()
        hour = time.localtime(ts).tm_hour
        late = (hour >= 22 or hour <= 4)
        boost = 0.0

        if late:
            p = _get_pref('like_nostalgia_late', db_path) or {'count': 0}
            cnt = int(p.get('count', 0))
            if cnt > 0 and 'nostalgic' in (candidate.get('vibes') or []):
                boost += min(0.12, 0.02 + cnt * 0.01)

            p2 = _get_pref('skip_low_energy_late', db_path) or {'count': 0}
            cnt2 = int(p2.get('count', 0))
            if cnt2 > 0 and (candidate.get('energy') or 0) < 45:
                boost -= min(0.15, 0.02 + cnt2 * 0.02)

        return float(max(-0.3, min(0.3, boost)))
    except Exception:
        return 0.0
