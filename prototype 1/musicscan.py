#!/usr/bin/env python
"""musicscan.py

Reads a JSON list of audio file paths from stdin and outputs JSON with
estimated features per track:
 - bpm (tempo)
 - key (tonic + mode)
 - energy (0-100)
 - danceability (0-100, heuristic)
 - loudness (dB)
 - genre cluster id

This script uses `librosa` and `scikit-learn`. If they are not installed
the script exits with a helpful message.
"""
import sys
import json
import os
import math
import sqlite3
import time
from multiprocessing import Pool

try:
    import numpy as np
    import librosa
    from sklearn.cluster import KMeans
except Exception as e:
    err = {"error": "missing_dependency", "message": str(e),
           "hint": "pip install librosa numpy scikit-learn soundfile"}
    print(json.dumps(err))
    sys.exit(3)
try:
    import soundfile as sf
except Exception:
    sf = None

# Allow optional full-track analysis via environment variable. Set to '1' to load full track.
FULL_TRACK = os.environ.get('MUSCISCAN_FULL', '0') in ('1', 'true', 'True')


KEYS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
# Krumhansl major/minor profiles
MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])


def detect_key(chroma_mean):
    """Estimate key tonic and mode by correlating chroma with major/minor profiles."""
    best = (None, None, -1)
    for i in range(12):
        # rotate profiles
        maj = np.roll(MAJOR_PROFILE, i)
        minp = np.roll(MINOR_PROFILE, i)
        maj_corr = np.corrcoef(chroma_mean, maj)[0,1]
        min_corr = np.corrcoef(chroma_mean, minp)[0,1]
        if not np.isfinite(maj_corr): maj_corr = -10
        if not np.isfinite(min_corr): min_corr = -10
        if maj_corr > best[2]: best = (KEYS[i], 'major', maj_corr)
        if min_corr > best[2]: best = (KEYS[i], 'minor', min_corr)
    return {'tonic': best[0], 'mode': best[1]}


def normalize_bpm(tempo):
    """Normalize tempo into a reasonable range (70-180) by doubling/halving.
    Return normalized tempo as float. Keep 0 as-is.
    """
    try:
        t = float(tempo)
    except Exception:
        return 0.0
    if t <= 0:
        return 0.0
    # Apply simple half/double time normalization but limit iterations
    # to avoid runaway doubling/halving on pathological tempos.
    if t < 90:
        t *= 2.0
    if t > 180:
        t /= 2.0
    # iterative adjust with a safety limit
    iter_lim = 6
    it = 0
    while t < 70.0 and it < iter_lim:
        t *= 2.0
        it += 1
    it = 0
    while t > 180.0 and it < iter_lim:
        t /= 2.0
        it += 1
    # hard clamp to a reasonable range to avoid unrealistic doubling
    t = float(np.clip(t, 60.0, 180.0))
    return t


def assign_vibes(feats):
    """Assign human-friendly 'vibe' tags using heuristics on extracted features.

    Returns a list of short tags such as: late-night, nostalgic, background,
    hype. These are intentionally heuristic and conservative.
    """
    vibes = []
    try:
        energy = int(feats.get('energy') or 0)
        dance = int(feats.get('danceability') or 0)
        loud = float(feats.get('loudness_db') or -999.0)
        bpm = float(feats.get('bpm') or 0.0)
        centroid = float(feats.get('centroid_pct') or 0.0)
        delta = int(feats.get('energy_delta') or 0)
    except Exception:
        return vibes

    # Hype: loud, high energy and danceability
    if energy >= 70 and dance >= 60 and loud > -18:
        vibes.append('hype')

    # Late-night: low energy, quiet, slower tempo
    if energy <= 40 and loud < -14 and bpm <= 110:
        vibes.append('late-night')

    # (singalong tag removed)

    # Nostalgic: darker timbre (low centroid) and moderate energy
    if centroid < 0.25 and 30 <= energy <= 70:
        vibes.append('nostalgic')

    # Background: low energy and low danceability
    if energy < 35 and dance < 45:
        vibes.append('background')

    # Small build/drop indicator: rising energy -> good for builds
    if delta >= 20 and energy >= 40:
        vibes.append('build-friendly')
    if delta <= -20 and energy >= 40:
        vibes.append('drop-heavy')

    return vibes


# removed unused rms_db; using librosa.amplitude_to_db directly


FLAT_TO_SHARP = {
    'DB': 'C#', 'EB': 'D#', 'GB': 'F#', 'AB': 'G#', 'BB': 'A#',
    'CB': 'B', 'FB': 'E'
}


def normalize_tonic(t):
    if not t: return None
    t = str(t).upper()
    # map flats to sharps (Db -> C#)
    if len(t) >= 2 and t[1] == 'B':
        return FLAT_TO_SHARP.get(t, t)
    return t


def compute_compatibility_matrix(feats_list, feature_matrix, clusters):
    """Compute compact compatibility mapping using k-nearest neighbors.
    Returns a dict mapping path -> list of neighbor compatibility objects.
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        return {"error": "sklearn missing"}

    n_feats = len(feature_matrix)
    if n_feats < 2:
        return {}

    X = np.array(feature_matrix)
    k = min(10, n_feats - 1)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)

    # build path->cluster mapping for O(1) lookup
    path_to_cluster = {}
    for lab, members in clusters.items():
        for p in members:
            path_to_cluster[p] = lab

    def bpm_from_norm(norm):
        return norm * (180.0 - 70.0) + 70.0

    key_to_idx = {k:i for i,k in enumerate(KEYS)}
    compat = {}

    for i in range(n_feats):
        a = feats_list[i]
        row = []
        for jpos in range(1, indices.shape[1]):
            j = indices[i, jpos]
            b = feats_list[j]
            try:
                bpm_a = bpm_from_norm(feature_matrix[i][0])
                bpm_b = bpm_from_norm(feature_matrix[j][0])
            except Exception:
                bpm_a = (a.get('bpm') or 0.0)
                bpm_b = (b.get('bpm') or 0.0)
            bpm_diff = abs(bpm_a - bpm_b)
            bpm_score = max(0.0, 1.0 - (bpm_diff / 30.0))

            # key score
            try:
                ta = a.get('key', {}) or {}
                tb = b.get('key', {}) or {}
                ia = key_to_idx.get(ta.get('tonic'))
                ib = key_to_idx.get(tb.get('tonic'))
                if ia is None or ib is None:
                    key_score = 0.5
                else:
                    sem = abs(ia - ib)
                    sem = min(sem, 12 - sem)
                    key_score = max(0.0, 1.0 - (sem / 6.0))
                    if ta.get('mode') == tb.get('mode'):
                        key_score = min(1.0, key_score + 0.15)
            except Exception:
                key_score = 0.5

            # energy similarity (directional-aware): compare overall energy but
            # also prefer transitions that build (a.end -> b.start) when appropriate
            try:
                ea = (a.get('energy') or 0) / 100.0
                eb = (b.get('energy') or 0) / 100.0
                energy_similarity = max(0.0, 1.0 - abs(ea - eb))
            except Exception:
                energy_similarity = 0.5

            try:
                a_end = (a.get('energy_end') or 0) / 100.0
                b_start = (b.get('energy_start') or 0) / 100.0
                transition = b_start - a_end
                # scale transition into -1..1 roughly; small shifts rewarded,
                # big shifts capped
                if transition >= 0:
                    trans_score = min(1.0, transition / 0.5)
                else:
                    trans_score = -min(1.0, abs(transition) / 0.5) * 0.7
            except Exception:
                trans_score = 0.0

            # combine similarity and directional transition: builds get bonus,
            # drops are allowed but penalized more
            if trans_score >= 0:
                energy_score = max(0.0, min(1.0, energy_similarity + trans_score * 0.5))
            else:
                energy_score = max(0.0, min(1.0, energy_similarity + trans_score * 0.7))

            # cluster bonus via path_to_cluster
            try:
                ca = path_to_cluster.get(a.get('path'))
                cb = path_to_cluster.get(b.get('path'))
                cluster_score = 1.0 if (ca is not None and ca == cb) else 0.5
            except Exception:
                cluster_score = 0.5

            # translation (transpose) scoring: how easily one track can be pitch-shifted
            # to match the other's tonic. Small shifts are preferred.
            try:
                ta = a.get('key', {}) or {}
                tb = b.get('key', {}) or {}
                ia = key_to_idx.get(ta.get('tonic'))
                ib = key_to_idx.get(tb.get('tonic'))
                if ia is None or ib is None:
                    translation_score = 0.5
                    transpose_semitones = None
                else:
                    # signed minimal semitone difference in range -6..+6
                    delta = (ib - ia) % 12
                    if delta > 6:
                        delta -= 12
                    # cap translation to -6..6 semitones explicitly
                    transpose_semitones = int(np.clip(int(delta), -6, 6))
                    abs_shift = abs(transpose_semitones)
                    # base score: full for 0, decreasing to 0 at 3 semitones
                    translation_score = max(0.0, 1.0 - (abs_shift / 3.0))
                    # penalize mode changes slightly
                    if ta.get('mode') != tb.get('mode'):
                        translation_score *= 0.8
            except Exception:
                translation_score = 0.5
                transpose_semitones = None

            comp = (0.35 * bpm_score + 0.10 * key_score + 0.15 * translation_score
                    + 0.25 * energy_score + 0.15 * cluster_score)
            comp_score = int(round(max(0.0, min(1.0, comp)) * 100))

            row.append({
                'path': b.get('path'),
                'score': comp_score,
                'components': {
                    'bpm_a': bpm_a, 'bpm_b': bpm_b, 'bpm_score': round(bpm_score, 3),
                    'key_score': round(key_score, 3),
                    'translation_score': round(translation_score, 3),
                    'transpose_semitones': transpose_semitones,
                    'energy_score': round(energy_score, 3),
                    'cluster_score': round(cluster_score, 3)
                }
            })
        compat[a.get('path')] = row

    return compat


DB_FILE = os.path.join(os.path.dirname(__file__), 'musicscan.db')


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS tracks (
        path TEXT PRIMARY KEY,
        mtime REAL,
        size INTEGER,
        features TEXT,
        vibes TEXT,
        updated_at REAL
    )
    ''')
    # robustly ensure expected columns exist (useful for schema evolution)
    try:
        cur.execute("PRAGMA table_info(tracks)")
        existing = {r[1] for r in cur.fetchall()}
    except Exception:
        existing = set()

    desired = {'vibes': 'TEXT', 'bpm': 'REAL', 'energy': 'INTEGER', 'mfcc': 'TEXT', 'familiarity': 'INTEGER'}
    # vocal_coverage stores a float (0.0-1.0) indicating estimated proportion
    # of frames that contain strong harmonic (vocal-like) energy. Add column
    # to support vocal-coverage features in the UI and harvesting.
    desired['vocal_coverage'] = 'REAL'
    for name, typ in desired.items():
        if name not in existing:
            try:
                cur.execute(f'ALTER TABLE tracks ADD COLUMN {name} {typ}')
            except Exception:
                # best-effort: ignore failures (older sqlite, read-only DB, etc.)
                pass
    conn.commit()
    return conn


def ensure_compat_table(conn):
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS compatibility (
        track_path TEXT,
        neighbor_path TEXT,
        score INTEGER,
        components TEXT,
        PRIMARY KEY(track_path, neighbor_path)
    )
    ''')
    try:
        cur.execute('CREATE INDEX IF NOT EXISTS idx_compat_track ON compatibility(track_path)')
    except Exception:
        pass
    try:
        # index on energy for faster queries
        cur.execute("CREATE INDEX IF NOT EXISTS idx_energy ON tracks(json_extract(features, '$.energy'))")
    except Exception:
        pass
    try:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_bpm ON tracks(json_extract(features, '$.bpm'))")
    except Exception:
        pass
    conn.commit()


def cleanup_db(conn):
    # remove rows for files that no longer exist
    cur = conn.cursor()
    cur.execute('SELECT path FROM tracks')
    rows = cur.fetchall()
    removed = 0
    for (p,) in rows:
        if not os.path.exists(p):
            cur.execute('DELETE FROM tracks WHERE path = ?', (p,))
            removed += 1
    if removed:
        conn.commit()


def get_cached(conn, path, mtime, size):
    cur = conn.cursor()
    cur.execute('SELECT mtime, size, features FROM tracks WHERE path = ?', (path,))
    r = cur.fetchone()
    if not r:
        return None
    db_mtime, db_size, db_features = r
    try:
        if float(db_mtime) == float(mtime) and int(db_size) == int(size):
            return json.loads(db_features)
    except Exception:
        return None
    return None


def upsert_features(conn, path, mtime, size, features):
    cur = conn.cursor()
    vibes_json = None
    try:
        vibes_json = json.dumps(features.get('vibes')) if features.get('vibes') is not None else None
    except Exception:
        vibes_json = None
    # include optional columns (bpm, energy, mfcc, familiarity) dynamically
    try:
        cur.execute("PRAGMA table_info(tracks)")
        cols = [r[1] for r in cur.fetchall()]
    except Exception:
        cols = []

    bpm_val = None
    energy_val = None
    mfcc_json = None
    familiarity_val = None
    vocal_val = None
    try:
        bpm_val = float(features.get('bpm')) if features.get('bpm') is not None else None
    except Exception:
        bpm_val = None
    try:
        energy_val = int(features.get('energy')) if features.get('energy') is not None else None
    except Exception:
        energy_val = None
    try:
        vocal_val = float(features.get('vocal_coverage')) if features.get('vocal_coverage') is not None else None
    except Exception:
        vocal_val = None
    try:
        if isinstance(features.get('mfcc_mean'), list):
            mfcc_json = json.dumps(features.get('mfcc_mean'))
    except Exception:
        mfcc_json = None
    try:
        if features.get('familiarity') is not None:
            familiarity_val = int(features.get('familiarity'))
    except Exception:
        familiarity_val = None

    base_cols = ['path', 'mtime', 'size', 'features', 'vibes', 'updated_at']
    vals = [path, float(mtime), int(size), json.dumps(features), vibes_json, time.time()]
    opt_cols = []
    if 'bpm' in cols:
        opt_cols.append('bpm'); vals.append(bpm_val)
    if 'energy' in cols:
        opt_cols.append('energy'); vals.append(energy_val)
    if 'mfcc' in cols:
        opt_cols.append('mfcc'); vals.append(mfcc_json)
    if 'familiarity' in cols:
        opt_cols.append('familiarity'); vals.append(familiarity_val)
    if 'vocal_coverage' in cols:
        opt_cols.append('vocal_coverage'); vals.append(vocal_val)

    all_cols = base_cols + opt_cols
    placeholders = ','.join(['?'] * len(all_cols))
    cols_sql = ','.join(all_cols)
    try:
        cur.execute(f'REPLACE INTO tracks({cols_sql}) VALUES ({placeholders})', tuple(vals))
        conn.commit()
    except Exception:
        try:
            # final fallback: minimal insert
            cur.execute('REPLACE INTO tracks(path, mtime, size, features, vibes, updated_at) VALUES (?,?,?,?,?,?)',
                        (path, float(mtime), int(size), json.dumps(features), vibes_json, time.time()))
            conn.commit()
        except Exception:
            pass


def compute_features(path):
    # For performance on large libraries, avoid loading entire long files unless
    # `FULL_TRACK` is set. Strategy: if file <= 120s or FULL_TRACK True, load
    # up to 120s (or full); otherwise sample two 60s segments (start + middle)
    # and concatenate. This gives representative content while bounding I/O.
    try:
        max_sample = 120.0
        seg = 60.0
        duration = None
        if sf is not None:
            try:
                info = sf.info(path)
                duration = float(info.frames) / float(info.samplerate)
            except Exception:
                duration = None

        if FULL_TRACK or duration is None or duration <= max_sample:
            # load full track if explicitly requested, otherwise cap at max_sample
            if FULL_TRACK:
                y, sr = librosa.load(path, sr=22050, mono=True)
            else:
                y, sr = librosa.load(path, sr=22050, mono=True, duration=max_sample)
        else:
            # load first seg seconds and a middle seg seconds to capture variety
            try:
                y1, sr = librosa.load(path, sr=22050, mono=True, offset=0.0, duration=seg)
                mid_off = max(0.0, (duration - seg) / 2.0)
                y2, sr2 = librosa.load(path, sr=22050, mono=True, offset=mid_off, duration=seg)
                if sr2 != sr:
                    y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr)
                if y1 is None:
                    y = y2
                elif y2 is None:
                    y = y1
                else:
                    y = np.concatenate([y1, y2])
            except Exception:
                # fallback to loading up to max_sample seconds
                y, sr = librosa.load(path, sr=22050, mono=True, duration=max_sample)
    except Exception as e:
        return {"error": "load_failed", "message": str(e)}

    # tempo
    try:
        tempo = float(librosa.beat.tempo(y=y, sr=sr).mean())
    except Exception:
        tempo = 0.0
    raw_tempo = float(tempo or 0.0)
    # tempo fallback: if initial estimate is 0 or implausible, try alternatives
    if raw_tempo == 0.0:
        try:
            # try beat_track to compute beats and derive BPM from intervals
            _, beats = librosa.beat.beat_track(y=y, sr=sr)
            if len(beats) > 1:
                # beats are frame indices; convert to times
                times = librosa.frames_to_time(beats, sr=sr)
                diffs = np.diff(times)
                if diffs.size:
                    bpm_est = 60.0 / float(np.median(diffs))
                    raw_tempo = float(bpm_est)
        except Exception:
            pass
    # final fallback: try tempo with different hop_length
    if raw_tempo == 0.0:
        try:
            alt = librosa.beat.tempo(y=y, sr=sr, hop_length=512).mean()
            if alt and alt > 0:
                raw_tempo = float(alt)
        except Exception:
            pass
    # normalize BPM for better downstream scoring (handle half/double-time)
    try:
        tempo = normalize_bpm(raw_tempo)
    except Exception:
        tempo = raw_tempo

    # chroma -> key
    try:
        # use chroma_cens for a more stable chroma representation for key detection
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key = detect_key(chroma_mean)
    except Exception:
        key = {'tonic': None, 'mode': None}

    # energy: compute per-frame RMS so we can extract start/end energy (directionality)
    try:
        rms = librosa.feature.rms(y=y)
        rms_frames = np.array(rms).reshape(-1)
        rms_mean = float(np.mean(rms_frames)) if rms_frames.size else 0.0
        # use max for relative start/end normalization
        max_rms = float(np.max(rms_frames)) if rms_frames.size and np.max(rms_frames) > 0 else 1e-9
        n_frames = rms_frames.shape[0]
        edge = max(1, int(max(1, round(0.10 * n_frames))))
        start_mean = float(np.mean(rms_frames[:edge])) if n_frames >= 1 else rms_mean
        end_mean = float(np.mean(rms_frames[-edge:])) if n_frames >= 1 else rms_mean
        energy_start_pct = max(0.0, min(1.0, start_mean / max_rms))
        energy_end_pct = max(0.0, min(1.0, end_mean / max_rms))
        # global loudness (dB) from mean
        rms_db_val = librosa.amplitude_to_db([rms_mean], ref=1.0)[0]
    except Exception:
        rms_db_val = -999.0
        energy_start_pct = 0.0
        energy_end_pct = 0.0

    # onset strength (used for both danceability and energy)
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_mean = float(np.mean(onset_env)) if onset_env.size else 0.0
        onset_score = max(0.0, min(1.0, math.tanh(onset_mean)))
    except Exception:
        onset_mean = 0.0
        onset_score = 0.0

    # spectral centroid (higher centroid -> perceived brightness/energy)
    try:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = float(np.mean(centroid)) if centroid.size else 0.0
        # normalize centroid relative to Nyquist (sr/2) using np.clip
        centroid_pct = float(np.clip(centroid_mean / (sr / 2.0), 0.0, 1.0))
    except Exception:
        centroid_mean = 0.0
        centroid_pct = 0.0

    # loudness percentage (0..1) from rms dB, assume -60dB -> 0, 0dB -> 1
    try:
        loudness_pct = float(np.clip((rms_db_val + 60.0) / 60.0, 0.0, 1.0))
    except Exception:
        loudness_pct = 0.0

    # weighted energy model: 0.4*loudness + 0.3*centroid + 0.3*onset
    try:
        energy_score = 0.4 * loudness_pct + 0.3 * centroid_pct + 0.3 * onset_score
        energy = int(max(0, min(100, round(energy_score * 100))))
    except Exception:
        energy = 0

    # danceability heuristic: combine tempo proximity and onset strength
    try:
        tempo_score = max(0.0, 1.0 - abs(120.0 - (tempo or 0.0)) / 60.0)
        danceability = int(max(0, min(100, round((0.6 * tempo_score + 0.4 * onset_score) * 100))))
    except Exception:
        danceability = 0

    # loudness (dB)
    try:
        loudness = float(rms_db_val)
    except Exception:
        loudness = -999.0

    # mfcc for clustering
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
    except Exception:
        mfcc_mean = np.zeros(20)
    # energy curve (downsampled) and simple segmentation
    try:
        if isinstance(rms_frames, np.ndarray) and rms_frames.size:
            n_frames = int(rms_frames.shape[0])
            bins = 100
            idxs = np.linspace(0, max(0, n_frames - 1), bins)
            curve = np.interp(idxs, np.arange(n_frames), (rms_frames / max_rms))
            energy_curve = [int(max(0, min(100, round(float(v) * 100)))) for v in curve]
            peak_idx = int(np.argmax(curve)) if len(curve) else 0
            peak_val = float(curve[peak_idx]) if len(curve) else 0.0
            half = peak_val * 0.5
            left = peak_idx
            while left > 0 and curve[left] >= half:
                left -= 1
            right = peak_idx
            while right < len(curve) - 1 and curve[right] >= half:
                right += 1
            peak_start_pct = (left / len(curve)) if len(curve) else 0.0
            peak_end_pct = (right / len(curve)) if len(curve) else 0.0
            peak_mean = int(round(np.mean(curve[left:right + 1]) * 100)) if right >= left else int(round(peak_val * 100))
        else:
            energy_curve = []
            peak_start_pct = 0.0
            peak_end_pct = 0.0
            peak_mean = 0
    except Exception:
        energy_curve = []
        peak_start_pct = 0.0
        peak_end_pct = 0.0
        peak_mean = 0

    energy_segments = {
        'intro_mean': int(round(energy_start_pct * 100)),
        'peak_mean': peak_mean,
        'outro_mean': int(round(energy_end_pct * 100)),
        'peak_start_pct': float(peak_start_pct),
        'peak_end_pct': float(peak_end_pct)
    }

    # best transition suggestion based on end energy
    try:
        if energy_end_pct >= 0.7:
            best_transition_out = 'high-energy cut'
        elif energy_end_pct >= 0.4:
            best_transition_out = 'echo or reverb trail'
        else:
            best_transition_out = 'low-energy crossfade or fade out'
    except Exception:
        best_transition_out = None

    # vocal_coverage heuristic: estimate proportion of frames dominated by
    # harmonic (vocal-like) energy using harmonic-percussive separation.
    try:
        y_harm, y_perc = librosa.effects.hpss(y)
        hrms = librosa.feature.rms(y=y_harm)
        trms = librosa.feature.rms(y=y)
        hr = np.array(hrms).reshape(-1) if isinstance(hrms, np.ndarray) or hasattr(hrms, '__len__') else np.array([])
        tr = np.array(trms).reshape(-1) if isinstance(trms, np.ndarray) or hasattr(trms, '__len__') else np.array([])
        if hr.size and tr.size:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = hr / (tr + 1e-9)
            # ignore very low-energy frames to avoid false positives
            eng_thr = float(np.median(tr) * 0.1) if tr.size else 0.0
            vocal_frames = ((ratios > 0.6) & (tr > eng_thr))
            vocal_coverage = float(np.sum(vocal_frames)) / float(ratios.size) if ratios.size else 0.0
            # derive coarse vocal regions (in seconds) by grouping contiguous
            # vocal-detected frames. We use librosa.frames_to_time with the
            # default hop_length (512) which matches librosa.feature.rms defaults.
            vocal_regions = []
            try:
                if isinstance(vocal_frames, np.ndarray) and vocal_frames.size:
                    idx = np.where(vocal_frames)[0]
                    if idx.size:
                        regions = []
                        s = int(idx[0]); p = s
                        for ii in idx[1:]:
                            if int(ii) == p + 1:
                                p = int(ii)
                                continue
                            regions.append((s, p))
                            s = int(ii); p = int(ii)
                        regions.append((s, p))
                        hop_length = 512
                        for (fs, fe) in regions:
                            try:
                                start_t = float(librosa.frames_to_time(fs, sr=sr, hop_length=hop_length))
                                # use end frame + 1 to compute end time
                                end_t = float(librosa.frames_to_time(fe + 1, sr=sr, hop_length=hop_length))
                                # skip extremely short detections
                                if end_t - start_t >= 0.25:
                                    vocal_regions.append([round(start_t, 3), round(end_t, 3)])
                            except Exception:
                                continue
            except Exception:
                vocal_regions = []
        else:
            vocal_coverage = 0.0
            vocal_regions = []
    except Exception:
        vocal_coverage = 0.0
        vocal_regions = []

    # generate vibe tags using heuristics
    try:
        temp_feats = {
            'energy': energy,
            'danceability': danceability,
            'loudness_db': loudness,
            'bpm': tempo,
            'centroid_pct': centroid_pct,
            'energy_delta': int(round((energy_end_pct - energy_start_pct) * 100))
        }
        vibes = assign_vibes(temp_feats)
    except Exception:
        vibes = []

    return {
        'path': path,
        'raw_bpm': raw_tempo,
        'bpm': tempo,
        'key': key,
        'energy': energy,
        'energy_start': int(round(energy_start_pct * 100)),
        'energy_end': int(round(energy_end_pct * 100)),
        'energy_delta': int(round((energy_end_pct - energy_start_pct) * 100)),
        'danceability': danceability,
        'loudness_db': loudness,
        'centroid_pct': centroid_pct,
        'energy_curve': energy_curve,
        'energy_segments': energy_segments,
        'best_transition_out': best_transition_out,
        'vibes': vibes,
        'vocal_coverage': float(vocal_coverage),
        'vocal_regions': vocal_regions,
        'mfcc_mean': mfcc_mean.tolist()
    }


def main():
    # read newline-separated file paths from stdin and process one-by-one
    conn = init_db()
    ensure_compat_table(conn)
    cleanup_db(conn)
    feature_matrix = []
    paths = []
    feats_list = []
    count = 0
    to_compute = []
    stats_map = {}

    # first pass: read all input paths, normalize and check cache. Cached
    # results are printed immediately; others are scheduled for parallel compute.
    for line in sys.stdin:
        p = line.strip()
        if not p:
            continue
        # normalize paths for cross-platform consistency
        try:
            p = os.path.normpath(os.path.abspath(p))
        except Exception:
            pass
        count += 1
        # check file stats for caching
        try:
            st = os.stat(p)
            mtime = st.st_mtime
            size = st.st_size
        except Exception:
            mtime = None
            size = None

        cached = None
        if mtime is not None and size is not None:
            try:
                cached = get_cached(conn, p, mtime, size)
            except Exception:
                cached = None

        if cached is not None:
            feats = cached
            feats['_cached'] = True
            # emit cached immediately
            try:
                print(json.dumps(feats), flush=True)
            except Exception:
                print(json.dumps({"path": p, "error": "print_failed"}), flush=True)
            # collect for clustering
            if isinstance(feats.get('mfcc_mean'), list):
                try:
                    bpm_val = float(feats.get('bpm') or 0.0)
                    bpm_norm = float(np.clip((bpm_val - 70.0) / (180.0 - 70.0), 0.0, 1.0))
                except Exception:
                    bpm_norm = 0.0
                energy_pct = float(np.clip((feats.get('energy') or 0) / 100.0, 0.0, 1.0))
                dance_pct = float(np.clip((feats.get('danceability') or 0) / 100.0, 0.0, 1.0))
                vec = [bpm_norm, energy_pct, dance_pct] + list(feats['mfcc_mean'])
                feature_matrix.append(vec)
                paths.append(p)
            feats_list.append(feats)
        else:
            # schedule for compute; remember stats for DB upsert later
            to_compute.append(p)
            stats_map[p] = (mtime or 0.0, size or 0)

    # parallel compute for remaining tracks
    if to_compute:
        proc_count = max(1, min(8, (os.cpu_count() or 1)))
        try:
            with Pool(processes=proc_count) as pool:
                for feats in pool.imap_unordered(compute_features, to_compute):
                    if not isinstance(feats, dict):
                        continue
                    p = feats.get('path')
                    feats['_cached'] = False
                    # persist to DB if compute succeeded
                    try:
                        if not feats.get('error'):
                            mtime, size = stats_map.get(p, (0.0, 0))
                            upsert_features(conn, p, mtime, size, feats)
                    except Exception:
                        pass
                    try:
                        print(json.dumps(feats), flush=True)
                    except Exception:
                        print(json.dumps({"path": p, "error": "print_failed"}), flush=True)
                    # collect for clustering
                    if isinstance(feats.get('mfcc_mean'), list):
                        try:
                            bpm_val = float(feats.get('bpm') or 0.0)
                            bpm_norm = float(np.clip((bpm_val - 70.0) / (180.0 - 70.0), 0.0, 1.0))
                        except Exception:
                            bpm_norm = 0.0
                        energy_pct = float(np.clip((feats.get('energy') or 0) / 100.0, 0.0, 1.0))
                        dance_pct = float(np.clip((feats.get('danceability') or 0) / 100.0, 0.0, 1.0))
                        vec = [bpm_norm, energy_pct, dance_pct] + list(feats['mfcc_mean'])
                        feature_matrix.append(vec)
                        paths.append(p)
                    feats_list.append(feats)
        except Exception:
            # fallback to sequential compute on error
            for p in to_compute:
                feats = compute_features(p)
                feats['_cached'] = False
                try:
                    if not feats.get('error'):
                        mtime, size = stats_map.get(p, (0.0, 0))
                        upsert_features(conn, p, mtime, size, feats)
                except Exception:
                    pass
                try:
                    print(json.dumps(feats), flush=True)
                except Exception:
                    print(json.dumps({"path": p, "error": "print_failed"}), flush=True)
                if isinstance(feats.get('mfcc_mean'), list):
                    try:
                        bpm_val = float(feats.get('bpm') or 0.0)
                        bpm_norm = float(np.clip((bpm_val - 70.0) / (180.0 - 70.0), 0.0, 1.0))
                    except Exception:
                        bpm_norm = 0.0
                    energy_pct = float(np.clip((feats.get('energy') or 0) / 100.0, 0.0, 1.0))
                    dance_pct = float(np.clip((feats.get('danceability') or 0) / 100.0, 0.0, 1.0))
                    vec = [bpm_norm, energy_pct, dance_pct] + list(feats['mfcc_mean'])
                    feature_matrix.append(vec)
                    paths.append(p)
                feats_list.append(feats)

    # after processing all tracks, run a limited clustering step and emit summary
    n = len(feature_matrix)
    clusters = {}
    try:
        if n >= 2:
            # choose k modestly based on number of tracks
            k = min(12, max(2, int(math.sqrt(n))))
            X = np.array(feature_matrix)
            # if dataset is very large, sample for fitting then predict for all
            max_fit = 5000
            if n > max_fit:
                idx = np.random.choice(n, max_fit, replace=False)
                kmeans = KMeans(n_clusters=k, random_state=0).fit(X[idx])
            else:
                kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
            labels = kmeans.predict(X).tolist()
            for i, lab in enumerate(labels):
                clusters.setdefault(str(int(lab)), []).append(paths[i])
        else:
            clusters = {'0': paths}
    except Exception as e:
        clusters = {"error": str(e)}

    # compute and attach a simple 'familiarity' proxy per track based on cluster sizes,
    # then persist updates and emit per-track summaries so the UI can display them.
    try:
        # build mapping path->feat
        path_to_feat = {}
        for f in feats_list:
            if isinstance(f, dict):
                pth = f.get('path')
                if pth:
                    path_to_feat[pth] = f

        # only consider cluster entries that are lists
        path_to_cluster = {}
        cluster_sizes = {}
        for lab, members in (clusters.items() if isinstance(clusters, dict) else []):
            if isinstance(members, list):
                cluster_sizes[lab] = len(members)
                for p in members:
                    path_to_cluster[p] = lab

        max_cluster = max(cluster_sizes.values()) if cluster_sizes else 1

        for p in paths:
            feat = path_to_feat.get(p)
            if not isinstance(feat, dict):
                continue
            lab = path_to_cluster.get(p)
            size = cluster_sizes.get(lab, 1)
            try:
                norm = (size - 1) / (max_cluster - 1) if max_cluster > 1 else 0.0
            except Exception:
                norm = 0.0
            # familiarity proxy: scale into 20..100 so singletons aren't zero
            fam = int(round(20 + 80 * norm))
            feat['familiarity'] = max(0, min(100, fam))

            # ensure vibes exist for cached entries too
            try:
                if 'vibes' not in feat or feat.get('vibes') is None:
                    feat['vibes'] = assign_vibes(feat)
            except Exception:
                feat['vibes'] = feat.get('vibes') or []

            # best transition suggestion if missing
            if 'best_transition_out' not in feat:
                try:
                    eend = (feat.get('energy_end') or 0) / 100.0
                    if eend >= 0.7:
                        bto = 'high-energy cut'
                    elif eend >= 0.4:
                        bto = 'echo or reverb trail'
                    else:
                        bto = 'low-energy crossfade or fade out'
                except Exception:
                    bto = None
                feat['best_transition_out'] = bto

            # upsert updated features into DB
            try:
                try:
                    st = os.stat(p)
                    mtime = st.st_mtime
                    fsize = st.st_size
                except Exception:
                    mtime = 0.0
                    fsize = 0
                upsert_features(conn, p, mtime, fsize, feat)
            except Exception:
                pass

            # emit a compact update for UI consumers
            try:
                print(json.dumps({'path': p, 'familiarity': feat.get('familiarity'), 'vibes': feat.get('vibes'), 'best_transition_out': feat.get('best_transition_out')}), flush=True)
            except Exception:
                pass
    except Exception:
        pass

    # emit cluster summary as JSON
    try:
        print(json.dumps({"__clusters__": clusters}), flush=True)
    except Exception:
        pass
    # compute and emit compatibility mapping (compact)
    try:
        compat = compute_compatibility_matrix(feats_list, feature_matrix, clusters)
        try:
            print(json.dumps({"__compatibility__": compat}), flush=True)
        except Exception:
            pass
        # persist compatibility mapping into DB for later querying
        try:
            cur = conn.cursor()
            # clear existing compatibility for the set of scanned tracks
            for p in paths:
                try:
                    cur.execute('DELETE FROM compatibility WHERE track_path = ?', (p,))
                except Exception:
                    pass
            # insert compact mapping
            for src, neighs in compat.items():
                for nb in neighs:
                    try:
                        cur.execute('REPLACE INTO compatibility(track_path, neighbor_path, score, components) VALUES (?,?,?,?)',
                                    (src, nb.get('path'), int(nb.get('score') or 0), json.dumps(nb.get('components') or {})))
                    except Exception:
                        pass
            conn.commit()
        except Exception:
            pass
    except Exception:
        pass

if __name__ == '__main__':
    main()
