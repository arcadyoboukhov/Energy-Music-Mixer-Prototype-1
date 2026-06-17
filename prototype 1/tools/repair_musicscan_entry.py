#!/usr/bin/env python3
"""Recompute features for a single audio file and upsert into musicscan.db.

Usage: python tools/repair_musicscan_entry.py <dbPath> <filePath>
Outputs the computed feature JSON on stdout on success, or a JSON error object.
"""
import sys
import os
import json
import sqlite3
import subprocess


def fallback_run_musicscan_single(file_path):
    # fallback: spawn musicscan.py and feed a single path on stdin
    try:
        here = os.path.dirname(os.path.dirname(__file__))
        proc = subprocess.Popen([sys.executable, os.path.join(here, 'musicscan.py')], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate(file_path + '\n')
        # take last non-empty line as JSON
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if not lines:
            return {'error': 'no_output', 'stderr': err}
        try:
            return json.loads(lines[-1])
        except Exception as e:
            return {'error': 'parse_failed', 'message': str(e), 'stdout': out, 'stderr': err}
    except Exception as e:
        return {'error': 'fallback_failed', 'message': str(e)}


def main():
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'missing_args'}))
        sys.exit(1)
    dbpath = sys.argv[1]
    file_path = sys.argv[2]
    if not os.path.exists(file_path):
        print(json.dumps({'error': 'file_missing', 'path': file_path}))
        sys.exit(1)

    # Try to import musicscan module and call compute/upsert directly for minimal work
    try:
        # ensure import from project root
        proj_root = os.path.dirname(os.path.dirname(__file__))
        if proj_root not in sys.path:
            sys.path.insert(0, proj_root)
        import musicscan
    except Exception as e:
        # fallback to spawning full musicscan.py
        res = fallback_run_musicscan_single(file_path)
        print(json.dumps(res))
        sys.exit(0 if (isinstance(res, dict) and not res.get('error')) else 2)

    try:
        feats = musicscan.compute_features(file_path)
    except Exception as e:
        print(json.dumps({'error': 'compute_failed', 'message': str(e), 'path': file_path}))
        sys.exit(1)

    try:
        st = os.stat(file_path)
        mtime = st.st_mtime
        size = st.st_size
    except Exception:
        mtime = 0.0
        size = 0

    try:
        conn = sqlite3.connect(dbpath)
        musicscan.upsert_features(conn, file_path, mtime, size, feats)
        conn.close()
    except Exception as e:
        print(json.dumps({'error': 'db_upsert_failed', 'message': str(e)}))
        sys.exit(1)

    try:
        print(json.dumps(feats))
    except Exception:
        print(json.dumps({'error': 'print_failed'}))


if __name__ == '__main__':
    main()
