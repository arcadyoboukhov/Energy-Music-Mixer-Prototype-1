#!/usr/bin/env python3
"""Check whether all audio files under a folder are already present in musicscan.db.

Usage: python tools/check_musicscan_cache.py <folderPath> <dbPath>
Outputs a single JSON object on stdout with keys:
  - all_cached: bool
  - count: int (number of audio files found)
  - paths: list of str (only when all_cached is true)
  - missing: list of str (when not all cached, a sample of missing paths)
All human-readable progress/log messages are emitted to stderr so stdout remains
valid JSON for machine parsing.
"""
import os
import sys
import json
import sqlite3
import time

def collect_files(folder, exts={'.mp3', '.flac'}):
    out = []
    file_count = 0
    dir_count = 0
    start = time.time()
    print(f"check_musicscan_cache: starting file walk in {folder}", file=sys.stderr)
    for root, dirs, files in os.walk(folder):
        dir_count += 1
        if dir_count % 100 == 0:
            print(f"check_musicscan_cache: visited {dir_count} directories, found {file_count} audio files so far", file=sys.stderr)
        for f in files:
            try:
                if os.path.splitext(f)[1].lower() in exts:
                    abs_p = os.path.normcase(os.path.abspath(os.path.join(root, f)))
                    out.append(abs_p)
                    file_count += 1
                    if file_count % 500 == 0:
                        print(f"check_musicscan_cache: collected {file_count} audio files...", file=sys.stderr)
            except Exception:
                continue
    elapsed = time.time() - start
    print(f"check_musicscan_cache: finished file walk: {file_count} audio files in {dir_count} directories (elapsed {elapsed:.1f}s)", file=sys.stderr)
    return out

def read_db_paths(dbpath):
    out = set()
    if not os.path.exists(dbpath):
        print(f"check_musicscan_cache: DB not found at {dbpath}", file=sys.stderr)
        return out
    try:
        print(f"check_musicscan_cache: opening DB {dbpath}", file=sys.stderr)
        conn = sqlite3.connect(dbpath)
        cur = conn.cursor()
        try:
            # read the explicit `path` column which is faster than parsing JSON blobs
            cur.execute('SELECT path FROM tracks')
            rows = cur.fetchall()
        except Exception as e:
            print(f"check_musicscan_cache: DB query failed: {e}", file=sys.stderr)
            rows = []
        try:
            conn.close()
        except Exception:
            pass
        print(f"check_musicscan_cache: DB contains {len(rows)} track rows", file=sys.stderr)
        for (p,) in rows:
            try:
                if not p:
                    continue
                out.add(os.path.normcase(os.path.abspath(p)))
            except Exception:
                continue
    except Exception as e:
        print(f"check_musicscan_cache: error opening DB: {e}", file=sys.stderr)
    return out

def main():
    if len(sys.argv) < 3:
        print(json.dumps({'all_cached': False, 'reason': 'missing_args'}))
        return
    folder = sys.argv[1]
    db = sys.argv[2]
    folder = os.path.abspath(folder)
    print(f"check_musicscan_cache: checking folder {folder} against DB {db}", file=sys.stderr)
    start = time.time()
    files = collect_files(folder)
    db_paths = read_db_paths(db)
    missing = [p for p in files if p not in db_paths]
    elapsed = time.time() - start
    print(f"check_musicscan_cache: comparison done in {elapsed:.1f}s - files={len(files)} db_rows={len(db_paths)} missing={len(missing)}", file=sys.stderr)
    if not missing:
        print(json.dumps({'all_cached': True, 'count': len(files), 'paths': files}))
    else:
        print(json.dumps({'all_cached': False, 'count': len(files), 'missing_count': len(missing), 'missing_sample': missing[:100]}))

if __name__ == '__main__':
    main()
