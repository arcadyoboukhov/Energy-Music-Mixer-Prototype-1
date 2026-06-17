#!/usr/bin/env python3
import os, sys, json
try:
    import musicscan
except Exception as e:
    print('musicscan import failed:', e)
    sys.exit(2)

def main():
    files = ['test_wavs/track_build.wav','test_wavs/track_drop.wav']
    conn = musicscan.init_db()
    musicscan.ensure_compat_table(conn)
    for rel in files:
        p = os.path.abspath(rel)
        if not os.path.exists(p):
            print('missing', p)
            continue
        print('processing', p)
        feats = musicscan.compute_features(p)
        try:
            st = os.stat(p)
            musicscan.upsert_features(conn, p, st.st_mtime, st.st_size, feats)
            print('seeded', p)
        except Exception as e:
            print('seed failed', p, e)
    try:
        conn.close()
    except Exception:
        pass

if __name__ == '__main__':
    main()
