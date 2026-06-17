#!/usr/bin/env python3
import sys
import os
import sqlite3
import json

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'missing_args'}))
        sys.exit(1)
    db = sys.argv[1]
    path = sys.argv[2]
    try:
        if not os.path.exists(db):
            print(json.dumps({'error': 'db_missing'}))
            sys.exit(1)
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute('SELECT features FROM tracks WHERE path = ?', (path,))
        row = cur.fetchone()
        conn.close()
        if not row:
            print(json.dumps({}))
            sys.exit(0)
        try:
            feats = json.loads(row[0])
            print(json.dumps(feats))
            sys.exit(0)
        except Exception:
            # if stored features are not valid JSON, return raw string
            print(json.dumps({'error': 'parse_failed', 'raw': row[0]}))
            sys.exit(0)
    except Exception as e:
        print(json.dumps({'error': 'exception', 'message': str(e)}))
        sys.exit(1)
