import sqlite3
con = sqlite3.connect('musicscan.db')
cur = con.cursor()
cur.execute('SELECT path, played_at, duration FROM play_history ORDER BY played_at DESC LIMIT 5')
rows = cur.fetchall()
for r in rows:
    print(r)
con.close()
