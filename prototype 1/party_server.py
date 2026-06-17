#!/usr/bin/env python3
"""Lightweight party server: JSON stdin/stdout control bridge for the UI.

Commands (JSON lines on stdin):
  {"cmd":"start_party","vibe":"Chill"}
  {"cmd":"set_energy_target","value":75}
  {"cmd":"adjust_familiarity_bias","delta":10}
  {"cmd":"record_play","params": {"path":"...","played_at":..., "duration":30}}
  {"cmd":"select_next","params": {"candidates": [...], "prev_track_path": "...", "record_choice": true}}

Emits (JSON lines on stdout):
  {"type":"party_state","state":{...}}
  {"type":"select_result","best":..., "scored":[...]}
"""

import sys
import json
import time
import threading
import traceback
import os
import sqlite3

try:
    import party_engine
    import memory
except Exception:
    # allow running even if the environment lacks full deps
    party_engine = None
    memory = None


class Server:
    def __init__(self):
        # party state
        self.lock = threading.Lock()
        # use party_engine.PartyState if available
        if party_engine is not None and hasattr(party_engine, 'PartyState'):
            self.state = party_engine.PartyState()
        else:
            class S: pass
            self.state = S()
            self.state.energy_level = 65
            self.state.trajectory = 'steady'
            self.state.crowd_type = 'mixed'
            self.state.time_elapsed = 0.0
            self.state.familiarity_bias = 0.7
            self.state.current_genre_cluster = None
            self.state.prev_track_path = None

        self.target = getattr(self.state, 'energy_level', 65)
        self.status = 'idle'
        self.vibe = None
        self.now_track = None
        self.next_track = None
        # advanced UI-controlled settings
        self.advanced = {}
        # feedback / ask state tracking
        self._prev_trajectory = None
        self._prev_next = None
        self._prev_now = None
        self.last_ask = 0
        # whether `next_track` was set dynamically by the selector
        self._next_is_dynamic = False
        # guest request queue
        self.requests = []  # list of dicts {id, path, title, energy, vibes, suggester, created_at, upvotes, voters, status}
        self.http_port = None

        # optional selector
        try:
            if party_engine is not None:
                self.engine = party_engine.EnergyEngine()
                self.selector = party_engine.TrackSelector(self.engine)
            else:
                self.engine = None
                self.selector = None
        except Exception:
            self.engine = None
            self.selector = None

        self._start_threads()
        # start guest HTTP server
        try:
            self._start_http_server()
        except Exception:
            self.send({'type': 'error', 'message': 'http_start_failed', 'detail': traceback.format_exc()})

    def send(self, obj):
        try:
            sys.stdout.write(json.dumps(obj, default=str) + '\n')
            sys.stdout.flush()
        except Exception:
            try:
                sys.stdout.write('{"type":"error","message":"send_failed"}\n')
                sys.stdout.flush()
            except Exception:
                pass

    def state_dict(self):
        with self.lock:
            return {
                'energy': int(getattr(self.state, 'energy_level', 0)),
                'target': int(self.target),
                'trend': getattr(self.state, 'trajectory', 'steady'),
                'status': self.status,
                'familiarity_bias': float(getattr(self.state, 'familiarity_bias', 0.7)),
                'vibe': self.vibe,
                'now_track': self.now_track,
                'next_track': self.next_track,
                'time_elapsed': float(getattr(self.state, 'time_elapsed', 0.0)),
            }

    def _vibe_to_target(self, vibe: str):
        """Map a human-friendly vibe to a reasonable default target energy (0-100).

        Returns an int target or None when no mapping.
        """
        if not vibe:
            return None
        try:
            v = (vibe or '').strip().lower()
            if 'party' in v:
                return 85
            if 'late' in v or 'late-night' in v or 'night' in v:
                return 70
            if 'chill' in v:
                return 45
            if 'dinner' in v:
                return 25
        except Exception:
            return None
        return None

    def _tick_loop(self):
        # update energy gradually toward target and broadcast
        while True:
            try:
                with self.lock:
                    cur = float(getattr(self.state, 'energy_level', 65))
                    tgt = float(self.target)
                    if cur < tgt:
                        diff = tgt - cur
                        step = min(3.0, max(1.0, round(diff * 0.06)))
                        cur = min(100.0, cur + step)
                        self.state.energy_level = cur
                        self.state.trajectory = 'rising'
                    elif cur > tgt:
                        diff = cur - tgt
                        step = min(3.0, max(1.0, round(diff * 0.06)))
                        cur = max(0.0, cur - step)
                        self.state.energy_level = cur
                        self.state.trajectory = 'falling'
                    else:
                        self.state.trajectory = 'steady'
                    # advance timeline
                    try:
                        self.state.time_elapsed = float(self.state.time_elapsed) + 1.0
                    except Exception:
                        self.state.time_elapsed = 0.0

                # process guest queue for possible injections
                try:
                    self._process_guest_queue()
                except Exception:
                    self.send({'type': 'error', 'message': 'process_queue_failed', 'detail': traceback.format_exc()})
                # compute feedback and ask prompts when appropriate
                try:
                    self._handle_feedback_tick()
                except Exception:
                    self.send({'type': 'error', 'message': 'feedback_failed', 'detail': traceback.format_exc()})
                # emit state
                self.send({'type': 'party_state', 'state': self.state_dict()})
            except Exception:
                self.send({'type': 'error', 'message': 'tick_failed', 'detail': traceback.format_exc()})
            time.sleep(1.0)

    def _stdin_loop(self):
        buf = ''
        while True:
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.1)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                self.send({'type': 'error', 'message': 'invalid_json', 'raw': line})
                continue
            try:
                cmd = obj.get('cmd')
                if cmd == 'set_energy_target':
                    val = int(obj.get('value') or obj.get('v') or 0)
                    with self.lock:
                        self.target = max(0, min(100, val))
                        self.status = 'Party is building'
                    # update dynamic queued next track to match new target
                    try:
                        self._update_dynamic_next()
                    except Exception:
                        pass
                    self.send({'type': 'ok', 'cmd': cmd, 'state': self.state_dict()})

                elif cmd == 'adjust_familiarity_bias':
                    delta = float(obj.get('delta') or obj.get('d') or 0.0)
                    with self.lock:
                        try:
                            old = float(getattr(self.state, 'familiarity_bias', 0.7))
                        except Exception:
                            old = 0.7
                        new = max(-1.0, min(1.0, old + float(delta) / 100.0))
                        self.state.familiarity_bias = new
                    self.send({'type': 'ok', 'cmd': cmd, 'state': self.state_dict()})

                elif cmd == 'start_party':
                    vibe = obj.get('vibe') or obj.get('params', {}).get('vibe')
                    with self.lock:
                        self.status = 'Party is building'
                        self.vibe = vibe or self.vibe
                        # map vibe to a sensible default target energy so the
                        # initial selection and queued next match the host intent
                        try:
                            mapped = self._vibe_to_target(self.vibe)
                            if mapped is not None:
                                self.target = max(0, min(100, int(mapped)))
                        except Exception:
                            pass
                    # recompute dynamic next based on new vibe/target
                    try:
                        self._update_dynamic_next()
                    except Exception:
                        pass
                    self.send({'type': 'ok', 'cmd': cmd, 'state': self.state_dict()})

                elif cmd == 'connect_spotify':
                    # placeholder
                    with self.lock:
                        self.status = 'connected: spotify (placeholder)'
                    self.send({'type': 'ok', 'cmd': cmd, 'state': self.state_dict()})

                elif cmd == 'record_play':
                    params = obj.get('params') or {}
                    try:
                        if memory is not None:
                            memory.record_play(path=params.get('path'), played_at=params.get('played_at'), duration=params.get('duration'), skipped=params.get('skipped', False), moment_type=params.get('moment_type'), artist_cluster=params.get('artist_cluster'), energy=params.get('energy'))
                            memory.update_preferences_on_event({'type': 'play', 'path': params.get('path'), 'vibes': params.get('vibes', []), 'energy': params.get('energy'), 'played_at': params.get('played_at') or time.time()})
                        else:
                            # cannot persist
                            pass
                        # update now_track for UI
                        with self.lock:
                            self.now_track = params.get('path')
                    except Exception:
                        self.send({'type': 'error', 'message': 'record_play_failed', 'detail': traceback.format_exc()})
                    self.send({'type': 'ok', 'cmd': cmd})

                elif cmd == 'playback_play':
                    params = obj.get('params') or {}
                    try:
                        with self.lock:
                            self.status = 'playing'
                            p = params.get('path')
                            if p:
                                self.now_track = p
                                # when user explicitly plays a track, refresh the dynamic next
                                try:
                                    # mark next as dynamic only if selector produces it
                                    pass
                                except Exception:
                                    pass
                        # record play event if memory available
                        if memory is not None and params.get('path'):
                            try:
                                memory.record_play(path=params.get('path'), played_at=params.get('played_at') or time.time(), duration=params.get('duration'), skipped=False)
                            except Exception:
                                pass
                        try:
                            # refresh computed next track now that we've set `now_track`
                            self._update_dynamic_next()
                        except Exception:
                            pass
                    except Exception:
                        self.send({'type': 'error', 'message': 'playback_play_failed', 'detail': traceback.format_exc()})
                    self.send({'type': 'ok', 'cmd': cmd, 'state': self.state_dict()})

                elif cmd == 'playback_pause':
                    try:
                        with self.lock:
                            self.status = 'paused'
                    except Exception:
                        self.send({'type': 'error', 'message': 'playback_pause_failed', 'detail': traceback.format_exc()})
                    self.send({'type': 'ok', 'cmd': cmd, 'state': self.state_dict()})

                elif cmd == 'playback_skip':
                    params = obj.get('params') or {}
                    try:
                        prev = None
                        to_play = None
                        with self.lock:
                            prev = self.now_track
                            # record skipped in memory
                            try:
                                if memory is not None and prev:
                                    memory.record_play(path=prev, played_at=params.get('played_at') or time.time(), duration=params.get('duration'), skipped=True)
                            except Exception:
                                pass
                            # prefer an already-queued next if present
                            if self.next_track:
                                to_play = self.next_track
                                self.next_track = None
                                try:
                                    self._next_is_dynamic = False
                                except Exception:
                                    pass
                            else:
                                to_play = None

                        # if no queued next, attempt to compute one immediately
                        if to_play is None:
                            try:
                                candidates = self._load_candidates_from_db()
                                if candidates and self.selector is not None:
                                    adv = dict(getattr(self, 'advanced', {}) or {})
                                    if getattr(self, 'vibe', None):
                                        adv['vibe'] = self.vibe
                                    best, scored = self.selector.select_next_track(self.state, candidates, recent_played=None, prev_track_path=prev, record_choice=False, advanced=adv, target_override=getattr(self, 'target', None))
                                    if best and best.get('path'):
                                        to_play = best.get('path')
                            except Exception:
                                to_play = None

                        # set now_track to chosen play path (if any) and recompute following next
                        try:
                            with self.lock:
                                if to_play:
                                    self.now_track = to_play
                                else:
                                    self.now_track = None
                        except Exception:
                            pass

                        # let selector compute the following queued next track
                        try:
                            self._update_dynamic_next()
                        except Exception:
                            pass

                        # notify UI
                        self.send({'type': 'feedback', 'message': 'Track skipped'})
                    except Exception:
                        self.send({'type': 'error', 'message': 'playback_skip_failed', 'detail': traceback.format_exc()})
                    # respond with updated state and explicit skip result
                    try:
                        self.send({'type': 'party_state', 'state': self.state_dict()})
                        self.send({'type': 'skip_result', 'now': self.now_track, 'next': self.next_track})
                    except Exception:
                        pass
                    self.send({'type': 'ok', 'cmd': cmd})

                elif cmd == 'select_initial' or cmd == 'select-initial':
                    # choose an initial track by loading cached features from musicscan.db
                    try:
                        dbfile = os.path.join(os.path.dirname(__file__), 'musicscan.db')
                        if not os.path.exists(dbfile):
                            self.send({'type': 'error', 'message': 'no_scan_db'})
                            continue
                        conn = sqlite3.connect(dbfile)
                        cur = conn.cursor()
                        try:
                            cur.execute("SELECT features FROM tracks")
                            rows = cur.fetchall()
                        except Exception:
                            rows = []
                        candidates = []
                        for (fstr,) in rows:
                            try:
                                feat = json.loads(fstr)
                                candidates.append(feat)
                            except Exception:
                                continue
                        try:
                            conn.close()
                        except Exception:
                            pass
                        if not candidates:
                            self.send({'type': 'error', 'message': 'no_candidates'})
                            continue
                        if self.selector is None:
                            self.send({'type': 'error', 'message': 'no_selector'})
                            continue
                        try:
                            # prefer explicit server target when choosing the initial track
                            adv = dict(getattr(self, 'advanced', {}) or {})
                            if getattr(self, 'vibe', None):
                                adv['vibe'] = self.vibe
                            best, scored = self.selector.select_next_track(self.state, candidates, recent_played=None, prev_track_path=getattr(self.state, 'prev_track_path', None), record_choice=True, advanced=adv, target_override=getattr(self, 'target', None))
                            ssc = []
                            for it in scored:
                                tr = it.get('track') if isinstance(it, dict) else None
                                pth = (tr.get('path') if isinstance(tr, dict) else None) if tr else (it.get('path') if isinstance(it, dict) else None)
                                ssc.append({'path': pth, 'score': it.get('score'), 'components': it.get('components')})
                            out = {'type': 'select_result', 'best': (best.get('path') if best else None), 'scored': ssc}
                            # set now_track to the chosen initial track so UI can reflect it
                            with self.lock:
                                if best and best.get('path'):
                                    self.now_track = best.get('path')
                            self.send(out)
                            # broadcast state update
                            self.send({'type': 'party_state', 'state': self.state_dict()})
                            # queue a dynamic next track based on current target
                            try:
                                self._update_dynamic_next()
                            except Exception:
                                pass
                        except Exception:
                            self.send({'type': 'error', 'message': 'select_initial_failed', 'detail': traceback.format_exc()})
                    except Exception:
                        self.send({'type': 'error', 'message': 'select_initial_error', 'detail': traceback.format_exc()})

                elif cmd == 'select_next':
                    params = obj.get('params') or {}
                    candidates = params.get('candidates') or []
                    prev = params.get('prev_track_path')
                    record_choice = bool(params.get('record_choice'))
                    # accept advanced filters from UI or fallback to server-stored advanced config
                    # merge and include host-selected `vibe` so selection matches the party
                    adv = dict(getattr(self, 'advanced', {}) or {})
                    if isinstance(params.get('advanced'), dict):
                        try:
                            adv.update(params.get('advanced') or {})
                        except Exception:
                            pass
                    if getattr(self, 'vibe', None):
                        adv['vibe'] = self.vibe
                    advanced = adv
                    try:
                        if self.selector is not None:
                            # allow explicit target in params (fallback to server target)
                            target_override = params.get('target') if params.get('target') is not None else getattr(self, 'target', None)
                            best, scored = self.selector.select_next_track(self.state, candidates, recent_played=params.get('recent_played'), prev_track_path=prev, record_choice=record_choice, advanced=advanced, target_override=target_override)
                            # serialize
                            ssc = []
                            for it in scored:
                                tr = it.get('track')
                                ssc.append({'path': tr.get('path'), 'score': it.get('score'), 'components': it.get('components')})
                            out = {'type': 'select_result', 'best': (best.get('path') if best else None), 'scored': ssc}
                            self.send(out)
                        else:
                            self.send({'type': 'error', 'message': 'no_selector'})
                    except Exception:
                        self.send({'type': 'error', 'message': 'select_failed', 'detail': traceback.format_exc()})

                elif cmd == 'inject_request':
                    rid = obj.get('id') or obj.get('request_id')
                    if not rid:
                        self.send({'type': 'error', 'message': 'missing_request_id'})
                    else:
                        injected = False
                        with self.lock:
                            for i, r in enumerate(self.requests):
                                if r.get('id') == rid:
                                    req = self.requests.pop(i)
                                    self.next_track = req.get('path')
                                    # injected request should override dynamic queuing
                                    try:
                                        self._next_is_dynamic = False
                                    except Exception:
                                        pass
                                    req['status'] = 'injected'
                                    req['scheduled_at'] = time.time()
                                    injected = True
                                    # notify guests and host
                                    self.send({'type': 'guest_inject', 'id': rid, 'path': req.get('path')})
                                    break
                        if injected:
                            # broadcast updated request list
                            try:
                                self.send({'type': 'guest_requests', 'requests': self._requests_summary()})
                            except Exception:
                                pass
                        else:
                            self.send({'type': 'error', 'message': 'request_not_found', 'id': rid})

                elif cmd == 'set_advanced':
                    params = obj.get('params') or {}
                    try:
                        with self.lock:
                            self.advanced = getattr(self, 'advanced', {})
                            if isinstance(params, dict):
                                self.advanced.update(params)
                            else:
                                # accept single values wrapped as params
                                self.advanced = params
                        # recompute next to respect new advanced filters
                        try:
                            self._update_dynamic_next()
                        except Exception:
                            pass
                        self.send({'type': 'ok', 'cmd': cmd, 'advanced': self.advanced})
                    except Exception:
                        self.send({'type': 'error', 'message': 'set_advanced_failed', 'detail': traceback.format_exc()})

                elif cmd == 'failure_response' or cmd == 'failure-response':
                    # host/user responded to an uncertainty prompt
                    choice = obj.get('choice') or obj.get('params') or obj.get('value')
                    try:
                        # delegate to helper for easier programmatic testing
                        self.apply_failure_choice(choice)
                    except Exception:
                        self.send({'type': 'error', 'message': 'failure_response_failed', 'detail': traceback.format_exc()})
                    self.send({'type': 'ok', 'cmd': cmd})

                else:
                    self.send({'type': 'error', 'message': 'unknown_cmd', 'cmd': cmd})
            except Exception:
                self.send({'type': 'error', 'message': 'handler_failed', 'detail': traceback.format_exc()})

    def _start_threads(self):
        t = threading.Thread(target=self._tick_loop, daemon=True)
        t.start()
        r = threading.Thread(target=self._stdin_loop, daemon=True)
        r.start()

    def _start_http_server(self):
        # start a simple threaded HTTP server to accept guest suggestions and upvotes
        import socket
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from socketserver import ThreadingMixIn
        import urllib.parse
        import uuid

        server_self = self

        class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True

        class Handler(BaseHTTPRequestHandler):
            def _set_cors(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')

            def _send_json(self, code, obj):
                bs = json.dumps(obj, default=str).encode('utf-8')
                self.send_response(code)
                self._set_cors()
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(bs)))
                self.end_headers()
                self.wfile.write(bs)

            def do_OPTIONS(self):
                self.send_response(204)
                self._set_cors()
                self.end_headers()

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path
                if path == '/requests':
                    try:
                        self._send_json(200, {'requests': server_self._requests_summary()})
                    except Exception as e:
                        self._send_json(500, {'error': str(e)})
                elif path == '/status':
                    self._send_json(200, {'state': server_self.state_dict()})
                elif path == '/guest' or path == '/':
                    # simple guest UI
                    html = '''<!doctype html><html><head><meta charset="utf-8"><title>Guest — Suggest</title></head><body>
<h3>Suggest a song</h3>
<div><input id="title" placeholder="title or path" style="width:70%"></div>
<div><input id="energy" placeholder="energy (0-100)" style="width:120px"></div>
<div><input id="name" placeholder="your name (optional)" style="width:200px"></div>
<div style="margin-top:8px"><button onclick="submit()">Suggest</button></div>
<h4>Requests</h4>
<div id="list">loading...</div>
<script>
async function refresh(){
  const r = await fetch('/requests');
  const j = await r.json();
  const el = document.getElementById('list');
  el.innerHTML = '';
  (j.requests||[]).forEach(function(req){
    const d = document.createElement('div'); d.style.margin='8px 0';
    d.innerHTML = `<b>${req.title||req.path}</b> — upvotes:${req.upvotes} <button onclick="upvote('${req.id}')">Upvote</button>`;
    el.appendChild(d);
  });
}
async function submit(){
  const title=document.getElementById('title').value;
  const energy=parseInt(document.getElementById('energy').value||'50');
  const name=document.getElementById('name').value||'guest';
  await fetch('/suggest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title,energy,suggester:name})});
  document.getElementById('title').value='';document.getElementById('energy').value='';
  setTimeout(refresh,300);
}
async function upvote(id){await fetch('/upvote',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});setTimeout(refresh,200);} 
setInterval(refresh,3000);refresh();
</script>
</body></html>'''
                    bs = html.encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(bs)))
                    self.end_headers()
                    self.wfile.write(bs)
                else:
                    self._send_json(404, {'error': 'not_found'})

            def do_POST(self):
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path
                content_length = int(self.headers.get('Content-Length', '0'))
                body = self.rfile.read(content_length) if content_length else b''
                try:
                    data = json.loads(body.decode('utf-8') or '{}')
                except Exception:
                    data = {}

                if path == '/suggest':
                    title = data.get('title') or data.get('path') or 'guest-suggestion'
                    energy = int(data.get('energy') or 50)
                    vibes = data.get('vibes') or []
                    suggester = data.get('suggester') or data.get('name') or 'guest'
                    rid = str(uuid.uuid4())
                    req = {'id': rid, 'path': data.get('path') or title, 'title': title, 'energy': energy, 'vibes': vibes, 'suggester': suggester, 'created_at': time.time(), 'upvotes': 1, 'voters': [suggester], 'status': 'pending'}
                    with server_self.lock:
                        server_self.requests.append(req)
                    # broadcast update
                    server_self.send({'type': 'guest_requests', 'requests': server_self._requests_summary()})
                    self._send_json(201, {'ok': True, 'id': rid})
                elif path == '/upvote':
                    rid = data.get('id')
                    voter = data.get('voter') or data.get('name') or 'guest'
                    found = False
                    with server_self.lock:
                        for r in server_self.requests:
                            if r.get('id') == rid:
                                if voter not in r.get('voters', []):
                                    r['voters'].append(voter)
                                    r['upvotes'] = int(r.get('upvotes', 0)) + 1
                                found = True
                                break
                    if found:
                        server_self.send({'type': 'guest_requests', 'requests': server_self._requests_summary()})
                        self._send_json(200, {'ok': True})
                    else:
                        self._send_json(404, {'error': 'not_found'})
                else:
                    self._send_json(404, {'error': 'not_found'})

        # bind to an available port
        httpd = ThreadedHTTPServer(('0.0.0.0', 0), Handler)
        port = httpd.server_address[1]
        self.http_port = port
        # discover local IP for guest QR (best-effort)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            ip = '127.0.0.1'

        url = f'http://{ip}:{port}/guest'
        # notify host/UI
        self.send({'type': 'guest_ready', 'url': url, 'port': port})

        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()

    def apply_failure_choice(self, choice):
        """Programmatic helper to apply a host's failure-response choice (for testing).

        choice may be a string like 'hype' or 'switch' or a dict containing a 'choice'/'value'.
        """
        try:
            with self.lock:
                # normalize
                if isinstance(choice, dict):
                    choice = choice.get('choice') or choice.get('value')
                c = (choice or '').lower() if choice else ''
                if c in ('hype', 'keep it hype', 'keep'):
                    self.target = min(100, int(self.target) + 10)
                    self.send({'type': 'feedback', 'message': 'Keeping it hype — boosting energy 🔥'})
                elif c in ('switch', 'switch it up', 'cool'):
                    self.target = max(0, int(self.target) - 15)
                    self.send({'type': 'feedback', 'message': 'Switching it up — cooling the vibe 🎛️'})
                else:
                    self.send({'type': 'feedback', 'message': f'Applied choice: {choice}'})
        except Exception:
            self.send({'type': 'error', 'message': 'apply_failure_choice_failed', 'detail': traceback.format_exc()})

    def _requests_summary(self):
        with self.lock:
            out = []
            for r in self.requests:
                out.append({'id': r.get('id'), 'title': r.get('title'), 'path': r.get('path'), 'upvotes': int(r.get('upvotes', 0)), 'created_at': float(r.get('created_at')), 'energy': int(r.get('energy') or 50), 'status': r.get('status')})
            return out

    def _handle_feedback_tick(self):
        # emit human-friendly feedback messages based on state changes and prompt for decisions
        with self.lock:
            try:
                tr = getattr(self.state, 'trajectory', 'steady')
                if tr != getattr(self, '_prev_trajectory', None):
                    if tr == 'rising':
                        self.send({'type': 'feedback', 'message': 'Crowd energy rising 📈'})
                    elif tr == 'falling':
                        self.send({'type': 'feedback', 'message': 'Energy cooling down — shifting vibe soon…'})
                    self._prev_trajectory = tr

                if getattr(self, 'next_track', None) and not getattr(self, '_prev_next', None):
                    self.send({'type': 'feedback', 'message': 'Switching vibe soon…'})
                self._prev_next = self.next_track

                if getattr(self, 'now_track', None) and getattr(self, 'now_track', None) != getattr(self, '_prev_now', None):
                    self.send({'type': 'feedback', 'message': 'Bringing in a classic 🔥'})
                    self._prev_now = self.now_track

                # Ask-choice prompt disabled: do not prompt the host periodically.
                # Historically the server would ask the host for a choice when the
                # selector was missing; this was removed to avoid blocking the UI.
            except Exception:
                # swallow; feedback should not crash the server
                pass

    def _process_guest_queue(self):
        # choose top request by upvotes/time and decide whether to inject now
        with self.lock:
            if not self.requests:
                return
            now = time.time()
            def score(r):
                age = now - float(r.get('created_at') or now)
                return int(r.get('upvotes', 0)) * 10 + max(0, int(300 - age))

            # sort by score desc
            sorted_reqs = sorted(self.requests, key=score, reverse=True)
            top = sorted_reqs[0]
            cur_energy = float(getattr(self.state, 'energy_level', 50))
            trend = getattr(self.state, 'trajectory', 'steady')
            req_energy = float(top.get('energy') or 50)

            # if request is much lower energy and party is rising -> delay
            if (cur_energy - req_energy) >= 25 and trend == 'rising':
                # keep delayed until cooldown
                return

            # if upvotes are high, or not mismatched, inject
            inject_immediate = False
            if int(top.get('upvotes', 0)) >= 5:
                inject_immediate = True
            else:
                if (req_energy >= cur_energy - 10) or trend == 'falling' or trend == 'steady':
                    inject_immediate = True

            if inject_immediate:
                # remove top from queue and schedule
                for i, r in enumerate(self.requests):
                    if r.get('id') == top.get('id'):
                        req = self.requests.pop(i)
                        req['status'] = 'scheduled'
                        req['scheduled_at'] = time.time()
                        self.next_track = req.get('path')
                        # scheduled guest request should override dynamic queuing
                        try:
                            self._next_is_dynamic = False
                        except Exception:
                            pass
                        # notify host/guests
                        self.send({'type': 'guest_inject', 'id': req.get('id'), 'path': req.get('path')})
                        break
                # broadcast updated request list
                self.send({'type': 'guest_requests', 'requests': self._requests_summary()})

    def _load_candidates_from_db(self):
        """Load cached feature dicts from musicscan.db (same format used by select_initial)."""
        try:
            dbfile = os.path.join(os.path.dirname(__file__), 'musicscan.db')
            if not os.path.exists(dbfile):
                return []
            conn = sqlite3.connect(dbfile)
            cur = conn.cursor()
            try:
                cur.execute("SELECT features FROM tracks")
                rows = cur.fetchall()
            except Exception:
                rows = []
            try:
                conn.close()
            except Exception:
                pass
            candidates = []
            for (fstr,) in rows:
                try:
                    feat = json.loads(fstr)
                    candidates.append(feat)
                except Exception:
                    continue
            return candidates
        except Exception:
            return []

    def _update_dynamic_next(self):
        """Recompute and set `next_track` dynamically based on current `target` and `now_track`.

        This will not override an explicitly injected request (guest injection).
        """
        if self.selector is None:
            return
        # do not override an explicitly injected next
        try:
            with self.lock:
                if getattr(self, 'next_track', None) and not getattr(self, '_next_is_dynamic', False):
                    return
                prev = self.now_track
        except Exception:
            prev = getattr(self, 'now_track', None)

        candidates = self._load_candidates_from_db()
        if not candidates:
            return

        try:
            adv = dict(getattr(self, 'advanced', {}) or {})
            if getattr(self, 'vibe', None):
                adv['vibe'] = self.vibe
            best, scored = self.selector.select_next_track(self.state, candidates, recent_played=[prev] if prev else None, prev_track_path=prev, record_choice=False, advanced=adv, target_override=getattr(self, 'target', None))
            if best and best.get('path'):
                with self.lock:
                    self.next_track = best.get('path')
                    self._next_is_dynamic = True
                self.send({'type': 'next_queued', 'path': self.next_track})
                self.send({'type': 'party_state', 'state': self.state_dict()})
        except Exception:
            self.send({'type': 'error', 'message': 'update_dynamic_next_failed', 'detail': traceback.format_exc()})


if __name__ == '__main__':
    srv = Server()
    # keep process alive; threads handle I/O
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
