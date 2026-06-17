# Energy Music Mixer — Prototype 1

Energy Music Mixer (prototype) is a desktop prototype that combines a lightweight Electron UI with Python audio tooling to scan a local music library, extract per-track audio features, and drive a simple "party" selection engine that chooses tracks to match a target energy curve.

Key ideas:
- Scan a folder of audio files and compute features (BPM, key, energy, danceability, loudness, MFCCs, vocal coverage, "vibe" tags).
- Cache results in a local SQLite DB (`musicscan.db`) to avoid re-processing.
- Run a small party server that maintains party state (energy, trajectory, familiarity bias) and selects the next track to play using heuristics.
- Provide a desktop UI to start scans, control energy/targets, and view scan progress.

**Architecture & important files**

- **Electron UI**: [main.js](main.js), [renderer.js](renderer.js), [preload.js](preload.js), [index.html](index.html)
- **Music feature extraction**: [musicscan.py](musicscan.py) (uses `librosa`, `numpy`, `soundfile`)
- **Party server / selector**: [party_server.py](party_server.py), [party_engine.py](party_engine.py)
- **Project config**: [package.json](package.json), [requirements.txt](requirements.txt)
- **Tools & helpers**: [tools/check_musicscan_cache.py](tools/check_musicscan_cache.py) and other scripts under `tools/`

Features
- Scans local folders for `.mp3` and `.flac` files and computes audio features.
- Persistent cache of features and simple clustering / compatibility maps.
- Heuristic "TrackSelector" that picks next tracks to move the party toward a target energy.
- Guest suggestion HTTP endpoint (served by the party server) for simple guest requests and upvotes.

Quick Start (development)

1. Create and activate a Python virtual environment, then install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Install Node dependencies and run the Electron app:

```bash
npm install
npm start
```

Notes
- The Electron `main.js` attempts to prefer a `.venv` Python binary when spawning Python helpers (see `getPythonCmd()`), so activating the venv before running the app is recommended.
- `musicscan.py` reads newline-separated file paths on stdin and emits JSON lines with extracted features and compact summaries. See the script for details and heuristics.
- The party server (`party_server.py`) runs a small JSON stdin/stdout control loop and starts a minimal HTTP guest interface to accept suggestions.

Developer notes
- Run unit tests with `pytest` (there are `test_party_engine.py` and `test_scan_runner.py` present).
- To re-scan a folder from the command line, feed full paths to `musicscan.py` via stdin or use the Electron UI which streams paths to the scanner.

Contributing
- This is a prototype repository. Contributions, issues and PRs are welcome — please include a short description and a reproducible test when possible.

License
- No license file is included. Add a `LICENSE` if you want to publish under a specific license.

If you'd like, I can also:
- add a short screenshot and GIF for the README
- add usage examples for the party server and musicscan CLI

