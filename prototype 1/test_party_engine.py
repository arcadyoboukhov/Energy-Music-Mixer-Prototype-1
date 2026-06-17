import os
import json

import musicscan
import party_engine
import memory
import time


def ensure_test_wavs():
    os.makedirs('test_wavs', exist_ok=True)
    try:
        from test_scan_runner import make_sine
        p1 = make_sine('test_wavs/track_build.wav', ramp=True, freq=220)
        p2 = make_sine('test_wavs/track_drop.wav', drop=True, freq=440)
        return [p1, p2]
    except Exception:
        return []


def run_demo():
    paths = ensure_test_wavs()
    if not paths:
        print('No test wavs available; create some and retry.')
        return

    feats = [musicscan.compute_features(p) for p in paths]

    engine = party_engine.EnergyEngine()
    selector = party_engine.TrackSelector(engine)

    # Scenario A: 1m30s in, energy target high
    state_a = party_engine.PartyState(energy_level=65, trajectory='rising', time_elapsed=90, familiarity_bias=0.7, crowd_type='mixed', prev_track_path=paths[0])
    best_a, scored_a = selector.select_next_track(state_a, feats, recent_played=[paths[0]], prev_track_path=paths[0])

    print('--- Scenario A (target 90) ---')
    print(json.dumps({
        'target_energy': engine.target_at(state_a.time_elapsed),
        'best': best_a.get('path') if best_a else None,
        'scores': [
            {
                'path': s['track'].get('path'),
                'score': s['score'],
                'components': s['components']
            } for s in scored_a
        ]
    }, indent=2))

    # Scenario B: 1m00s in, target 70, current energy 65 -> prefer slightly higher energy
    state_b = party_engine.PartyState(energy_level=65, trajectory='rising', time_elapsed=60, familiarity_bias=0.7, crowd_type='mixed', prev_track_path=paths[0])
    # create synthetic candidates with different energies to test energy bias
    cand_low = dict(feats[0])
    cand_high = dict(feats[1])
    cand_low['energy'] = 55
    cand_high['energy'] = 72
    best_b, scored_b = selector.select_next_track(state_b, [cand_low, cand_high], recent_played=[paths[0]], prev_track_path=paths[0])

    print('\n--- Scenario B (target 70) ---')
    print(json.dumps({
        'target_energy': engine.target_at(state_b.time_elapsed),
        'best': best_b.get('path') if best_b else None,
        'scores': [
            {
                'path': s['track'].get('path'),
                'score': s['score'],
                'components': s['components']
            } for s in scored_b
        ]
    }, indent=2))

    # Scenario C: if a candidate was played very recently, it should be avoided
    print('\n--- Scenario C (recent play avoidance) ---')
    # record that track_drop was played 30 seconds ago
    try:
        memory.record_play(paths[1], played_at=time.time() - 30, skipped=False, moment_type='normal', artist_cluster=None, energy=feats[1].get('energy'))
    except Exception:
        pass
    best_c, scored_c = selector.select_next_track(state_b, feats, recent_played=[], prev_track_path=paths[0])
    print(json.dumps({
        'target_energy': engine.target_at(state_b.time_elapsed),
        'best': best_c.get('path') if best_c else None,
        'scores': [
            {
                'path': s['track'].get('path'),
                'score': s['score'],
                'components': s['components']
            } for s in scored_c
        ]
    }, indent=2))

    # Scenario D: latency test for selection decision time
    print('\n--- Scenario D (latency test) ---')
    import random, time
    N = 300
    synth = []
    for i in range(N):
        synth.append({
            'path': f'fake_{i}.mp3',
            'bpm': 100 + (i % 40) - 20,
            'energy': 30 + (i % 50),
            'key': {'tonic': 'C', 'mode': 'major'},
            'vibes': []
        })
    t0 = time.perf_counter()
    best_d, scored_d = selector.select_next_track(state_b, synth, recent_played=[], prev_track_path=None)
    dt = (time.perf_counter() - t0) * 1000.0
    print(f'Selection time for {N} candidates: {dt:.2f} ms, best: {best_d.get("path") if best_d else None}')


if __name__ == '__main__':
    run_demo()
