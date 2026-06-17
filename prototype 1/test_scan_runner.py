import os
import json
import numpy as np
import soundfile as sf

import musicscan


def make_sine(path, sr=22050, dur=6.0, freq=220, ramp=False, drop=False):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    if ramp:
        amp = np.linspace(0.05, 1.0, t.size)
    elif drop:
        amp = np.linspace(1.0, 0.05, t.size)
    else:
        amp = np.ones(t.size) * 0.6
    y = amp * 0.5 * np.sin(2 * np.pi * freq * t)
    sf.write(path, y.astype(np.float32), sr)
    return path


if __name__ == '__main__':
    os.makedirs('test_wavs', exist_ok=True)
    p1 = make_sine('test_wavs/track_build.wav', ramp=True, freq=220)
    p2 = make_sine('test_wavs/track_drop.wav', drop=True, freq=440)

    paths = [p1, p2]
    feats_list = []
    for p in paths:
        feats = musicscan.compute_features(p)
        print(json.dumps(feats))
        feats_list.append(feats)

    # build feature_matrix like musicscan.main
    feature_matrix = []
    for feats in feats_list:
        if isinstance(feats.get('mfcc_mean'), list):
            try:
                bpm_val = float(feats.get('bpm') or 0.0)
                bpm_norm = (bpm_val - 70.0) / (180.0 - 70.0)
                bpm_norm = max(0.0, min(1.0, bpm_norm))
            except Exception:
                bpm_norm = 0.0
            energy_pct = (feats.get('energy') or 0) / 100.0
            dance_pct = (feats.get('danceability') or 0) / 100.0
            vec = [bpm_norm, energy_pct, dance_pct] + list(feats['mfcc_mean'])
            feature_matrix.append(vec)

    clusters = {'0': paths}
    compat = musicscan.compute_compatibility_matrix(feats_list, feature_matrix, clusters)
    print(json.dumps({'__compatibility__': compat}, indent=2))
