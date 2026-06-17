"""production.py
Compute production / mixing heuristics for an audio file.

Heuristics implemented:
- Stereo width (mid/side RMS ratio)
- Per-frame pan mean/std and histogram
- Compression detection (dynamic range, crest factor, compression_score)
- EQ profile (band energies and spectral tilt)
- Integrated LUFS (using pyloudnorm if available; fallback approx)
- Layering density (average number of spectral peaks per frame)

This module is intentionally dependency-light and uses lazy imports
where possible. Results are returned as a JSON-serializable dict.
"""

from __future__ import annotations
import math
import json
import os
from typing import Dict, Any

import numpy as np
import soundfile as sf
from scipy import signal

EPS = 1e-12


def _resample_if_needed(data: np.ndarray, sr: int, target_sr: int | None):
    if target_sr is None or sr == target_sr:
        return data, sr
    try:
        from scipy.signal import resample_poly

        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        data_rs = resample_poly(data, up, down, axis=0)
        return data_rs, target_sr
    except Exception:
        return data, sr


def compute_production_features_from_file(path: str, target_sr: int | None = None) -> Dict[str, Any]:
    data, sr = sf.read(path, always_2d=True)
    data, sr = _resample_if_needed(data, sr, target_sr)
    data = np.asarray(data, dtype=np.float32)

    features: Dict[str, Any] = {}
    n_channels = data.shape[1]
    features["channels"] = int(n_channels)

    # Mono mix for many measures
    mono = np.mean(data, axis=1)

    # Stereo width / pan
    if n_channels >= 2:
        left = data[:, 0]
        right = data[:, 1]
    else:
        left = mono
        right = mono

    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    rms_mid = float(np.sqrt(np.mean(mid * mid) + EPS))
    rms_side = float(np.sqrt(np.mean(side * side) + EPS))
    stereo_width = float(rms_side / (rms_mid + EPS))
    features["stereo"] = {
        "stereo_width": stereo_width,
        "rms_mid": rms_mid,
        "rms_side": rms_side,
    }

    # Per-frame pan (energy-based)
    frame_ms = 50
    hop_ms = 25
    frame_len = max(256, int(sr * frame_ms / 1000))
    hop = max(128, int(sr * hop_ms / 1000))
    pans = []
    left_energy_frames = []
    right_energy_frames = []
    for i in range(0, max(1, len(left) - frame_len + 1), hop):
        L = left[i : i + frame_len]
        R = right[i : i + frame_len]
        eL = math.sqrt(float(np.mean(L * L) + EPS))
        eR = math.sqrt(float(np.mean(R * R) + EPS))
        pan = (eR - eL) / (eR + eL + EPS)
        pans.append(float(pan))
        left_energy_frames.append(eL)
        right_energy_frames.append(eR)
    pans_arr = np.array(pans) if len(pans) else np.array([0.0])
    pan_hist = np.histogram(pans_arr, bins=21, range=(-1.0, 1.0))[0].tolist()
    features["stereo"].update(
        {
            "mean_pan": float(float(np.mean(pans_arr))),
            "pan_std": float(float(np.std(pans_arr))),
            "pan_hist": pan_hist,
        }
    )

    # Compression heuristics
    rms_frames = []
    for i in range(0, max(1, len(mono) - frame_len + 1), hop):
        f = mono[i : i + frame_len]
        rms_frames.append(math.sqrt(float(np.mean(f * f) + EPS)))
    rms_frames_arr = np.array(rms_frames) if len(rms_frames) else np.array([1e-9])
    dyn_range_db = float(20.0 * math.log10((np.percentile(rms_frames_arr, 95) / (np.percentile(rms_frames_arr, 5) + EPS)) + EPS))
    peak = float(np.max(np.abs(mono)))
    mean_rms = float(np.mean(rms_frames_arr) + EPS)
    crest_db = float(20.0 * math.log10(peak / (mean_rms) + EPS))
    compression_score = float(np.clip((8.0 - dyn_range_db) / 8.0, 0.0, 1.0))
    features["compression"] = {
        "dynamic_range_db": dyn_range_db,
        "crest_db": crest_db,
        "compression_score": compression_score,
    }

    # EQ profile: band energies using Welch PSD
    try:
        freqs, psd = signal.welch(mono, sr, nperseg=16384)
    except Exception:
        freqs, psd = signal.welch(mono, sr, nperseg=8192)
    bands = [20, 60, 250, 500, 2000, 6000, 12000, 20000]
    band_energies_db = []
    band_centers = []
    for i in range(len(bands) - 1):
        f_low = bands[i]
        f_high = bands[i + 1]
        mask = (freqs >= f_low) & (freqs < f_high)
        if np.any(mask):
            # Use numpy.trapz when available, otherwise fallback to manual trapezoid
            y = psd[mask]
            x = freqs[mask]
            try:
                if hasattr(np, "trapz"):
                    en = float(np.trapz(y, x) + EPS)
                else:
                    if len(x) < 2:
                        en = float(np.sum(y) + EPS)
                    else:
                        en = float(((y[:-1] + y[1:]) * 0.5 * np.diff(x)).sum() + EPS)
            except Exception:
                # final fallback: sum of magnitudes
                en = float(np.sum(y) + EPS)
        else:
            en = float(EPS)
        band_energies_db.append(10.0 * math.log10(en))
        band_centers.append((f_low + f_high) / 2.0)
    mean_db = float(np.mean(band_energies_db))
    bands_relative_db = [float(b - mean_db) for b in band_energies_db]
    try:
        tilt = float(np.polyfit(np.log10(np.array(band_centers)), np.array(band_energies_db), 1)[0])
    except Exception:
        tilt = 0.0
    features["eq"] = {
        "bands_db": [float(x) for x in band_energies_db],
        "bands_relative_db": bands_relative_db,
        "band_centers_hz": band_centers,
        "spectral_tilt": tilt,
    }

    # LUFS (prefer pyloudnorm if available)
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        integrated = float(meter.integrated_loudness(mono))
        try:
            lra = float(meter.loudness_range(mono))
        except Exception:
            lra = None
        features["loudness"] = {
            "lufs": integrated,
            "loudness_range": lra,
            "recommended_normalization_lufs": -14.0,
        }
    except Exception:
        rms_all = math.sqrt(float(np.mean(mono * mono) + EPS))
        approx_lufs = float(20.0 * math.log10(rms_all + EPS))
        features["loudness"] = {"lufs_approx_db": approx_lufs, "recommended_normalization_lufs": -14.0, "approx": True}

    # Layering density: spectral-peak counts per frame
    nfft = 4096
    hop_spec = 512
    try:
        f, t, Z = signal.stft(mono, sr, nperseg=nfft, noverlap=nfft - hop_spec)
        mag = np.abs(Z)
        counts = []
        for j in range(mag.shape[1]):
            spec = mag[:, j]
            logspec = np.log1p(spec)
            try:
                peaks, props = signal.find_peaks(logspec, height=np.median(logspec) + np.std(logspec))
                counts.append(int(len(peaks)))
            except Exception:
                counts.append(0)
        layering_density = float(np.mean(counts)) if len(counts) else 0.0
    except Exception:
        layering_density = 0.0
        counts = []
    features["layering"] = {"average_simultaneous_peaks": layering_density, "frame_peaks_hist": np.histogram(counts, bins=20)[0].tolist()}

    return features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute production/mixing heuristics for an audio file")
    parser.add_argument("input", help="input audio file")
    parser.add_argument("--out-json", default="features_production.json", help="output JSON file")
    parser.add_argument("--target-sr", type=int, default=None)
    args = parser.parse_args()
    out = compute_production_features_from_file(args.input, target_sr=args.target_sr)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump({"production": out}, fh, indent=2)
