#!/usr/bin/env python3
"""Lightweight audio feature extractor.

Features extracted:
- sample rate, bit depth, duration
- amplitude (RMS over time)
- frequency spectrum (top peaks)
- spectrogram (saved image)
- estimated fundamental frequency and harmonics
- simple noise vs tonal estimate via spectral flatness

Usage examples:
  python audio_features.py --input file.wav --out-json features.json --spectrogram spec.png
  python audio_features.py --demo
"""
import argparse
import json
import os
import math
import tempfile
# Performance tuning: set sensible defaults for BLAS/OMP thread counts.
# These can be overridden by environment variables set externally.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import soundfile as sf
from scipy import signal
from numpy.lib import stride_tricks
import time
import functools

# We'll lazily import optional heavy libraries (pyFFTW, librosa, matplotlib)
def _lazy_import_pyfftw():
    try:
        import pyfftw
        import pyfftw.interfaces.numpy_fft as fftw_numpy_fft_local
        try:
            pyfftw.config.NUM_THREADS = int(os.environ.get("PYFFTW_NUM_THREADS", "4"))
        except Exception:
            pass
        return fftw_numpy_fft_local
    except Exception:
        return None


def _lazy_import_librosa():
    try:
        import librosa as _librosa
        return _librosa
    except Exception:
        return None


def _lazy_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_local
        return plt_local
    except Exception:
        return None


def _lazy_import_lyrics():
    try:
        import lyrics as lyrics_mod

        return lyrics_mod
    except Exception:
        try:
            # try package-relative import
            from . import lyrics as lyrics_mod

            return lyrics_mod
        except Exception:
            return None

EPS = 1e-10
import logging

logger = logging.getLogger(__name__)

# Benchmarking/timing support
TIMINGS = {}
BENCHMARK_ENABLED = False

def timed(name=None):
    def decorator(func):
        nm = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not BENCHMARK_ENABLED:
                return func(*args, **kwargs)
            t0 = time.perf_counter()
            res = func(*args, **kwargs)
            dt = time.perf_counter() - t0
            TIMINGS[nm] = TIMINGS.get(nm, 0.0) + dt
            return res

        return wrapper

    return decorator


def read_audio(path):
    logger.debug("read_audio: path=%s", path)
    info = sf.info(path)
    logger.debug("read_audio: info: samplerate=%s channels=%s subtype=%s", getattr(info, "samplerate", None), getattr(info, "channels", None), getattr(info, "subtype", None))
    y, sr = sf.read(path, dtype="float32")
    logger.debug("read_audio: raw data shape=%s dtype=%s sr=%s", getattr(y, "shape", None), getattr(y, "dtype", None), sr)
    if y.ndim > 1:
        logger.debug("read_audio: converting multi-channel audio to mono by averaging channels")
        y = np.mean(y, axis=1)
    logger.debug("read_audio: returning mono shape=%s", getattr(y, "shape", None))
    return y, sr, info


def bit_depth_from_subtype(subtype):
    logger.debug("bit_depth_from_subtype: subtype=%s", subtype)
    if not subtype:
        logger.debug("bit_depth_from_subtype: no subtype provided")
        return None
    s = subtype.upper()
    mapping = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "PCM_U8": 8, "FLOAT": 32, "DOUBLE": 64}
    for k, v in mapping.items():
        if k in s:
            logger.debug("bit_depth_from_subtype: matched mapping %s -> %s", k, v)
            return v
    digits = "".join([c for c in s if c.isdigit()])
    if digits:
        try:
            bd = int(digits)
            logger.debug("bit_depth_from_subtype: parsed digits -> %s", bd)
            return bd
        except Exception:
            logger.debug("bit_depth_from_subtype: failed to parse digits %s", digits)
            return None
    logger.debug("bit_depth_from_subtype: unable to determine bit depth")
    return None


@timed()
def rms_over_time(y, sr, frame_length=2048, hop_length=512):
    logger.debug("rms_over_time: signal_length=%d frame_length=%d hop_length=%d", len(y), frame_length, hop_length)
    y = np.asarray(y, dtype=np.float32)
    if len(y) < 1:
        logger.debug("rms_over_time: empty signal")
        return np.array([]), np.array([])
    if len(y) < frame_length:
        logger.debug("rms_over_time: signal shorter than frame_length, padding")
        frame = np.pad(y, (0, frame_length - len(y)))
        rms_val = float(np.sqrt(np.mean(frame * frame) + EPS))
        logger.debug("rms_over_time: single_frame_rms=%f", rms_val)
        return np.array([rms_val], dtype=np.float32), np.array([frame_length / 2.0 / sr], dtype=np.float32)
    # Try to use a vectorized framing via sliding_window_view if available
    try:
        frames = stride_tricks.sliding_window_view(y, frame_length)[::hop_length]
        # frames shape: (n_frames, frame_length)
        rms = np.sqrt(np.mean(frames * frames, axis=1) + EPS).astype(np.float32)
        times = ((np.arange(len(rms)) * hop_length) + frame_length / 2.0) / sr
        logger.debug("rms_over_time: vectorized first_rms=%s", rms[:5].tolist())
        return rms, times.astype(np.float32)
    except Exception:
        # fallback to the safe loop-based computation
        frames = 1 + (len(y) - frame_length) // hop_length
        logger.debug("rms_over_time: computed frame_count=%d (fallback)", frames)
        rms = np.empty(frames, dtype=np.float32)
        times = np.empty(frames, dtype=np.float32)
        for i in range(frames):
            start = i * hop_length
            frame = y[start:start + frame_length]
            rms[i] = float(np.sqrt(np.mean(frame * frame) + EPS))
            times[i] = (start + frame_length / 2.0) / sr
        logger.debug("rms_over_time: fallback first_rms=%s", rms[:5].tolist())
        return rms, times


@timed()
def compute_stft(y, sr, n_fft=2048, hop_length=512, use_gpu=False, use_pyfftw=False):
    """Compute magnitude STFT and dB-scaled STFT.

    Supports an optional GPU path (PyTorch) and an optional pyFFTW-accelerated CPU path.
    """
    logger.debug("compute_stft: n_fft=%d hop_length=%d signal_length=%d use_gpu=%s use_pyfftw=%s", n_fft, hop_length, len(y), use_gpu, use_pyfftw)
    y = np.asarray(y, dtype=np.float32)
    # GPU path (lazy import of torch)
    if use_gpu:
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.debug("compute_stft: torch available, device=%s", device)
            if device.type == "cuda":
                # pad to at least n_fft
                if len(y) < n_fft:
                    pad = np.zeros((n_fft - len(y),), dtype=np.float32)
                    y_proc = np.concatenate([y, pad])
                else:
                    y_proc = y
                t_y = torch.from_numpy(y_proc).to(device)
                window = torch.hann_window(n_fft, device=device, dtype=torch.float32)
                # center=False to match non-centered STFT used elsewhere
                Z = torch.stft(t_y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, center=False, return_complex=True)
                S_t = torch.abs(Z)
                S = S_t.cpu().numpy()
                n_frames = S.shape[1]
                f = np.linspace(0.0, float(sr) / 2.0, int(n_fft // 2 + 1))
                t = ((np.arange(n_frames) * hop_length) + n_fft / 2.0) / float(sr)
                S_db = 20.0 * np.log10(S + EPS)
                logger.debug("compute_stft: GPU STFT computed f_bins=%d t_bins=%d", len(f), len(t))
                return f, t, S, S_db
            else:
                logger.warning("compute_stft: --use-gpu requested but CUDA not available; falling back to CPU path")
        except Exception:
            logger.exception("compute_stft: torch-based STFT failed; falling back to CPU path")

    # CPU path: vectorized framing + rFFT (optionally via pyFFTW)
    try:
        if len(y) < n_fft:
            y_proc = np.pad(y, (0, n_fft - len(y)))
        else:
            y_proc = y
        # Create overlapping frames via sliding window view
        frames = stride_tricks.sliding_window_view(y_proc, window_shape=n_fft)[::hop_length]
        # reuse hann window if available
        window = _WINDOW_CACHE.get(n_fft)
        if window is None:
            window = np.hanning(n_fft).astype(np.float32)
            _WINDOW_CACHE[n_fft] = window
        frames_windowed = frames * window[None, :]
        # Choose FFT backend (lazy import pyFFTW if requested)
        rfft = np.fft.rfft
        if use_pyfftw:
            fftw_mod = _lazy_import_pyfftw()
            if fftw_mod is not None:
                rfft = fftw_mod.rfft
            else:
                logger.debug("compute_stft: pyFFTW requested but not available; using numpy FFT")
        # compute rFFT along axis=1 -> shape (n_frames, n_fft//2+1)
        F = rfft(frames_windowed, axis=1)
        S = np.abs(F).T.astype(np.float32)
        n_frames = S.shape[1]
        f = np.fft.rfftfreq(n_fft, 1.0 / float(sr))
        t = ((np.arange(n_frames) * hop_length) + n_fft / 2.0) / float(sr)
        S_db = 20.0 * np.log10(S + EPS)
        logger.debug("compute_stft: CPU STFT computed f_bins=%d t_bins=%d pyfftw_requested=%s", len(f), len(t), bool(use_pyfftw))
        return f, t, S, S_db
    except Exception:
        logger.exception("compute_stft: CPU STFT fallback to scipy.signal.stft")
        f, t, Zxx = signal.stft(y, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None, padded=False)
        S = np.abs(Zxx)
        S_db = 20.0 * np.log10(S + EPS)
        return f, t, S, S_db


def top_spectrum_peaks(avg_mags, freqs, top_n=10):
    mags = np.array(avg_mags)
    logger.debug("top_spectrum_peaks: bins=%d top_n=%d", len(mags), top_n)
    if len(mags) == 0:
        logger.debug("top_spectrum_peaks: empty magnitudes")
        return []
    mags[0] = 0.0
    idx = np.argsort(mags)[-top_n:][::-1]
    peaks = []
    for i in idx:
        peaks.append({"freq": float(freqs[i]), "amplitude": float(mags[i])})
    logger.debug("top_spectrum_peaks: peaks=%s", peaks[:5])
    return peaks


@timed()
def autocorr_f0(y, sr, frame_length=2048, hop_length=512, fmin=50.0, fmax=2000.0, use_pyfftw=False):
    """Vectorized autocorrelation-based F0 estimation per frame.

    Uses FFT-based autocorrelation (via power spectrum -> irfft) to compute
    per-frame autocorrelations in a vectorized manner for speed.
    """
    logger.debug("autocorr_f0: frame_length=%d hop_length=%d fmin=%f fmax=%f", frame_length, hop_length, fmin, fmax)
    y = np.asarray(y, dtype=np.float32)
    min_lag = max(1, int(sr / fmax))
    max_lag = max(2, int(sr / fmin))

    # Frame the signal (pad last frame if needed). Use a safe fallback if
    # `sliding_window_view` is unavailable or fails for this input.
    if len(y) < frame_length:
        frames = np.pad(y, (0, frame_length - len(y))).astype(np.float32)[None, :]
    else:
        try:
            frames = stride_tricks.sliding_window_view(y, frame_length)[::hop_length]
        except Exception:
            logger.debug("autocorr_f0: sliding_window_view failed; using safe fallback framing")
            # fallback: explicit framing (may be slower but robust)
            n_frames = 1 + (len(y) - frame_length) // hop_length
            if n_frames <= 0:
                frames = np.pad(y, (0, frame_length - len(y))).astype(np.float32)[None, :]
            else:
                frames = np.empty((n_frames, frame_length), dtype=np.float32)
                for i in range(n_frames):
                    start = i * hop_length
                    frames[i, :] = np.asarray(y[start:start + frame_length], dtype=np.float32)
    if getattr(frames, "size", 0) == 0:
        logger.debug("autocorr_f0: no frames to process")
        return None, []

    # Remove DC and apply window; ensure numeric dtype
    frames = np.asarray(frames, dtype=np.float32)
    frames = frames - np.mean(frames, axis=1, keepdims=True)
    window = np.hanning(frame_length).astype(np.float32)
    frames_windowed = frames * window[None, :]

    # FFT-based autocorrelation (batch rFFT + irFFT of power spectrum)
    # allow optional pyFFTW backend for CPU acceleration (lazy import)
    if use_pyfftw:
        fftw_mod = _lazy_import_pyfftw()
        if fftw_mod is not None:
            rfft = fftw_mod.rfft
            irfft = fftw_mod.irfft if hasattr(fftw_mod, 'irfft') else np.fft.irfft
        else:
            logger.debug("autocorr_f0: pyFFTW requested but not available; using numpy FFT")
            rfft = np.fft.rfft
            irfft = np.fft.irfft
    else:
        rfft = np.fft.rfft
        irfft = np.fft.irfft
    F = rfft(frames_windowed, axis=1)
    PSD = np.abs(F) ** 2
    corr = irfft(PSD, n=frame_length, axis=1)

    # search lags
    if max_lag >= corr.shape[1]:
        max_lag = corr.shape[1] - 1
    search = corr[:, min_lag: max_lag + 1]
    if search.size == 0:
        logger.debug("autocorr_f0: empty search region for lags")
        return None, [0.0] * frames.shape[0]

    peaks = np.argmax(search, axis=1) + min_lag
    peak_vals = search[np.arange(search.shape[0]), np.argmax(search, axis=1)]
    f0s = np.where(peak_vals > 0, sr / peaks, 0.0)
    voiced = f0s > 0
    logger.debug("autocorr_f0: voiced_count=%d total_frames=%d", int(np.sum(voiced)), len(f0s))
    if np.sum(voiced) == 0:
        return None, f0s.tolist()
    fundamental = float(np.median(f0s[voiced]))
    logger.debug("autocorr_f0: estimated_fundamental=%s", fundamental)
    return fundamental, f0s.tolist()


def harmonic_strengths(avg_mags, freqs, f0, n_harmonics=6):
    logger.debug("harmonic_strengths: f0=%s n_harmonics=%d", f0, n_harmonics)
    strengths = []
    if not f0:
        logger.debug("harmonic_strengths: no fundamental provided")
        return strengths
    mags = np.array(avg_mags)
    for k in range(1, n_harmonics + 1):
        target = k * f0
        idx = int(np.argmin(np.abs(freqs - target)))
        strengths.append({"harmonic": k, "freq": float(freqs[idx]), "amplitude": float(mags[idx])})
    logger.debug("harmonic_strengths: strengths=%s", strengths)
    return strengths


def spectral_flatness(S):
    logger.debug("spectral_flatness: S_shape=%s", S.shape)
    gm = np.exp(np.mean(np.log(S + EPS), axis=0))
    am = np.mean(S, axis=0) + EPS
    sf = gm / am
    logger.debug("spectral_flatness: median=%f mean=%f", float(np.median(sf)), float(np.mean(sf)))
    return sf


@timed()
def compute_chroma_from_S(S, freqs, sr=None):
    """Compute a 12-bin chroma matrix from a magnitude spectrogram S and frequency bins.

    S: shape (n_freq_bins, n_frames)
    freqs: array of length n_freq_bins with center frequencies (Hz)
    returns chroma: shape (12, n_frames)
    """
    logger.debug("compute_chroma_from_S: S_shape=%s freqs_len=%d sr=%s", S.shape, len(freqs), sr)
    S = np.asarray(S, dtype=np.float32)
    freqs = np.asarray(freqs, dtype=np.float32)
    n_bins, n_frames = S.shape
    # map frequency bins to nearest MIDI pitch class and aggregate via matrix multiply
    valid = freqs > 0
    if not np.any(valid):
        logger.debug("compute_chroma_from_S: no valid frequency bins")
        return np.zeros((12, n_frames), dtype=np.float32)
    freqs_valid = freqs[valid]
    S_valid = S[valid, :]
    # compute MIDI pitch per bin and map to pitch class
    midi = 69.0 + 12.0 * np.log2(freqs_valid / 440.0)
    midi_rounded = np.round(midi).astype(int)
    pitch_classes = midi_rounded % 12

    # build mapping matrix (12 x n_valid) using advanced indexing
    n_valid = len(pitch_classes)
    mapping = np.zeros((12, n_valid), dtype=np.float32)
    mapping[pitch_classes, np.arange(n_valid)] = 1.0

    # aggregate into chroma via a single matrix multiplication (BLAS-accelerated)
    chroma = mapping.dot(S_valid)
    # normalize per frame
    chroma = chroma / (np.sum(chroma, axis=0, keepdims=True) + EPS)
    logger.debug("compute_chroma_from_S: chroma_shape=%s", chroma.shape)
    return chroma.astype(np.float32)


MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)
NOTE_NAMES = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]

# Precompute chord triad templates (major/minor) to avoid rebuilding them repeatedly.
# Templates shape: (24, 12) where rows are templates and columns are pitch classes.
_CHORD_TEMPLATES_LIST = []
_CHORD_TEMPLATE_NAMES = []
for _r in range(12):
    _maj = np.zeros(12, dtype=np.float32)
    _maj[[0, 4, 7]] = 1.0
    _maj = np.roll(_maj, _r)
    _CHORD_TEMPLATES_LIST.append(_maj)
    _CHORD_TEMPLATE_NAMES.append(f"{NOTE_NAMES[_r]}:maj")
    _min_t = np.zeros(12, dtype=np.float32)
    _min_t[[0, 3, 7]] = 1.0
    _min_t = np.roll(_min_t, _r)
    _CHORD_TEMPLATES_LIST.append(_min_t)
    _CHORD_TEMPLATE_NAMES.append(f"{NOTE_NAMES[_r]}:min")

CHORD_TEMPLATES_MATRIX = np.vstack(_CHORD_TEMPLATES_LIST).astype(np.float32)  # (24,12)
CHORD_TEMPLATE_NAMES = _CHORD_TEMPLATE_NAMES
CHORD_TEMPLATE_NORMS = np.linalg.norm(CHORD_TEMPLATES_MATRIX, axis=1)

# Simple cache for chroma mapping matrices keyed by number of frequency bins.
_CHROMA_MAPPING_CACHE = {}
# Cache for commonly used windows to avoid recomputing them repeatedly
_WINDOW_CACHE = {}


@timed()
def detect_key_from_chroma(chroma):
    """Estimate global key (root + mode) from a chroma matrix using Krumhansl profiles."""
    logger.debug("detect_key_from_chroma: chroma_shape=%s", chroma.shape)
    mean_chroma = np.mean(chroma, axis=1)
    best = {"root": None, "mode": None, "confidence": -1.0}
    for r in range(12):
        rotated = np.roll(mean_chroma, -r)
        # major
        num = float(np.dot(rotated, MAJOR_PROFILE))
        den = (np.linalg.norm(rotated) * np.linalg.norm(MAJOR_PROFILE) + EPS)
        corr_major = num / den
        if corr_major > best["confidence"]:
            best.update({"root": NOTE_NAMES[r], "mode": "major", "confidence": float(corr_major)})
        # minor
        num = float(np.dot(rotated, MINOR_PROFILE))
        den = (np.linalg.norm(rotated) * np.linalg.norm(MINOR_PROFILE) + EPS)
        corr_minor = num / den
        if corr_minor > best["confidence"]:
            best.update({"root": NOTE_NAMES[r], "mode": "minor", "confidence": float(corr_minor)})
    logger.debug("detect_key_from_chroma: best=%s", best)
    return best


@timed()
def key_over_time_from_chroma(chroma, frame_times, window_size=8.0, hop=2.0):
    logger.debug("key_over_time_from_chroma: frames=%d duration_est=%s window=%f hop=%f", chroma.shape[1], frame_times[-1] if len(frame_times) else None, window_size, hop)
    if chroma.shape[1] == 0:
        return []
    duration = float(frame_times[-1])
    centers = np.arange(window_size / 2.0, max(duration - window_size / 2.0, window_size / 2.0) + 1e-9, hop)
    out = []
    for c in centers:
        w0 = c - window_size / 2.0
        w1 = c + window_size / 2.0
        idxs = np.where((frame_times >= w0) & (frame_times <= w1))[0]
        if idxs.size == 0:
            continue
        ch = chroma[:, idxs]
        k = detect_key_from_chroma(ch)
        out.append({"time": float(c), "key": f"{k['root']} {k['mode']}", "confidence": k["confidence"]})
    logger.debug("key_over_time_from_chroma: entries=%d", len(out))
    return out


@timed()
def detect_chords_from_chroma(chroma, frame_times, threshold=0.3):
    logger.debug("detect_chords_from_chroma: frames=%d threshold=%f", chroma.shape[1], threshold)
    # Vectorized template matching using precomputed templates.
    n_frames = chroma.shape[1]
    if n_frames == 0:
        return []

    # norms
    frame_norms = np.linalg.norm(chroma, axis=0) + EPS  # (n_frames,)
    template_norms = CHORD_TEMPLATE_NORMS  # (n_templates,)

    # dot products: (n_templates, n_frames)
    dots = CHORD_TEMPLATES_MATRIX.dot(chroma)
    denom = template_norms[:, None] * frame_norms[None, :]
    corr = dots / (denom + EPS)

    best_idx = np.argmax(corr, axis=0)
    best_conf = corr[best_idx, np.arange(n_frames)]

    frame_labels = np.array(["N"] * n_frames, dtype=object)
    confidences = np.zeros(n_frames, dtype=np.float32)
    mask = best_conf >= threshold
    if np.any(mask):
        names_arr = np.array(CHORD_TEMPLATE_NAMES)
        frame_labels[mask] = names_arr[best_idx[mask]]
        confidences[mask] = best_conf[mask].astype(np.float32)

    # collapse contiguous frames with same label into segments (preserve original behavior)
    segments = []
    cur_label = frame_labels[0]
    cur_start_idx = 0
    for j in range(1, n_frames):
        if frame_labels[j] != cur_label:
            start_time = float(frame_times[cur_start_idx])
            end_time = float(frame_times[j])
            if cur_label != "N":
                seg_conf = float(np.mean(confidences[cur_start_idx:j]) if np.sum(confidences[cur_start_idx:j]) > 0 else 0.0)
                segments.append({"start": start_time, "end": end_time, "chord": cur_label, "confidence": seg_conf})
            cur_label = frame_labels[j]
            cur_start_idx = j
    # last segment
    start_time = float(frame_times[cur_start_idx])
    end_time = float(frame_times[-1])
    if cur_label != "N":
        seg_conf = float(np.mean(confidences[cur_start_idx:]) if np.sum(confidences[cur_start_idx:]) > 0 else 0.0)
        segments.append({"start": start_time, "end": end_time, "chord": cur_label, "confidence": seg_conf})
    logger.debug("detect_chords_from_chroma: segments=%d", len(segments))
    return segments


@timed()
def melody_contours_from_f0(f0_values, sr, frame_length, hop_length):
    """Convert per-frame F0 values into melody note contours (start,end,median_freq,median_midi)."""
    logger.debug("melody_contours_from_f0: frames=%d frame_length=%d hop_length=%d", len(f0_values), frame_length, hop_length)
    f0 = np.array(f0_values, dtype=np.float32)
    n = len(f0)
    if n == 0:
        return {"contours": [], "f0_times": [], "f0_values": []}
    times = ((np.arange(n) * hop_length) + frame_length / 2.0) / float(sr)
    voiced = f0 > 0
    contours = []
    i = 0
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        vals = []
        while j < n and voiced[j]:
            vals.append(f0[j])
            j += 1
        seg_times = times[i:j]
        seg_vals = np.array(vals)
        median_freq = float(np.median(seg_vals))
        median_midi = float(69.0 + 12.0 * np.log2(median_freq / 440.0)) if median_freq > 0 else None
        contours.append({"start": float(seg_times[0]), "end": float(seg_times[-1]), "median_freq": median_freq, "median_midi": median_midi})
        i = j
    logger.debug("melody_contours_from_f0: contours=%d", len(contours))
    return {"contours": contours, "f0_times": times.tolist(), "f0_values": [float(x) for x in f0]}


@timed()
def pitch_distribution_from_f0(f0_values):
    f0 = np.array(f0_values, dtype=np.float32)
    f0 = f0[f0 > 0]
    if f0.size == 0:
        return {"midi_hist": {}, "pitch_class_hist": [0.0] * 12, "range_midi": [None, None]}
    midi = np.round(69.0 + 12.0 * np.log2(f0 / 440.0)).astype(int)
    min_midi = int(np.min(midi))
    max_midi = int(np.max(midi))
    counts = np.bincount(midi - min_midi)
    midi_hist = {str(int(min_midi + i)): int(int(c)) for i, c in enumerate(counts)}
    pcs = midi % 12
    pc_counts = np.bincount(pcs, minlength=12).astype(float)
    pc_hist = (pc_counts / (np.sum(pc_counts) + EPS)).tolist()
    return {"midi_hist": midi_hist, "pitch_class_hist": pc_hist, "range_midi": [min_midi, max_midi]}


@timed()
def consonance_from_chroma(chroma, frame_times):
    logger.debug("consonance_from_chroma: frames=%d", chroma.shape[1])
    # Vectorized consonance proxy: max template correlation per frame using precomputed templates
    if chroma.shape[1] == 0:
        return {"time_series": [], "mean_consonance": 0.0, "median_consonance": 0.0, "dissonance": 1.0}
    frame_norms = np.linalg.norm(chroma, axis=0) + EPS
    dots = CHORD_TEMPLATES_MATRIX.dot(chroma)
    denom = CHORD_TEMPLATE_NORMS[:, None] * frame_norms[None, :]
    corr = dots / (denom + EPS)
    confs = np.max(corr, axis=0)
    mean_conf = float(np.mean(confs))
    median_conf = float(np.median(confs))
    dissonance = 1.0 - mean_conf
    return {"time_series": confs.tolist(), "mean_consonance": mean_conf, "median_consonance": median_conf, "dissonance": dissonance}


@timed()
def compute_mfcc_from_S(S, freqs, sr, n_mfcc=13, n_mels=40):
    """Compute MFCCs from a magnitude spectrogram S and frequency bin centers freqs.

    This is a lightweight fallback implementation that avoids importing librosa
    unless the user specifically requests higher-quality computation.
    Returns MFCC matrix (n_mfcc, n_frames) and summary stats (mean, std).
    """
    logger.debug("compute_mfcc_from_S: S_shape=%s freqs_len=%d sr=%s n_mfcc=%d n_mels=%d", S.shape, len(freqs), sr, n_mfcc, n_mels)
    S = np.asarray(S, dtype=np.float32)
    freqs = np.asarray(freqs, dtype=np.float32)
    n_bins, n_frames = S.shape
    fmin = float(max(20.0, freqs[1] if len(freqs) > 1 else 20.0))
    fmax = float(freqs[-1] if len(freqs) and freqs[-1] > 0 else float(sr) / 2.0)

    # build mel filterbank using triangular filters mapped to the provided freqs
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    m_min = hz_to_mel(fmin)
    m_max = hz_to_mel(fmax)
    m_points = np.linspace(m_min, m_max, n_mels + 2)
    hz_points = mel_to_hz(m_points)

    # allocate filterbank (n_mels x n_bins)
    filterbank = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(n_mels):
        left = hz_points[m]
        center = hz_points[m + 1]
        right = hz_points[m + 2]
        # avoid degenerate bands
        if center <= left or right <= center:
            continue
        # rising slope
        left_mask = (freqs >= left) & (freqs <= center)
        if np.any(left_mask):
            filterbank[m, left_mask] = (freqs[left_mask] - left) / (center - left)
        # falling slope
        right_mask = (freqs >= center) & (freqs <= right)
        if np.any(right_mask):
            filterbank[m, right_mask] = (right - freqs[right_mask]) / (right - center)

    # apply filterbank -> mel spectrogram
    mel_spec = filterbank.dot(S)
    # log-energy
    log_mel = np.log(mel_spec + EPS)

    # DCT (type II) along mel axis to get MFCCs
    try:
        from scipy.fftpack import dct
    except Exception:
        from scipy import fftpack as _fftpack

        def dct(x, type=2, axis=0, norm="ortho"):
            return _fftpack.dct(x, type=type, axis=axis, norm=norm)

    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[0:n_mfcc, :]
    mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)
    mfcc_std = np.std(mfcc, axis=1).astype(np.float32)
    logger.debug("compute_mfcc_from_S: computed mfcc shape=%s mean_first=%s", mfcc.shape, float(mfcc_mean[0]) if mfcc_mean.size else None)
    return mfcc.astype(np.float32), mfcc_mean.tolist(), mfcc_std.tolist()


@timed()
def spectral_centroid_and_rolloff(S, freqs, roll_percent=0.85):
    freqs = np.asarray(freqs, dtype=np.float32)
    S = np.asarray(S, dtype=np.float32)
    energy = np.sum(S, axis=0) + EPS
    centroid = (np.sum((freqs[:, None] * S), axis=0) / energy).astype(np.float32)
    # rolloff per frame
    cumsum = np.cumsum(S, axis=0)
    totals = cumsum[-1, :] + EPS
    thresh = roll_percent * totals
    rolloff = np.zeros_like(thresh, dtype=np.float32)
    for i in range(cumsum.shape[1]):
        idx = np.searchsorted(cumsum[:, i], thresh[i])
        if idx >= len(freqs):
            idx = len(freqs) - 1
        rolloff[i] = float(freqs[idx])
    return centroid.tolist(), rolloff.tolist()


@timed()
def high_frequency_energy_ratio(S, freqs, threshold_hz=5000.0):
    freqs = np.asarray(freqs, dtype=np.float32)
    S = np.asarray(S, dtype=np.float32)
    total = np.sum(S, axis=0) + EPS
    hf_mask = freqs >= threshold_hz
    if not np.any(hf_mask):
        return 0.0
    hf = np.sum(S[hf_mask, :], axis=0)
    ratio = float(np.mean(hf / total))
    logger.debug("high_frequency_energy_ratio: threshold=%f ratio=%f", threshold_hz, ratio)
    return ratio


@timed()
def clipping_proportion(y, thresh=0.999):
    if y is None or len(y) == 0:
        return 0.0
    prop = float(np.mean(np.abs(y) >= thresh))
    logger.debug("clipping_proportion: thresh=%f prop=%f", thresh, prop)
    return prop


@timed()
def reverb_score_from_onsets(onset_times, rms, rms_times, tail_seconds=0.5):
    if len(onset_times) == 0 or len(rms_times) == 0:
        return 0.0
    ratios = []
    rms = np.asarray(rms, dtype=np.float32)
    rms_times = np.asarray(rms_times, dtype=np.float32)
    for t in onset_times:
        idx = np.searchsorted(rms_times, t)
        if idx >= len(rms_times) - 1:
            continue
        peak = float(rms[idx]) if idx < len(rms) else float(np.max(rms))
        tail_end_time = t + tail_seconds
        tail_idx_end = np.searchsorted(rms_times, tail_end_time)
        if tail_idx_end <= idx:
            continue
        tail_mean = float(np.mean(rms[idx + 1: tail_idx_end]))
        if peak > EPS:
            ratios.append(tail_mean / peak)
    if len(ratios) == 0:
        return 0.0
    score = float(np.median(ratios))
    logger.debug("reverb_score_from_onsets: median_tail_ratio=%f", score)
    return float(min(1.0, score * 2.0))


@timed()
def autotune_score_from_f0(f0_values):
    f0 = np.array(f0_values, dtype=np.float32)
    f0 = f0[f0 > 0]
    if f0.size == 0:
        return 0.0
    midi_f = 69.0 + 12.0 * np.log2(f0 / 440.0)
    residual = 100.0 * (midi_f - np.round(midi_f))
    std_cents = float(np.std(residual))
    frac_close = float(np.mean(np.abs(residual) < 20.0))
    score = max(0.0, (1.0 - (std_cents / 50.0))) * frac_close
    logger.debug("autotune_score_from_f0: std_cents=%f frac_close=%f score=%f", std_cents, frac_close, score)
    return float(np.clip(score, 0.0, 1.0))


@timed()
def distortion_score(y, S, freqs, sf_median):
    clip = clipping_proportion(y)
    hf = high_frequency_energy_ratio(S, freqs, threshold_hz=6000.0)
    # normalize components and combine heuristically
    clip_norm = min(1.0, clip * 50.0)
    hf_norm = min(1.0, hf * 5.0)
    sf_norm = float(np.clip(sf_median, 0.0, 1.0))
    score = 0.6 * clip_norm + 0.25 * hf_norm + 0.15 * sf_norm
    score = float(np.clip(score, 0.0, 1.0))
    logger.debug("distortion_score: clip=%f hf=%f sf_median=%f score=%f", clip, hf, sf_median, score)
    return score


@timed()
def roughness_score(S):
    # simple proxy for roughness: mean spectral flux across frames (positive diffs)
    if S.shape[1] < 2:
        return 0.0
    diff = np.diff(S, axis=1)
    flux = np.sum(np.clip(np.abs(diff), a_min=0.0, a_max=None), axis=0)
    score = float(np.mean(flux) / (np.mean(S) + EPS))
    logger.debug("roughness_score: score=%f", score)
    return float(np.clip(score, 0.0, 1.0))


@timed()
def classify_instrument_simple(onset_times, f0_values, sf_median, centroid_mean, hf_ratio, duration):
    """Very small heuristic instrument classifier that returns a best guess and per-class scores.

    This is intentionally lightweight and conservative; for production-grade
    instrument recognition use a trained classifier (e.g., pretrained CNN).
    """
    scores = {"drums": 0.0, "vocals": 0.0, "guitar": 0.0, "piano": 0.0, "synth": 0.0, "bass": 0.0, "other": 0.0}
    # onset density (onsets per second)
    onset_rate = (len(onset_times) / max(1.0, duration))
    voiced_ratio = float(np.mean(np.array(f0_values) > 0)) if len(f0_values) > 0 else 0.0

    # drums: many onsets and low voiced_ratio
    scores["drums"] = min(1.0, onset_rate / 6.0) * (1.0 - voiced_ratio)
    # vocals: voiced presence and low spectral flatness
    scores["vocals"] = min(1.0, voiced_ratio * 2.0) * (1.0 - float(np.clip(sf_median, 0.0, 1.0)))
    # bass: low centroid and low hf ratio
    scores["bass"] = float(np.clip(1.0 - (centroid_mean / 250.0), 0.0, 1.0)) * (1.0 - hf_ratio)
    # guitar: moderate voiced, low flatness, moderate centroid
    scores["guitar"] = float(np.clip(voiced_ratio * (1.0 - sf_median) * (1.0 - abs(centroid_mean - 800.0) / 1500.0), 0.0, 1.0))
    # piano: percussive harmonic (onset_rate moderate, voiced low/med)
    scores["piano"] = float(np.clip((onset_rate / 4.0) * (1.0 - sf_median) * (centroid_mean < 3000.0), 0.0, 1.0))
    # synth: high spectral flatness or high HF
    scores["synth"] = float(np.clip(sf_median * 0.8 + hf_ratio * 0.5, 0.0, 1.0))

    # normalize and pick best
    total = sum(scores.values()) + EPS
    for k in list(scores.keys()):
        scores[k] = float(scores[k] / total)
    best = max(scores.items(), key=lambda x: x[1])[0]
    logger.debug("classify_instrument_simple: onset_rate=%f voiced_ratio=%f centroid=%f hf_ratio=%f best=%s", onset_rate, voiced_ratio, centroid_mean, hf_ratio, best)
    return {"best": best, "scores": scores}



@timed()
def onset_envelope_from_S(S):
    """Compute a simple spectral-flux onset envelope from magnitude spectrogram S.

    Returns envelope (len = n_frames-1) and an aligned frame index offset (we align with frame times[1:]).
    """
    logger.debug("onset_envelope_from_S: S_shape=%s", S.shape)
    if S.shape[1] < 2:
        logger.debug("onset_envelope_from_S: too few frames for onset envelope")
        return np.array([])
    # positive differences across frames (spectral flux)
    diff = np.diff(S, axis=1)
    flux = np.sum(np.clip(diff, a_min=0.0, a_max=None), axis=0)
    # normalize
    if np.max(flux) > 0:
        flux = flux / (np.max(flux) + EPS)
    logger.debug("onset_envelope_from_S: envelope_len=%d max=%f mean=%f", len(flux), float(np.max(flux)) if len(flux) else 0.0, float(np.mean(flux)) if len(flux) else 0.0)
    return flux


@timed()
def resample_audio(y, orig_sr, target_sr):
    """Resample audio to target sample rate using scipy.signal.resample (FFT-based).

    This is a pragmatic choice that avoids adding extra dependencies; for large
    signals you may prefer resample_poly or a dedicated resampler.
    """
    logger.debug("resample_audio: orig_sr=%d target_sr=%d len=%d", orig_sr, target_sr, len(y))
    if target_sr == orig_sr:
        return y
    target_len = int(round(len(y) * float(target_sr) / float(orig_sr)))
    if target_len <= 0:
        logger.debug("resample_audio: computed non-positive target length")
        return y
    y_rs = signal.resample(y, target_len)
    logger.debug("resample_audio: resampled length=%d", len(y_rs))
    return y_rs.astype(np.float32)


def detect_onsets_from_env(env, frame_times, sr, hop_length, threshold_factor=0.3, min_distance_sec=0.03):
    logger.debug("detect_onsets_from_env: env_len=%d frame_times_len=%d", len(env), len(frame_times))
    if len(env) == 0:
        return []
    # env corresponds to frame_times[1:]
    times = frame_times[1:]
    mean = float(np.mean(env))
    std = float(np.std(env))
    thresh = mean + threshold_factor * std
    logger.debug("detect_onsets_from_env: mean=%f std=%f thresh=%f", mean, std, thresh)
    min_distance_frames = max(1, int(min_distance_sec * sr / hop_length))
    peaks, props = signal.find_peaks(env, height=thresh, distance=min_distance_frames)
    onset_times = times[peaks].tolist()
    logger.debug("detect_onsets_from_env: detected_onsets=%d first_times=%s", len(onset_times), onset_times[:5])
    return onset_times


def estimate_tempo_from_onsets(onset_times, duration, min_bpm=40.0, max_bpm=240.0):
    logger.debug("estimate_tempo_from_onsets: onset_count=%d duration=%f", len(onset_times), duration)
    if len(onset_times) < 2:
        logger.debug("estimate_tempo_from_onsets: not enough onsets")
        return None, []
    iois = np.diff(np.array(onset_times))
    iois = iois[iois > 0.03]
    if len(iois) == 0:
        logger.debug("estimate_tempo_from_onsets: iois too small after filtering")
        return None, []
    median_ioi = float(np.median(iois))
    bpm = 60.0 / median_ioi
    # fold into reasonable range
    while bpm < min_bpm:
        bpm *= 2.0
    while bpm > max_bpm:
        bpm /= 2.0
    logger.debug("estimate_tempo_from_onsets: median_ioi=%f bpm=%f", median_ioi, bpm)
    return float(round(bpm, 2)), iois.tolist()


def tempo_over_time(onset_times, duration, window_size=8.0, hop=2.0):
    logger.debug("tempo_over_time: onset_count=%d duration=%f window=%f hop=%f", len(onset_times), duration, window_size, hop)
    if len(onset_times) < 2:
        return []
    onset = np.array(onset_times)
    centers = np.arange(window_size / 2.0, duration - window_size / 2.0 + 1e-9, hop)
    out = []
    for c in centers:
        w0 = c - window_size / 2.0
        w1 = c + window_size / 2.0
        in_window = onset[(onset >= w0) & (onset <= w1)]
        if len(in_window) < 2:
            continue
        iois = np.diff(in_window)
        iois = iois[iois > 0.03]
        if len(iois) == 0:
            continue
        bpm = 60.0 / float(np.median(iois))
        out.append({"time": float(c), "bpm": float(round(bpm, 2))})
    logger.debug("tempo_over_time: entries=%d", len(out))
    return out


@timed()
def find_beats_from_onset_env(onset_env, frame_times, duration, bpm, sr, hop_length):
    logger.debug("find_beats_from_onset_env: env_len=%d bpm=%s duration=%f", len(onset_env), bpm, duration)
    if bpm is None or bpm <= 0:
        logger.debug("find_beats_from_onset_env: invalid bpm")
        return [], []
    period = 60.0 / bpm
    # frame_times correspond to STFT frames; onset_env is aligned to frame_times[1:]
    env_times = frame_times[1:]
    # sample candidate phase offsets
    n_phases = max(20, int(period * 10))
    phases = np.linspace(0.0, period, n_phases, endpoint=False)
    best_score = -1.0
    best_phase = 0.0
    for ph in phases:
        beats = np.arange(ph, duration, period)
        if len(beats) == 0:
            continue
        vals = np.interp(beats, env_times, onset_env, left=0.0, right=0.0)
        score = float(np.sum(vals))
        if score > best_score:
            best_score = score
            best_phase = ph
    # produce final beats
    beat_times = np.arange(best_phase, duration, period).tolist()
    beat_strengths = np.interp(beat_times, env_times, onset_env, left=0.0, right=0.0).tolist()
    logger.debug("find_beats_from_onset_env: beats_found=%d best_phase=%f score=%f", len(beat_times), best_phase, best_score)
    beats = []
    for bt, bs in zip(beat_times, beat_strengths):
        beats.append({"time": float(bt), "strength": float(bs)})
    return beats, beat_strengths


def estimate_time_signature(beat_strengths, max_group=8):
    logger.debug("estimate_time_signature: beat_count=%d", len(beat_strengths))
    if len(beat_strengths) < 4:
        return None
    x = np.array(beat_strengths)
    x = x - np.mean(x)
    if np.allclose(x, 0.0):
        return None
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(ac)//2:]
    # ignore lag 0
    ac[0] = 0
    # examine lags 2..max_group
    lag_range = np.arange(2, min(max_group+1, len(ac)))
    if len(lag_range) == 0:
        return None
    lag_scores = ac[lag_range]
    best = lag_range[int(np.argmax(lag_scores))]
    logger.debug("estimate_time_signature: best_lag=%d", int(best))
    # choose denominator heuristically
    denominator = 4
    if best == 3:
        denominator = 4
    if best >= 6 and best % 3 == 0:
        # e.g., 6 -> often 6/8
        denominator = 8
    return {"numerator": int(best), "denominator": int(denominator)}


def estimate_groove_and_swing(beat_times, onset_times):
    logger.debug("estimate_groove_and_swing: beats=%d onsets=%d", len(beat_times), len(onset_times))
    if len(beat_times) < 2 or len(onset_times) < 1:
        return {"swing_ratio": None, "swing_percent": None}
    beats = np.array(beat_times)
    onsets = np.array(onset_times)
    periods = np.diff(beats)
    if len(periods) == 0:
        return {"swing_ratio": None, "swing_percent": None}
    median_period = float(np.median(periods))
    offbeat_ratios = []
    for b in beats[:-1]:
        # expected offbeat at b + period/2
        window_start = b + 0.2 * median_period
        window_end = b + 0.8 * median_period
        candidates = onsets[(onsets >= window_start) & (onsets <= window_end)]
        if len(candidates) == 0:
            continue
        off = candidates[0] - b
        offbeat_ratios.append(off / median_period)
    if len(offbeat_ratios) == 0:
        return {"swing_ratio": None, "swing_percent": None}
    median_ratio = float(np.median(offbeat_ratios))
    # straight eighths => 0.5; swing moves toward ~0.66
    swing_percent = (median_ratio - 0.5) / 0.5 * 100.0
    logger.debug("estimate_groove_and_swing: median_ratio=%f swing_percent=%f", median_ratio, swing_percent)
    return {"swing_ratio": float(median_ratio), "swing_percent": float(swing_percent)}


@timed()
def _aggregate_windows_for_structure(feature_mat, frame_times, window_sec=3.0, hop_sec=1.0):
    """Aggregate feature frames into larger windows for structure analysis.

    feature_mat: (n_feat, n_frames)
    frame_times: times for each frame (len = n_frames)
    returns: (W, centers) where W is (n_feat, n_windows) and centers are window center times
    """
    if feature_mat is None or feature_mat.size == 0:
        return np.zeros((0, 0), dtype=np.float32), np.array([])
    if frame_times is None or len(frame_times) == 0:
        frame_times = np.arange(feature_mat.shape[1], dtype=np.float32)
    duration = float(frame_times[-1]) if len(frame_times) else 0.0
    # build centers
    if duration <= window_sec or duration == 0.0:
        centers = np.array([min(window_sec / 2.0, duration / 2.0)], dtype=np.float32)
    else:
        centers = np.arange(window_sec / 2.0, max(duration - window_sec / 2.0, window_sec / 2.0) + 1e-9, hop_sec, dtype=np.float32)
    W_cols = []
    for c in centers:
        w0 = c - window_sec / 2.0
        w1 = c + window_sec / 2.0
        idxs = np.where((frame_times >= w0) & (frame_times <= w1))[0]
        if idxs.size == 0:
            W_cols.append(np.zeros((feature_mat.shape[0],), dtype=np.float32))
        else:
            W_cols.append(np.mean(feature_mat[:, idxs], axis=1))
    if len(W_cols) == 0:
        return np.zeros((feature_mat.shape[0], 0), dtype=np.float32), centers
    W = np.vstack([c.reshape(1, -1) for c in np.array(W_cols)])
    # W currently shape (n_windows, n_feat) -> transpose
    W = W.T.astype(np.float32)
    # normalize columns
    norms = np.linalg.norm(W, axis=0) + EPS
    W = W / norms[None, :]
    return W, centers


@timed()
def _self_similarity(W):
    if W is None or W.size == 0 or W.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    R = np.dot(W.T, W)
    # clip numerical noise
    R = np.clip(R, -1.0, 1.0)
    return R


@timed()
def _novelty_from_recurrence(R, L=3):
    if R is None or R.size == 0:
        return np.array([])
    k = 2 * L + 1
    K = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            if i == L or j == L:
                K[i, j] = 0.0
            elif (i < L and j < L) or (i > L and j > L):
                K[i, j] = -1.0
            else:
                K[i, j] = 1.0
    conv = signal.convolve2d(R, K, mode="same", boundary="symm")
    nov = np.diag(conv)
    if nov.size == 0:
        return nov
    nov = nov - np.min(nov)
    mx = np.max(nov)
    if mx > 0:
        nov = nov / float(mx)
    return nov


@timed()
def _detect_boundaries_from_novelty(novelty, centers, min_seg_sec=3.0, hop_sec=1.0, peak_thresh_factor=1.0):
    if novelty is None or novelty.size == 0:
        return [0.0, float(centers[-1] if len(centers) else 0.0)]
    mean = float(np.mean(novelty))
    std = float(np.std(novelty))
    thresh = mean + peak_thresh_factor * std
    min_distance_frames = max(1, int(max(1, min_seg_sec / float(hop_sec))))
    peaks, props = signal.find_peaks(novelty, height=thresh, distance=min_distance_frames)
    peak_times = centers[peaks].tolist() if len(peaks) else []
    duration = float(centers[-1] + hop_sec / 2.0) if len(centers) else 0.0
    bounds = [0.0] + peak_times + [duration]
    # clamp and sort
    bounds = sorted(list({float(max(0.0, min(duration, b))) for b in bounds}))
    return bounds


@timed()
def _group_segments_by_similarity(seg_vecs, sim_threshold=0.75):
    n = seg_vecs.shape[0]
    if n == 0:
        return []
    norms = np.linalg.norm(seg_vecs, axis=1, keepdims=True) + EPS
    normed = seg_vecs / norms
    sim = np.dot(normed, normed.T)
    groups = [-1] * n
    gid = 0
    for i in range(n):
        if groups[i] != -1:
            continue
        groups[i] = gid
        for j in range(i + 1, n):
            if groups[j] == -1 and sim[i, j] >= sim_threshold:
                groups[j] = gid
        gid += 1
    return groups


@timed()
def compute_structure_features(chroma, chroma_times, y, sr, onset_times, rms, rms_times, window_sec=3.0, hop_sec=1.0, sim_threshold=0.75, min_seg_sec=3.0):
    """Detect song sections, repetition patterns, transitions, and phrase boundaries.

    Returns a dict with `segments`, `repetition_groups`, and `transitions`.
    """
    if chroma is None or chroma.size == 0 or chroma.shape[1] == 0:
        logger.debug("compute_structure_features: no chroma available; skipping structure analysis")
        return {"segments": [], "repetition_groups": [], "transitions": [], "phrase_boundaries": []}

    W, centers = _aggregate_windows_for_structure(chroma, chroma_times, window_sec=window_sec, hop_sec=hop_sec)
    if W.shape[1] == 0:
        return {"segments": [], "repetition_groups": [], "transitions": [], "phrase_boundaries": []}

    R = _self_similarity(W)
    L = max(1, int(round(window_sec / float(hop_sec))))
    novelty = _novelty_from_recurrence(R, L=L)
    bounds = _detect_boundaries_from_novelty(novelty, centers, min_seg_sec=min_seg_sec, hop_sec=hop_sec)

    # build segments
    segments = []
    seg_vecs = []
    seg_rms = []
    seg_onsets = []
    for i in range(len(bounds) - 1):
        s = float(bounds[i])
        e = float(bounds[i + 1])
        # windows overlapping segment
        idxs = np.where((centers >= s) & (centers < e))[0]
        if idxs.size == 0:
            # fallback: use nearest center
            nearest = int(min(max(0, int(round((s + e) / 2.0 / hop_sec))), W.shape[1] - 1))
            vec = W[:, nearest]
        else:
            vec = np.mean(W[:, idxs], axis=1)
        seg_vecs.append(vec.astype(np.float32))
        # rms in segment
        if len(rms_times) and len(rms):
            ridxs = np.where((np.asarray(rms_times) >= s) & (np.asarray(rms_times) < e))[0]
            mean_r = float(np.mean(np.asarray(rms[ridxs])) if len(ridxs) else float(np.mean(rms)))
        else:
            mean_r = 0.0
        seg_rms.append(mean_r)
        # onsets in segment -> phrase boundaries
        pbs = [float(t) for t in onset_times if (t >= s and t < e)]
        seg_onsets.append(pbs)
        segments.append({"start": s, "end": e, "duration": float(e - s), "mean_energy": mean_r, "phrase_boundaries": pbs})

    seg_vecs = np.vstack(seg_vecs) if len(seg_vecs) else np.zeros((0, 12), dtype=np.float32)
    groups = _group_segments_by_similarity(seg_vecs, sim_threshold=sim_threshold) if seg_vecs.size else []

    # build repetition groups
    rep_map = {}
    for idx, g in enumerate(groups):
        rep_map.setdefault(g, []).append(idx)
    repetition_groups = []
    for gid, members in rep_map.items():
        total_dur = sum(segments[m]["duration"] for m in members)
        repetition_groups.append({"group_id": int(gid), "members": members, "count": len(members), "total_duration": float(total_dur)})

    # label segments heuristically
    # choose chorus candidate: group with count>=2 and max total_duration
    chorus_gid = None
    repeated_groups = [g for g in repetition_groups if g.get("count", 0) >= 2]
    if repeated_groups:
        chorus_gid = max(repeated_groups, key=lambda x: x["total_duration"]) ["group_id"]

    for i, seg in enumerate(segments):
        gid = groups[i] if i < len(groups) else -1
        seg["group_id"] = int(gid)
        seg["group_count"] = int(len(rep_map.get(gid, [])))
        # simple labels
        label = "section"
        if gid == chorus_gid:
            label = "chorus"
        elif i == 0 and seg["duration"] < max(15.0, 0.1 * float(centers[-1] + hop_sec / 2.0)):
            label = "intro"
        elif i == len(segments) - 1 and seg["duration"] < max(15.0, 0.1 * float(centers[-1] + hop_sec / 2.0)):
            label = "outro"
        elif seg["group_count"] == 1:
            # if surrounded by chorus segments, call it bridge
            prev_is_chorus = (i > 0 and segments[i - 1].get("group_id") == chorus_gid)
            next_is_chorus = (i < len(segments) - 1 and segments[i + 1].get("group_id") == chorus_gid)
            if prev_is_chorus and next_is_chorus:
                label = "bridge"
            else:
                label = "section"
        else:
            # repeated but not chorus -> verse
            label = "verse" if seg["group_count"] >= 2 else "section"
        seg["label"] = label

    # transitions between consecutive segments
    transitions = []
    for i in range(len(segments) - 1):
        cur = segments[i]
        nxt = segments[i + 1]
        energy_change = (nxt.get("mean_energy", 0.0) - cur.get("mean_energy", 0.0)) / (cur.get("mean_energy", EPS) + EPS)
        ttime = float(nxt.get("start", 0.0))
        typ = "transition"
        if energy_change <= -0.4:
            typ = "drop"
        elif energy_change >= 0.4:
            typ = "lift"
        transitions.append({"time": ttime, "from_segment": i, "to_segment": i + 1, "type": typ, "energy_change": float(energy_change)})

    structure = {
        "segments": segments,
        "repetition_groups": repetition_groups,
        "transitions": transitions,
    }
    return structure


def save_spectrogram(f, t, S_db, path):
    logger.debug("save_spectrogram: saving spectrogram to %s (f bins=%d t bins=%d)", path, len(f), len(t))
    plt = _lazy_import_matplotlib()
    if plt is None:
        logger.warning("save_spectrogram: matplotlib not available; skipping spectrogram save")
        return
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(t, f, S_db, shading="gouraud")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.debug("save_spectrogram: saved %s", path)


@timed()
def compute_mood_features(y, sr, tempo_bpm, rms, centroid_mean, roughness, consonance, beats, beat_strengths, timbre, harmony):
    """Estimate perceptual "feel" attributes: valence, arousal, tension, danceability, energy, and tags.

    This uses lightweight heuristics combining rhythm (tempo/beat strength),
    timbre (brightness/energy), and harmony (major/minor confidence).
    """
    try:
        logger.debug("compute_mood_features: tempo=%s rms_len=%d centroid=%f rough=%f", tempo_bpm, len(rms) if hasattr(rms, '__len__') else 0, float(centroid_mean), float(roughness))
    except Exception:
        logger.debug("compute_mood_features: input logging failed")

    # energy norm: mean RMS relative to peak signal
    mean_rms = float(np.mean(rms)) if (hasattr(rms, '__len__') and len(rms)) else 0.0
    peak = float(np.max(np.abs(y))) if (y is not None and len(y)) else 1.0
    energy_norm = float(np.clip(mean_rms / (peak + EPS), 0.0, 1.0))

    # tempo-based arousal
    tempo_norm = float(np.clip((tempo_bpm or 0.0) / 200.0, 0.0, 1.0))
    arousal = float(np.clip(0.5 * tempo_norm + 0.5 * energy_norm, 0.0, 1.0))

    # valence: combine brightness and major/minor key confidence
    key_info = harmony.get('key') if isinstance(harmony, dict) else None
    mode = None
    conf = 0.0
    if key_info and isinstance(key_info, dict):
        mode = key_info.get('mode')
        conf = float(key_info.get('confidence', 0.0))
    mode_factor = 0.0
    if mode == 'major':
        mode_factor = conf
    elif mode == 'minor':
        mode_factor = -conf
    brightness = float(timbre.get('brightness', 0.0) if isinstance(timbre, dict) else 0.0)
    valence = 0.5 + 0.25 * (brightness - 0.5) + 0.25 * mode_factor
    valence = float(np.clip(valence, 0.0, 1.0))

    # tension: use dissonance + roughness as proxies
    dissonance = consonance.get('dissonance', 0.5) if isinstance(consonance, dict) else 0.5
    tension = float(np.clip(0.6 * float(dissonance) + 0.4 * float(np.clip(roughness, 0.0, 1.0)), 0.0, 1.0))

    # beat strength average
    beat_strength_avg = float(np.mean(beat_strengths)) if (beat_strengths is not None and len(beat_strengths)) else 0.0
    danceability = float(np.clip(0.5 * tempo_norm + 0.4 * beat_strength_avg + 0.1 * energy_norm, 0.0, 1.0))

    # energy score (perceived intensity)
    energy_score = float(np.clip(0.6 * energy_norm + 0.4 * float(np.clip(roughness, 0.0, 1.0)), 0.0, 1.0))

    # heuristic mood tags
    tags = []
    if danceability > 0.6 and energy_score > 0.5:
        tags.append('danceable')
    if arousal < 0.35 and valence > 0.45 and (tempo_bpm or 0) < 100:
        tags.append('chill')
    if energy_score > 0.7 and roughness > 0.5:
        tags.append('aggressive')
    if valence > 0.6 and arousal > 0.5:
        tags.append('uplifting')
    if valence < 0.35 and arousal < 0.45:
        tags.append('melancholic')
    if len(tags) == 0:
        tags.append('neutral')

    mood = {
        'valence': float(valence),
        'arousal': float(arousal),
        'tension': float(tension),
        'danceability': float(danceability),
        'energy': float(energy_score),
        'tags': tags,
        'tempo_bpm': tempo_bpm,
    }
    logger.debug("compute_mood_features: mood=%s", mood)
    return mood

@timed()
def extract_features_from_array(y, sr, info=None, n_fft=2048, hop_length=512, save_spectrogram_path=None, use_gpu=False, use_pyfftw=False, rhythm_sr=None, rhythm_n_fft=1024, rhythm_hop_length=256, prefer_librosa=False, structure_window_sec=3.0, structure_hop_sec=1.0, structure_sim_threshold=0.75, structure_min_segment_sec=3.0):
    logger.debug("extract_features_from_array: sr=%s signal_length=%d n_fft=%d hop_length=%d use_gpu=%s use_pyfftw=%s", sr, len(y), n_fft, hop_length, use_gpu, use_pyfftw)
    duration = float(len(y) / sr)
    logger.debug("extract_features_from_array: duration=%f seconds", duration)
    rms, rms_times = rms_over_time(y, sr, frame_length=n_fft, hop_length=hop_length)
    f, t, S, S_db = compute_stft(y, sr, n_fft=n_fft, hop_length=hop_length, use_gpu=use_gpu, use_pyfftw=use_pyfftw)
    logger.debug("extract_features_from_array: STFT shapes f=%d t=%d S=%s", len(f), len(t), S.shape)
    avg_mags = np.mean(S, axis=1)
    peaks = top_spectrum_peaks(avg_mags, f, top_n=10)
    # Use vectorized autocorrelation-based F0 (allow pyFFTW acceleration)
    f0, f0_frames = autocorr_f0(y, sr, frame_length=n_fft, hop_length=hop_length, use_pyfftw=use_pyfftw)
    harmonics = harmonic_strengths(avg_mags, f, f0, n_harmonics=6)
    sf = spectral_flatness(S)
    percent_tonal = float(np.mean(sf < 0.5))
    logger.debug("extract_features_from_array: top_peaks_count=%d estimated_f0=%s percent_tonal=%f", len(peaks), f0, percent_tonal)

    # --- Pitch & Harmony features ---
    # Optionally use librosa implementations if requested via --use-librosa (default: False)
    librosa_mod = None
    try:
        if prefer_librosa:
            librosa_mod = _lazy_import_librosa()
            if librosa_mod is not None:
                try:
                    chroma = librosa_mod.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
                    logger.debug("extract_features_from_array: chroma computed with librosa chroma_cqt shape=%s", chroma.shape)
                except Exception:
                    logger.debug("extract_features_from_array: librosa chroma_cqt failed, falling back to chroma_from_S")
                    chroma = compute_chroma_from_S(S, f, sr=sr)
            else:
                logger.debug("extract_features_from_array: librosa requested but not available; using compute_chroma_from_S")
                chroma = compute_chroma_from_S(S, f, sr=sr)
        else:
            chroma = compute_chroma_from_S(S, f, sr=sr)
    except Exception:
        logger.exception("extract_features_from_array: failed to compute chroma; creating empty chroma")
        chroma = np.zeros((12, S.shape[1]), dtype=np.float32)

    # align chroma frame times to STFT times (truncate or pad as needed)
    try:
        chroma_frame_count = chroma.shape[1]
        if chroma_frame_count == len(t):
            chroma_times = t
        else:
            # if librosa computed chroma with a different frame hop, estimate times
            if prefer_librosa and (librosa_mod is not None):
                chroma_times = librosa_mod.frames_to_time(np.arange(chroma_frame_count), sr=sr, hop_length=hop_length)
            else:
                chroma_times = t[:chroma_frame_count].tolist()
        chroma_times = np.asarray(chroma_times, dtype=np.float32)
    except Exception:
        chroma_times = t

    key_info = detect_key_from_chroma(chroma)
    key_changes = key_over_time_from_chroma(chroma, chroma_times)
    chords = detect_chords_from_chroma(chroma, chroma_times)

    # Melody extraction: optionally prefer librosa.pyin (if requested), else fall back to autocorr frames
    if prefer_librosa and (librosa_mod is not None) and hasattr(librosa_mod, "pyin"):
        try:
            f0_lib = librosa_mod.pyin(y, fmin=50.0, fmax=2000.0, sr=sr, frame_length=n_fft, hop_length=hop_length)
            # pyin returns np.nan for unvoiced frames
            f0_values = [float(x) if not np.isnan(x) else 0.0 for x in f0_lib]
            logger.debug("extract_features_from_array: pyin extracted f0 frames=%d", len(f0_values))
        except Exception:
            logger.debug("extract_features_from_array: pyin failed; using autocorr f0 frames")
            f0_values = f0_frames
    else:
        f0_values = f0_frames

    melody = melody_contours_from_f0(f0_values, sr, frame_length=n_fft, hop_length=hop_length)
    pitch_dist = pitch_distribution_from_f0(melody.get("f0_values", []))
    consonance = consonance_from_chroma(chroma, chroma_times)

    # detect modulations (key changes)
    modulations = []
    prev = None
    for entry in key_changes:
        if prev is None:
            prev = entry.get("key")
            continue
        if entry.get("key") != prev:
            modulations.append({"time": entry.get("time"), "from": prev, "to": entry.get("key")})
            prev = entry.get("key")

    harmony = {
        "key": key_info,
        "key_over_time": key_changes,
        "chords": chords,
        "melody": melody,
        "pitch_distribution": pitch_dist,
        "consonance": consonance,
        "modulations": modulations,
    }
    logger.debug("extract_features_from_array: harmony keys=%s", list(harmony.keys()))

    # --- Rhythm & timing features ---
    # Optionally compute rhythm features on a downsampled/resampled copy
    if rhythm_sr and rhythm_sr < sr:
        logger.debug("extract_features_from_array: using downsampled rhythm_sr=%d", rhythm_sr)
        y_r = resample_audio(y, sr, rhythm_sr)
        f_r, t_r, S_r, S_db_r = compute_stft(y_r, rhythm_sr, n_fft=rhythm_n_fft, hop_length=rhythm_hop_length, use_gpu=use_gpu, use_pyfftw=use_pyfftw)
        onset_env = onset_envelope_from_S(S_r)
        onset_times = detect_onsets_from_env(onset_env, t_r, rhythm_sr, rhythm_hop_length)
        tempo_bpm, tempo_iois = estimate_tempo_from_onsets(onset_times, duration)
        tempo_changes = tempo_over_time(onset_times, duration)
        beats, beat_strengths = find_beats_from_onset_env(onset_env, t_r, duration, tempo_bpm, rhythm_sr, rhythm_hop_length)
    else:
        onset_env = onset_envelope_from_S(S)
        onset_times = detect_onsets_from_env(onset_env, t, sr, hop_length)
        tempo_bpm, tempo_iois = estimate_tempo_from_onsets(onset_times, duration)
        tempo_changes = tempo_over_time(onset_times, duration)
        beats, beat_strengths = find_beats_from_onset_env(onset_env, t, duration, tempo_bpm, sr, hop_length)
    time_sig = estimate_time_signature(beat_strengths)
    groove = estimate_groove_and_swing([b.get("time") for b in beats], onset_times)
    logger.debug("extract_features_from_array: rhythm onset_count=%d tempo=%s beats=%d time_sig=%s", len(onset_times), tempo_bpm, len(beats), time_sig)

    # --- Timbre & Sound Texture ---
    # MFCCs (prefer librosa when requested)
    try:
        if prefer_librosa and (librosa_mod is not None):
            try:
                mfcc_mat = librosa_mod.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
                mfcc_mean = np.mean(mfcc_mat, axis=1).astype(np.float32).tolist()
                mfcc_std = np.std(mfcc_mat, axis=1).astype(np.float32).tolist()
            except Exception:
                mfcc_mat, mfcc_mean, mfcc_std = compute_mfcc_from_S(S, f, sr, n_mfcc=13, n_mels=40)
        else:
            mfcc_mat, mfcc_mean, mfcc_std = compute_mfcc_from_S(S, f, sr, n_mfcc=13, n_mels=40)
    except Exception:
        logger.exception("extract_features_from_array: MFCC computation failed")
        mfcc_mean, mfcc_std = [], []

    centroid_list, rolloff_list = spectral_centroid_and_rolloff(S, f)
    centroid_mean = float(np.mean(centroid_list)) if len(centroid_list) else 0.0
    hf_ratio = high_frequency_energy_ratio(S, f, threshold_hz=min(6000.0, sr / 3.0))
    rough = roughness_score(S)
    sf_median = float(np.median(sf)) if hasattr(sf, 'shape') else float(sf)
    clip_prop = clipping_proportion(y)
    reverb = reverb_score_from_onsets(onset_times, rms, rms_times)
    autotune = autotune_score_from_f0(melody.get('f0_values', []))
    distortion = distortion_score(y, S, f, sf_median)

    # vocal / instrument heuristics
    voiced_ratio = float(np.mean(np.array(melody.get('f0_values', [])) > 0)) if len(melody.get('f0_values', [])) else 0.0
    instr = classify_instrument_simple(onset_times, melody.get('f0_values', []), sf_median, centroid_mean, hf_ratio, duration)

    timbre = {
        "mfcc": {"n_mfcc": 13, "mean": mfcc_mean, "std": mfcc_std},
        "spectral_centroid_mean_hz": centroid_mean,
        "spectral_rolloff_median_hz": float(np.median(rolloff_list)) if len(rolloff_list) else 0.0,
        "brightness": float(np.clip(centroid_mean / (sr / 2.0), 0.0, 1.0)),
        "warmth": float(np.clip(1.0 - (centroid_mean / (sr / 2.0)), 0.0, 1.0)),
        "roughness": rough,
        "high_freq_energy_ratio": hf_ratio,
        "clipping_proportion": clip_prop,
        "production": {"reverb_score": reverb, "distortion_score": distortion, "autotune_score": autotune},
        "vocal": {"voiced_ratio": voiced_ratio, "is_vocal": bool(voiced_ratio > 0.15)},
        "instrument": instr,
    }

    # --- Structure & Form features ---
    try:
        structure = compute_structure_features(chroma, chroma_times, y, sr, onset_times, rms, rms_times, window_sec=structure_window_sec, hop_sec=structure_hop_sec, sim_threshold=structure_sim_threshold, min_seg_sec=structure_min_segment_sec)
    except Exception:
        logger.exception("extract_features_from_array: compute_structure_features failed")
        structure = {"segments": [], "repetition_groups": [], "transitions": []}

    if save_spectrogram_path:
        save_spectrogram(f, t, S_db, save_spectrogram_path)

    # --- Mood / Feel estimation ---
    try:
        mood = compute_mood_features(y, sr, tempo_bpm, rms, centroid_mean, rough, consonance, [b.get('time') for b in beats], beat_strengths, timbre, harmony)
    except Exception:
        logger.exception("extract_features_from_array: compute_mood_features failed")
        mood = {}

    features = {
        "sample_rate": int(sr),
        "duration": duration,
        "amplitude": {"rms": rms.tolist(), "times": rms_times.tolist()},
        "spectrum": {"top_peaks": peaks},
        "fundamental_frequency_hz": f0,
        "f0_over_time": f0_frames,
        "harmonics": harmonics,
        "noise_vs_tonal": {"spectral_flatness_median": float(np.median(sf)), "percent_tonal": percent_tonal},
        "spectrogram": {"freq_bins": int(len(f)), "time_bins": int(len(t)), "image": save_spectrogram_path if save_spectrogram_path else None},
        "rhythm": {
            "onsets": onset_times,
            "tempo_bpm": tempo_bpm,
            "tempo_iois": tempo_iois,
            "tempo_over_time": tempo_changes,
            "beats": beats,
            "time_signature": time_sig,
            "groove": groove,
        },
        "harmony": harmony,
        "timbre": timbre,
        "structure": structure,
        "feel": mood,
    }
    if info is not None:
        bd = bit_depth_from_subtype(getattr(info, "subtype", None))
        features.update({"bit_depth": bd, "channels": int(getattr(info, "channels", 1))})
    logger.debug("extract_features_from_array: finished, feature_keys=%s", list(features.keys()))
    return features


def extract_features_from_file(path, out_json=None, spectrogram_path=None, use_gpu=False, use_pyfftw=False, rhythm_sr=None, rhythm_n_fft=1024, rhythm_hop_length=256, n_fft=None, hop_length=None, fast=False, prefer_librosa=False, structure_window_sec=3.0, structure_hop_sec=1.0, structure_sim_threshold=0.75, structure_min_segment_sec=3.0, lyrics=False, asr_backend="whisper", nlp_backend="transformers", lyrics_out=None):
    logger.debug("extract_features_from_file: path=%s out_json=%s spectrogram_path=%s use_gpu=%s use_pyfftw=%s", path, out_json, spectrogram_path, use_gpu, use_pyfftw)
    y, sr, info = read_audio(path)
    logger.debug("extract_features_from_file: read audio sr=%s length=%d channels=%s", sr, len(y), getattr(info, "channels", None))
    # determine effective STFT parameters
    effective_n_fft = n_fft if n_fft is not None else 2048
    effective_hop = hop_length if hop_length is not None else 512
    if fast:
        # faster heuristics: smaller FFT and hop, prefer pyFFTW when available
        effective_n_fft = min(effective_n_fft, 1024)
        effective_hop = min(effective_hop, 256)
        if not use_pyfftw and (_lazy_import_pyfftw() is not None):
            use_pyfftw = True

    # Optional lyrics / language processing (ASR + NLP). This is lazy and optional.
    lyrics_data = None
    if lyrics:
        lyrics_mod = _lazy_import_lyrics()
        if lyrics_mod is None:
            logger.warning("extract_features_from_file: --lyrics requested but lyrics module not available; skipping lyrics processing")
            # If user asked for an output path, write a placeholder explaining why no transcript is available.
            if lyrics_out:
                try:
                    dirpath = os.path.dirname(lyrics_out)
                    if dirpath:
                        os.makedirs(dirpath, exist_ok=True)
                    with open(lyrics_out, "w", encoding="utf-8") as _fh:
                        _fh.write("Transcription skipped: lyrics module not available.\n")
                    logger.debug("extract_features_from_file: wrote placeholder transcript to %s", lyrics_out)
                except Exception:
                    logger.exception("extract_features_from_file: failed to write placeholder lyrics_out=%s", lyrics_out)
        else:
            try:
                logger.info("extract_features_from_file: running ASR (backend=%s)", asr_backend)
                transcript, asr_meta = lyrics_mod.transcribe_audio(path, backend=asr_backend, language="en")
                logger.debug("extract_features_from_file: transcript_len=%d", len(transcript) if transcript is not None else 0)
                analysis = lyrics_mod.analyze_lyrics(transcript, backend=nlp_backend, language="en")
                lyrics_data = {"transcript": transcript, "asr_meta": asr_meta, "analysis": analysis}
            except Exception as e:
                # Log a concise warning and write an explanatory file if requested
                logger.warning("extract_features_from_file: lyrics processing failed (%s); continuing without lyrics", str(e))
                if lyrics_out:
                    try:
                        dirpath = os.path.dirname(lyrics_out)
                        if dirpath:
                            os.makedirs(dirpath, exist_ok=True)
                        with open(lyrics_out, "w", encoding="utf-8") as _fh:
                            _fh.write(f"Transcription failed: {str(e)}\n")
                        logger.debug("extract_features_from_file: wrote failure message to %s", lyrics_out)
                    except Exception:
                        logger.exception("extract_features_from_file: failed to write lyrics_out=%s", lyrics_out)
            else:
                # On success, write the transcript to disk if an output path was provided.
                if lyrics_out is not None:
                    try:
                        dirpath = os.path.dirname(lyrics_out)
                        if dirpath:
                            os.makedirs(dirpath, exist_ok=True)
                        with open(lyrics_out, "w", encoding="utf-8") as _fh:
                            _fh.write(transcript if transcript is not None else "")
                        logger.debug("extract_features_from_file: wrote transcript to %s", lyrics_out)
                    except Exception:
                        logger.exception("extract_features_from_file: failed to write lyrics_out=%s", lyrics_out)

    features = extract_features_from_array(
        y,
        sr,
        info=info,
        n_fft=effective_n_fft,
        hop_length=effective_hop,
        save_spectrogram_path=spectrogram_path,
        use_gpu=use_gpu,
        use_pyfftw=use_pyfftw,
        rhythm_sr=rhythm_sr,
        rhythm_n_fft=rhythm_n_fft,
        rhythm_hop_length=rhythm_hop_length,
        prefer_librosa=prefer_librosa,
        structure_window_sec=structure_window_sec,
        structure_hop_sec=structure_hop_sec,
        structure_sim_threshold=structure_sim_threshold,
        structure_min_segment_sec=structure_min_segment_sec,
    )
    # attach lyrics data if available
    try:
        if lyrics_data is not None:
            features["lyrics"] = lyrics_data
    except Exception:
        logger.exception("extract_features_from_file: failed to attach lyrics data to features")
    if out_json:
        logger.debug("extract_features_from_file: writing features JSON to %s", out_json)
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump(features, fh, indent=2)
        logger.debug("extract_features_from_file: wrote JSON")
    return features


def demo():
    logger.info("demo: generating test sine tone")
    sr = 44100
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    y = 0.6 * np.sin(2 * math.pi * 440.0 * t)
    tmp = os.path.join(tempfile.gettempdir(), "sine_spectrogram.png")
    features = extract_features_from_array(y.astype(np.float32), sr, info=None, save_spectrogram_path=tmp)
    print(json.dumps(features, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Extract raw audio and signal features from a waveform")
    p.add_argument("--input", "-i", help="Input audio file (wav, flac, etc)")
    p.add_argument("--out-json", help="Write features JSON to this path")
    p.add_argument("--spectrogram", help="Save spectrogram image to this path")
    p.add_argument("--demo", action="store_true", help="Run an internal demo (sine) and print features")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    p.add_argument("--use-gpu", action="store_true", help="Use GPU (PyTorch) for STFT when available")
    p.add_argument("--use-pyfftw", action="store_true", help="Use pyFFTW for faster CPU FFTs when available")
    p.add_argument("--benchmark", action="store_true", help="Enable lightweight timing/benchmark mode and print function timings")
    p.add_argument("--use-librosa", action="store_true", help="Prefer librosa implementations for chroma/pyin (higher-quality but slower)")
    p.add_argument("--nfft", type=int, default=None, help="STFT n_fft (default 2048)")
    p.add_argument("--hop-length", type=int, default=None, help="STFT hop_length (default 512)")
    p.add_argument("--fast", action="store_true", help="Enable faster heuristics (smaller n_fft/hop, prefer pyFFTW if available)")
    p.add_argument("--rhythm-sr", type=int, default=None, help="Optional sample rate to resample audio for rhythm/onset detection (e.g., 22050)")
    p.add_argument("--rhythm-nfft", type=int, default=1024, help="FFT size to use for rhythm STFT when downsampling")
    p.add_argument("--rhythm-hop", type=int, default=256, help="Hop length to use for rhythm STFT when downsampling")
    p.add_argument("--lyrics", action="store_true", help="Run ASR and lyrics NLP analysis (English)")
    p.add_argument("--asr-backend", type=str, default="whisper", choices=["whisper", "auto"], help="ASR backend to use for transcription (default: whisper)")
    p.add_argument("--nlp-backend", type=str, default="transformers", choices=["transformers", "simple"], help="NLP backend for sentiment/NER (default: transformers)")
    p.add_argument("--lyrics-out", type=str, default=None, help="Optional path to write the transcript text")
    p.add_argument("--structure-window", type=float, default=3.0, help="Window length (s) for structure detection (default 3.0)")
    p.add_argument("--structure-hop", type=float, default=1.0, help="Hop length (s) for structure detection (default 1.0)")
    p.add_argument("--structure-sim-threshold", type=float, default=0.75, help="Similarity threshold for grouping repeated sections")
    p.add_argument("--structure-min-seg", type=float, default=3.0, help="Minimum segment length in seconds for structure detection")
    return p.parse_args()
def main(input_path, out_json=None, spectrogram_path=None, use_gpu=False, use_pyfftw=False, prefer_librosa=False):
    """Main program entry.

    Caller must provide `input_path` (no hardcoded defaults are set inside this module).
    Returns the extracted features dictionary.
    """
    if not input_path:
        raise ValueError("input_path is required by main(); callers must provide an audio file path.")
    path = input_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    if spectrogram_path is None:
        spectrogram_path = os.path.join(tempfile.gettempdir(), "spectrogram.png")
    logger.debug("main: input=%s spectrogram_path=%s out_json=%s", input_path, spectrogram_path, out_json)
    features = extract_features_from_file(path, out_json=out_json, spectrogram_path=spectrogram_path, use_gpu=use_gpu, use_pyfftw=use_pyfftw, prefer_librosa=prefer_librosa)
    return features


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.debug("__main__: args=%s", vars(args))
    # enable benchmarking timings if requested
    if getattr(args, "benchmark", False):
        BENCHMARK_ENABLED = True
    if args.demo:
        demo()
    else:
        if not args.input:
            print("Error: --input is required unless --demo is used.")
            raise SystemExit(2)
        try:
            features = extract_features_from_file(
                args.input,
                out_json=args.out_json,
                spectrogram_path=args.spectrogram,
                use_gpu=args.use_gpu,
                use_pyfftw=args.use_pyfftw,
                rhythm_sr=args.rhythm_sr,
                rhythm_n_fft=args.rhythm_nfft,
                rhythm_hop_length=args.rhythm_hop,
                prefer_librosa=getattr(args, "use_librosa", False),
                n_fft=args.nfft,
                hop_length=args.hop_length,
                fast=getattr(args, "fast", False),
                    lyrics=getattr(args, "lyrics", False),
                    asr_backend=getattr(args, "asr_backend", "whisper"),
                    nlp_backend=getattr(args, "nlp_backend", "transformers"),
                    lyrics_out=getattr(args, "lyrics_out", None),
                structure_window_sec=getattr(args, "structure_window", 3.0),
                structure_hop_sec=getattr(args, "structure_hop", 1.0),
                structure_sim_threshold=getattr(args, "structure_sim_threshold", 0.75),
                structure_min_segment_sec=getattr(args, "structure_min_seg", 3.0),
            )
            print(json.dumps(features, indent=2))
            if BENCHMARK_ENABLED:
                print("\nTiming summary:")
                for k, v in sorted(TIMINGS.items(), key=lambda x: -x[1]):
                    print(f"{k}: {v:.4f}s")
        except (FileNotFoundError, ValueError) as e:
            logger.error("main: error occurred: %s", str(e))
            print(str(e))
