#!/usr/bin/env python3
"""Small visualization helpers for testing the extractor.

Provides:
- plot_waveform(y, sr, out_path)
- plot_spectrogram(y, sr, out_path)
- plot_spectrum(y, sr, out_path)
- test_visualizer(path) -> dict of generated image paths
"""
import os
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt

EPS = 1e-10
import logging

logger = logging.getLogger(__name__)


def plot_waveform(y, sr, out_path):
    logger.debug("plot_waveform: sr=%s len=%d out_path=%s", sr, len(y), out_path)
    times = np.arange(len(y)) / float(sr)
    plt.figure(figsize=(10, 3))
    plt.plot(times, y, linewidth=0.5)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.debug("plot_waveform: saved %s", out_path)
    return out_path


def plot_spectrogram(y, sr, out_path, n_fft=2048, hop_length=512):
    logger.debug("plot_spectrogram: n_fft=%d hop_length=%d signal_length=%d out_path=%s", n_fft, hop_length, len(y), out_path)
    f, t, Zxx = signal.stft(y, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None, padded=False)
    S = np.abs(Zxx)
    S_db = 20.0 * np.log10(S + EPS)
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(t, f, S_db, shading="gouraud")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.debug("plot_spectrogram: saved %s", out_path)
    return out_path


def plot_spectrum(y, sr, out_path, n_fft=4096):
    logger.debug("plot_spectrum: n_fft=%d len(y)=%d out_path=%s", n_fft, len(y), out_path)
    N = int(n_fft)
    if len(y) < N:
        y2 = np.pad(y, (0, N - len(y)))
    else:
        y2 = y[:N]
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    mags = np.abs(np.fft.rfft(y2)[: len(freqs)])
    plt.figure(figsize=(8, 4))
    plt.semilogx(freqs[1:], 20 * np.log10(mags[1:] + EPS))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude (dB)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.debug("plot_spectrum: saved %s", out_path)
    return out_path


def test_visualizer(path):
    logger.debug("test_visualizer: path=%s", path)
    y, sr = sf.read(path, dtype="float32")
    logger.debug("test_visualizer: raw shape=%s sr=%s", getattr(y, "shape", None), sr)
    if y.ndim > 1:
        logger.debug("test_visualizer: converting to mono by averaging channels")
        y = np.mean(y, axis=1)
    out_dir = tempfile.gettempdir()
    wf = os.path.join(out_dir, "visualizer_waveform.png")
    spec = os.path.join(out_dir, "visualizer_spectrogram.png")
    spc = os.path.join(out_dir, "visualizer_spectrum.png")
    plot_waveform(y, sr, wf)
    plot_spectrogram(y, sr, spec)
    plot_spectrum(y, sr, spc)
    logger.debug("test_visualizer: generated files: %s %s %s", wf, spec, spc)
    return {"waveform": wf, "spectrogram": spec, "spectrum": spc}
