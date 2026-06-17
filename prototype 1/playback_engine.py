"""Lightweight Playback Engine (skeleton)

Responsibilities:
 - prebuffer small segments of tracks using `soundfile` for low-latency handoff
 - provide crossfade blending buffers for transition application
 - simple effect helpers (echo)

This module is intentionally conservative: it does not attempt to be a full DJ engine,
but provides primitives that can be called from a real-time playback loop (Node, Rust, etc.).
"""
from typing import Optional, Tuple
import numpy as np
import soundfile as sf


class PlaybackEngine:
    def __init__(self, crossfade_duration: float = 6.0, sample_rate: int = 44100):
        self.crossfade_duration = float(crossfade_duration)
        self.sample_rate = int(sample_rate)

    def prebuffer(self, path: str, seconds: float = 6.0) -> Optional[np.ndarray]:
        """Load up to `seconds` of audio from `path` into a numpy buffer.

        Returns mono float32 numpy array or None on failure.
        """
        try:
            max_frames = int(self.sample_rate * float(seconds))
            data, sr = sf.read(path, dtype='float32', always_2d=True, stop=max_frames)
            if data is None:
                return None
            # mix to mono
            if data.shape[1] > 1:
                data = np.mean(data, axis=1)
            else:
                data = data.reshape(-1)
            # resample if needed (naive approach)
            if sr != self.sample_rate:
                # use simple linear resample (not high-quality) to avoid extra deps
                import numpy as _np
                ratio = float(self.sample_rate) / float(sr)
                idx = _np.linspace(0, len(data) - 1, int(len(data) * ratio))
                data = _np.interp(idx, _np.arange(len(data)), data).astype('float32')
            return data
        except Exception:
            return None

    def crossfade_buffers(self, a: np.ndarray, b: np.ndarray, duration: Optional[float] = None) -> np.ndarray:
        """Return a crossfaded buffer combining end of `a` with start of `b` over `duration` seconds.

        Both `a` and `b` are numpy mono float arrays. If either buffer shorter than needed,
        they will be zero-padded.
        """
        dur = float(duration or self.crossfade_duration)
        n = int(dur * self.sample_rate)
        a_tail = a[-n:] if a.shape[0] >= n else np.pad(a, (max(0, n - a.shape[0]), 0))[-n:]
        b_head = b[:n] if b.shape[0] >= n else np.pad(b, (0, max(0, n - b.shape[0])))[:n]
        # linear crossfade
        t = np.linspace(0.0, 1.0, n, endpoint=False).astype('float32')
        fade_out = (1.0 - t) * a_tail
        fade_in = t * b_head
        mix = fade_out + fade_in
        # return: prefix_a + mix + suffix_b
        prefix = a[:-n] if a.shape[0] > n else np.array([], dtype='float32')
        suffix = b[n:] if b.shape[0] > n else np.array([], dtype='float32')
        return np.concatenate([prefix, mix, suffix])

    def crossfade(self, current_path: str, next_path: str, duration: Optional[float] = None) -> Optional[np.ndarray]:
        """Convenience wrapper: prebuffer last/first segments and crossfade them.

        Returns combined buffer or None.
        """
        a = self.prebuffer(current_path, seconds=(duration or self.crossfade_duration) + 1.0)
        b = self.prebuffer(next_path, seconds=(duration or self.crossfade_duration) + 1.0)
        if a is None or b is None:
            return None
        return self.crossfade_buffers(a, b, duration=duration)

    def apply_echo(self, buffer: np.ndarray, delay_s: float = 0.25, decay: float = 0.4) -> np.ndarray:
        """Apply a simple echo effect (single tap) to a buffer."""
        try:
            delay_samples = int(delay_s * self.sample_rate)
            out = np.copy(buffer)
            if delay_samples <= 0:
                return out
            padded = np.pad(buffer, (delay_samples, 0))[:buffer.shape[0]]
            out += decay * padded
            # clip
            out = np.clip(out, -1.0, 1.0)
            return out.astype('float32')
        except Exception:
            return buffer


if __name__ == '__main__':
    # quick smoke test using test_wavs if present
    import os
    print('PlaybackEngine smoke test')
    try:
        pe = PlaybackEngine()
        a = pe.prebuffer('test_wavs/track_build.wav', seconds=5.0)
        b = pe.prebuffer('test_wavs/track_drop.wav', seconds=5.0)
        if a is not None and b is not None:
            out = pe.crossfade_buffers(a, b, duration=3.0)
            print('Crossfade len', len(out))
        else:
            print('Prebuffer failed')
    except Exception as e:
        print('PlaybackEngine error', e)
