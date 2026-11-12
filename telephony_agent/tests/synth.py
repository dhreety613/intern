"""Test-audio generator: espeak-ng speech + synthetic beeps/music/tones, all as 8 kHz mu-law."""
from __future__ import annotations

import ctypes
import functools

import numpy as np

from telephony_agent.tones import mulaw_encode

SR = 8000


@functools.lru_cache(maxsize=None)
def _lib():
    import espeakng_loader
    lib = ctypes.CDLL(espeakng_loader.get_library_path())
    lib.espeak_Initialize.restype = ctypes.c_int
    rate = lib.espeak_Initialize(2, 0, espeakng_loader.get_data_path().encode(), 0)   # SYNCHRONOUS
    return lib, rate


def speech(text: str, voice: str = "en-us", speed: int = 150) -> np.ndarray:
    lib, rate = _lib()
    chunks: list[np.ndarray] = []
    CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)

    def cb(wav, n, ev):
        if wav and n > 0:
            chunks.append(np.ctypeslib.as_array(wav, shape=(n,)).copy())
        return 0

    cbf = CB(cb)
    lib.espeak_SetSynthCallback(cbf)
    lib.espeak_SetVoiceByName(voice.encode())
    lib.espeak_SetParameter(1, speed, 0)       # espeakRATE
    b = text.encode() + b"\0"
    lib.espeak_Synth(b, len(b), 0, 0, 0, 1, None, None)
    lib.espeak_Synchronize()
    pcm = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(1, np.float32)
    n_out = int(len(pcm) * SR / rate)                       # crude resample (speech band << 4 kHz)
    k = max(1, int(rate / SR))
    pcm = np.convolve(pcm, np.ones(k) / k, mode="same")
    out = np.interp(np.linspace(0, len(pcm) - 1, n_out), np.arange(len(pcm)), pcm)
    return (out * (9000 / max(1.0, np.abs(out).max()))).astype(np.int16)


def silence(sec: float) -> np.ndarray:
    return np.zeros(int(SR * sec), np.int16)


def noise(sec: float, amp: float = 30, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).normal(0, amp, int(SR * sec))).astype(np.int16)


def tone(freq: float, sec: float, amp: float = 8000) -> np.ndarray:
    t = np.arange(int(SR * sec)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def dual_tone(f1: float, f2: float, sec: float, amp: float = 6000) -> np.ndarray:
    t = np.arange(int(SR * sec)) / SR
    return (amp * (np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)) / 2).astype(np.int16)


def music(sec: float, seed: int = 1) -> np.ndarray:
    """Hold-music-like: sustained chords with slow note changes, vibrato, light noise. No speech pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * sec)) / SR
    out = np.zeros_like(t)
    notes = [261.6, 329.6, 392.0, 440.0, 523.3, 392.0, 329.6, 293.7]
    seg = 0.5
    for i in range(int(sec / seg) + 1):
        f0 = notes[i % len(notes)] * rng.choice([1, 1, 2])
        m = (t >= i * seg) & (t < (i + 1) * seg + 0.1)
        tt = t[m] - i * seg
        env = np.minimum(1, tt * 20) * np.exp(-tt * 1.2)
        out[m] += env * sum(np.sin(2 * np.pi * f0 * h * tt + 0.3 * np.sin(2 * np.pi * 5 * tt)) / h for h in (1, 2, 3, 4))
    out += rng.normal(0, 0.02, len(t))
    return (6000 * out / np.abs(out).max()).astype(np.int16)


def cat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts)


def to_mulaw(pcm: np.ndarray) -> bytes:
    return mulaw_encode(pcm)


def frames(pcm: np.ndarray, n: int = 160):
    b = mulaw_encode(pcm)
    for i in range(0, len(b), n):
        yield b[i:i + n]
