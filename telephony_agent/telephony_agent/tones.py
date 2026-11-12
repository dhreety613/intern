"""8 kHz mu-law codec + in-band DTMF synth. (Tone/beep detection lives in detection/audio.py.)"""
from __future__ import annotations

import numpy as np

SR = 8000
_BIAS = 0x84
_CLIP = 32635


def mulaw_decode(data: bytes) -> np.ndarray:
    u = (~np.frombuffer(data, dtype=np.uint8)).astype(np.int32) & 0xFF
    sign = u & 0x80
    exp = (u >> 4) & 0x07
    man = u & 0x0F
    s = (((man << 3) + _BIAS) << exp) - _BIAS
    return np.where(sign != 0, -s, s).astype(np.int16)


def mulaw_encode(pcm: np.ndarray) -> bytes:
    x = pcm.astype(np.int32)
    sign = np.where(x < 0, 0x80, 0)
    x = np.minimum(np.abs(x), _CLIP) + _BIAS
    exp = np.floor(np.log2(x)).astype(np.int32) - 7
    man = (x >> (exp + 3)) & 0x0F
    return ((~(sign | (exp << 4) | man)) & 0xFF).astype(np.uint8).tobytes()


_DTMF = {
    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477), "A": (697, 1633),
    "4": (770, 1209), "5": (770, 1336), "6": (770, 1477), "B": (770, 1633),
    "7": (852, 1209), "8": (852, 1336), "9": (852, 1477), "C": (852, 1633),
    "*": (941, 1209), "0": (941, 1336), "#": (941, 1477), "D": (941, 1633),
}


def dtmf_mulaw(digits: str, tone_ms=250, gap_ms=120, amp=8000) -> bytes:   # amp per tone, as call_agent/app/dtmf.py
    """In-band DTMF as mu-law. 'w' = 500 ms pause. Sent over the media stream so the WS stays up."""
    out = []
    for d in digits.upper():
        if d == "W":
            out.append(np.zeros(int(SR * 0.5), dtype=np.int16))
            continue
        if d not in _DTMF:
            continue
        lo, hi = _DTMF[d]
        t = np.arange(int(SR * tone_ms / 1000)) / SR
        tone = amp * (np.sin(2 * np.pi * lo * t) + np.sin(2 * np.pi * hi * t))
        out.append(tone.astype(np.int16))
        out.append(np.zeros(int(SR * gap_ms / 1000), dtype=np.int16))
    return mulaw_encode(np.concatenate(out)) if out else b""
