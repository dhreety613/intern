"""DTMF tone generation as 8kHz mu-law, ready to send over a Twilio Media Stream."""

import math

_ROWS = {"1": 697, "2": 697, "3": 697, "4": 770, "5": 770, "6": 770,
         "7": 852, "8": 852, "9": 852, "*": 941, "0": 941, "#": 941}
_COLS = {"1": 1209, "2": 1336, "3": 1477, "4": 1209, "5": 1336, "6": 1477,
         "7": 1209, "8": 1336, "9": 1477, "*": 1209, "0": 1336, "#": 1477}

SAMPLE_RATE = 8000
_BIAS = 0x84
_CLIP = 32635


def _lin2ulaw(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0
    sample = min(abs(sample), _CLIP) + _BIAS
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (sample & mask):
        exponent -= 1
        mask >>= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


def dtmf_ulaw(digit: str, tone_ms: int = 250, gap_ms: int = 120) -> bytes:
    """mu-law audio for one key press: dual tone, then silence."""
    if digit not in _ROWS:
        raise ValueError(f"Not a DTMF key: {digit!r}")

    low, high = _ROWS[digit], _COLS[digit]
    n_tone = SAMPLE_RATE * tone_ms // 1000
    amplitude = 8000  # two tones sum to ~16000, well inside 16-bit range

    out = bytearray(
        _lin2ulaw(int(amplitude * (
            math.sin(2 * math.pi * low * i / SAMPLE_RATE)
            + math.sin(2 * math.pi * high * i / SAMPLE_RATE)
        )))
        for i in range(n_tone)
    )
    out.extend(b"\xff" * (SAMPLE_RATE * gap_ms // 1000))  # mu-law silence
    return bytes(out)
