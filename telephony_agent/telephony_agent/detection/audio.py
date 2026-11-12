"""Audio-event detectors (non-speech): tones (beep / busy / fax) and hold music.

ToneDetector  - FFT on 40 ms windows. BEEP_END fires when a voicemail/screening beep ENDS.
                Also busy / fast-busy cadence (480+620 Hz) and fax CNG (1100 Hz) / CED (2100 Hz).
MusicDetector - loud AND Silero says "not speech" for >= 2 s  => hold music (or ringback after a transfer).
Both are DSP/ONNX heuristics tuned on synthetic audio only; tune on real call recordings.
A model backend (e.g. YAMNet) can replace either by emitting the same Signals.
"""
from __future__ import annotations

import numpy as np

from ..tones import mulaw_decode
from .signals import Sig, Signal
from .vad import CHUNK, CHUNK_S, SR, make_backend, silero_available

WIN, HOP = 320, 160            # 40 ms window, 20 ms hop
HOP_S = HOP / SR


class ToneDetector:
    source = "tone"

    def __init__(self, t0: float = 0.0, min_beep_s=0.12, max_beep_s=1.5, lo=350, hi=2300, purity=0.80, min_rms=300.0):
        self.t0, self.n = t0, 0
        self.min_beep_s, self.max_beep_s, self.lo, self.hi = min_beep_s, max_beep_s, lo, hi
        self.purity, self.min_rms = purity, min_rms
        self.buf = np.zeros(0, np.float32)
        self.win = np.hanning(WIN).astype(np.float32)
        self.freqs = np.fft.rfftfreq(WIN, 1 / SR)
        # single-tone run
        self.run = 0
        self.run_f: list[float] = []
        self.sent_ced = False
        self.cng_times: list[float] = []
        # busy cadence
        self.pair_run = 0
        self.pair_on: list[tuple[float, float]] = []      # (start_t, dur)
        self.pair_gap_t: float | None = None
        self.busy_sent = False

    def _band(self, spec, f, w=40):
        return float(spec[(self.freqs >= f - w) & (self.freqs <= f + w)].sum())

    def _classify(self, frame):
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms < self.min_rms:
            return None, 0.0
        spec = np.abs(np.fft.rfft(frame * self.win)) ** 2
        total = float(spec.sum())
        if total <= 0:
            return None, 0.0
        k = int(np.argmax(spec))
        f = float(self.freqs[k])
        if self.lo <= f <= self.hi and float(spec[max(0, k - 3): k + 4].sum()) / total >= self.purity:
            return "single", f
        b4, b6 = self._band(spec, 480) / total, self._band(spec, 620) / total
        if b4 > 0.2 and b6 > 0.2 and b4 + b6 > 0.7:
            return "pair", 0.0
        return None, 0.0

    def feed(self, mulaw: bytes) -> list[Signal]:
        self.buf = np.concatenate([self.buf, mulaw_decode(mulaw).astype(np.float32)])
        out: list[Signal] = []
        while len(self.buf) >= WIN:
            frame, self.buf = self.buf[:WIN], self.buf[HOP:]
            self.n += 1
            t = self.t0 + self.n * HOP_S
            kind, f = self._classify(frame)
            out += self._single(kind == "single", f, t)
            out += self._pair(kind == "pair", t)
        return out

    def _single(self, on: bool, f: float, t: float) -> list[Signal]:
        out: list[Signal] = []
        if on:
            self.run += 1
            self.run_f.append(f)
            dur = self.run * HOP_S
            mf = float(np.median(self.run_f))
            if abs(mf - 2100) < 70 and dur >= 1.5 and not self.sent_ced:
                self.sent_ced = True
                out.append(Signal(self.source, Sig.TONE_FAX, t, 0.8, {"tone": "CED"}))
            elif dur >= self.max_beep_s and abs(mf - 2100) >= 70:
                out.append(Signal(self.source, Sig.BEEP_END, t, 0.6, {"freq": mf, "dur": dur, "capped": True}))
                self.run, self.run_f = 0, []
        else:
            if self.run:
                dur, mf = self.run * HOP_S, float(np.median(self.run_f))
                if abs(mf - 1100) < 60 and 0.35 <= dur <= 0.8:            # fax CNG: 0.5 s every 3 s
                    self.cng_times = [x for x in self.cng_times if t - x < 5.0] + [t]
                    out.append(Signal(self.source, Sig.TONE_FAX, t, 0.9 if len(self.cng_times) >= 2 else 0.5,
                                      {"tone": "CNG"}))
                elif dur >= self.min_beep_s and dur < self.max_beep_s:
                    out.append(Signal(self.source, Sig.BEEP_END, t, 0.85, {"freq": mf, "dur": dur}))
            self.run, self.run_f, self.sent_ced = 0, [], False
        return out

    def _pair(self, on: bool, t: float) -> list[Signal]:
        out: list[Signal] = []
        if on:
            self.pair_run += 1
        elif self.pair_run:
            dur = self.pair_run * HOP_S
            self.pair_run = 0
            if 0.12 <= dur <= 0.75:
                self.pair_on = [(s, d) for s, d in self.pair_on if t - s < 6.0] + [(t - dur, dur)]
            if len(self.pair_on) >= 3 and not self.busy_sent:
                ons = [d for _, d in self.pair_on[-3:]]
                gaps = [self.pair_on[i][0] - (self.pair_on[i - 1][0] + self.pair_on[i - 1][1])
                        for i in range(len(self.pair_on) - 2, len(self.pair_on))]
                if all(0.12 <= g <= 0.8 for g in gaps):
                    self.busy_sent = True
                    cad = "busy" if np.mean(ons) >= 0.38 else "reorder"
                    out.append(Signal(self.source, Sig.TONE_BUSY, t, 0.9, {"cadence": cad}))
        return out


class MusicDetector:
    source = "music"

    def __init__(self, t0: float = 0.0, on_windows=4, off_windows=2, min_rms=150.0, max_speech_prob=0.2):
        self.t0, self.n = t0, 0
        self.on_windows, self.off_windows = on_windows, off_windows
        self.min_rms, self.max_prob = min_rms, max_speech_prob
        self.backend = make_backend()
        self.weak = not silero_available()        # energy-only fallback can't separate music from speech
        self.buf = np.zeros(0, np.float32)
        self.probs: list[float] = []
        self.rms: list[float] = []
        self.on = False
        self._yes = self._no = 0
        self.win_chunks = int(0.5 / CHUNK_S)

    def feed(self, mulaw: bytes) -> list[Signal]:
        if self.weak:
            return []                              # never guess hold music without a real speech model
        pcm = mulaw_decode(mulaw)
        self.buf = np.concatenate([self.buf, pcm.astype(np.float32)])
        self.probs += self.backend.process(pcm)
        out: list[Signal] = []
        while len(self.probs) >= self.win_chunks:
            probs, self.probs = self.probs[: self.win_chunks], self.probs[self.win_chunks:]
            seg, self.buf = self.buf[: self.win_chunks * CHUNK], self.buf[self.win_chunks * CHUNK:]
            self.n += 1
            t = self.t0 + self.n * self.win_chunks * CHUNK_S
            rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
            music_like = rms >= self.min_rms and float(np.mean(probs)) < self.max_prob
            if music_like:
                self._yes, self._no = self._yes + 1, 0
                if not self.on and self._yes >= self.on_windows:
                    self.on = True
                    out.append(Signal(self.source, Sig.MUSIC_ON, t, 0.8))
            else:
                self._no, self._yes = self._no + 1, 0
                if self.on and self._no >= self.off_windows:
                    self.on = False
                    out.append(Signal(self.source, Sig.MUSIC_OFF, t, 0.8))
        return out
