"""Voice activity + rhythm detector. Silero VAD (ONNX, no torch) by default, energy VAD fallback.

Emits SPEECH_START / SPEECH_END and RHYTHM:
  machine : one continuous run of speech >= 4 s (0.65) / >= 7 s (0.9)  -> greeting / recording
  human   : short burst (<= 2.5 s) then quiet (>= 1.0 s)              -> "Hello?" style answer
RHYTHM is supporting evidence only; fusion never confirms a human on rhythm alone.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Protocol

import numpy as np

from ..tones import mulaw_decode
from .signals import Sig, Signal

SR = 8000
CHUNK = 256                     # Silero v5 window at 8 kHz = 32 ms
CHUNK_S = CHUNK / SR


class VadBackend(Protocol):
    def process(self, pcm: np.ndarray) -> list[float]:
        """int16 samples in, one speech probability per 32 ms chunk out (stateful)."""


_SESSION = None


def silero_model_path() -> str | None:
    bundled = os.path.join(os.path.dirname(__file__), "data", "silero_vad.onnx")   # shipped in the repo (MIT licence)
    if os.path.exists(bundled):
        return bundled
    spec = importlib.util.find_spec("silero_vad")
    if not spec or not spec.submodule_search_locations:
        return None
    p = os.path.join(list(spec.submodule_search_locations)[0], "data", "silero_vad.onnx")
    return p if os.path.exists(p) else None


def silero_available() -> bool:
    return importlib.util.find_spec("onnxruntime") is not None and silero_model_path() is not None


def _session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = o.inter_op_num_threads = 1
        _SESSION = ort.InferenceSession(silero_model_path(), sess_options=o, providers=["CPUExecutionProvider"])
    return _SESSION


class SileroBackend:
    def __init__(self):
        self.sess = _session()
        self.state = np.zeros((2, 1, 128), np.float32)
        self.ctx = np.zeros((1, 32), np.float32)
        self.buf = np.zeros(0, np.float32)
        self.sr = np.array(SR, dtype=np.int64)

    def process(self, pcm: np.ndarray) -> list[float]:
        self.buf = np.concatenate([self.buf, pcm.astype(np.float32) / 32768.0])
        out = []
        while len(self.buf) >= CHUNK:
            x = np.concatenate([self.ctx, self.buf[None, :CHUNK]], axis=1).astype(np.float32)
            self.buf = self.buf[CHUNK:]
            p, self.state = self.sess.run(None, {"input": x, "state": self.state, "sr": self.sr})
            self.ctx = x[:, -32:]
            out.append(float(p[0][0]))
        return out


class EnergyBackend:
    """Fallback: adaptive-floor energy gate. Much weaker than Silero (music/noise look like speech)."""

    def __init__(self):
        self.buf = np.zeros(0, np.float32)
        self.floor = 60.0

    def process(self, pcm: np.ndarray) -> list[float]:
        self.buf = np.concatenate([self.buf, pcm.astype(np.float32)])
        out = []
        while len(self.buf) >= CHUNK:
            f, self.buf = self.buf[:CHUNK], self.buf[CHUNK:]
            rms = float(np.sqrt(np.mean(f ** 2)))
            self.floor = min(self.floor * 1.002, rms) if rms < self.floor else self.floor * 1.002
            out.append(1.0 if rms > max(220.0, self.floor * 3.0) else 0.0)
        return out


def make_backend() -> VadBackend:
    return SileroBackend() if silero_available() else EnergyBackend()


class VadDetector:
    source = "vad"

    def __init__(self, backend: VadBackend | None = None, t0: float = 0.0, on_chunks=2, off_chunks=10,
                 human_max_s=2.5, human_quiet_s=1.0, machine_runs=((4.0, 0.65), (7.0, 0.9))):
        self.b = backend or make_backend()
        self.t0 = t0
        self.n = 0
        self.on_chunks, self.off_chunks = on_chunks, off_chunks
        self.human_max_s, self.human_quiet_s = human_max_s, human_quiet_s
        self.machine_runs = machine_runs
        self.speaking = False
        self._on = self._off = 0
        self._run_start = 0.0
        self._machine_sent = 0
        self._pending: tuple[float, float] | None = None     # (burst_end_t, burst_dur) awaiting quiet
        self.bursts: list[tuple[float, float]] = []
        self.last_prob = 0.0

    def feed(self, mulaw: bytes) -> list[Signal]:
        sigs: list[Signal] = []
        for p in self.b.process(mulaw_decode(mulaw)):
            self.n += 1
            t = self.t0 + self.n * CHUNK_S
            self.last_prob = p
            if not self.speaking:
                self._on = self._on + 1 if p >= 0.5 else 0
                if self._on >= self.on_chunks:
                    self.speaking, self._off = True, 0
                    self._run_start = t - self.on_chunks * CHUNK_S
                    self._machine_sent = 0
                    self._pending = None                      # continuing utterance, not a finished burst
                    sigs.append(Signal(self.source, Sig.SPEECH_START, self._run_start))
                elif self._pending and t - self._pending[0] >= self.human_quiet_s:
                    end_t, dur = self._pending
                    self._pending = None
                    sigs += self._rhythm_on_quiet(end_t, dur, t)
            else:
                self._off = self._off + 1 if p < 0.35 else 0
                run = t - self._run_start
                for i, (need, conf) in enumerate(self.machine_runs):
                    if i == self._machine_sent and run >= need:
                        self._machine_sent += 1
                        sigs.append(Signal(self.source, Sig.RHYTHM, t, conf, {"verdict": "machine", "run": run}))
                if self._off >= self.off_chunks:
                    self.speaking, self._on = False, 0
                    end_t = t - self.off_chunks * CHUNK_S
                    dur = end_t - self._run_start
                    self.bursts.append((self._run_start, dur))
                    sigs.append(Signal(self.source, Sig.SPEECH_END, end_t, 1.0, {"dur": dur}))
                    if self._machine_sent == 0:
                        self._pending = (end_t, dur)
        return sigs

    def _rhythm_on_quiet(self, end_t: float, dur: float, t: float) -> list[Signal]:
        if dur > self.human_max_s:
            return []
        conf = 0.65
        prior = self.bursts[:-1]
        if prior and all(d <= 3.0 for _, d in prior[-2:]):
            conf = 0.75                                        # short back-and-forth turns
        return [Signal(self.source, Sig.RHYTHM, t, conf, {"verdict": "human", "dur": dur})]
