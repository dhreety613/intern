"""Offline replay: push a recording (+ timed transcript / Twilio events) through the real detectors,
fusion and controller on a virtual clock. Used to score detection and to tune thresholds on real calls.

    python -m telephony_agent.replay samples/manifest.json

Manifest entry:
  {"file": "01_x.wav", "events": [{"t": 0, "type": "status", "status": "in-progress"},
                                  {"t": 6.1, "type": "text", "text": "...", "final": true},
                                  {"t": 3.0, "type": "amd", "answered_by": "machine_start"}],
   "expect": {"verdicts": ["vm_greeting", "vm_beep_ready"], "outcome": "voicemail_left",
              "never": ["human"], "actions": ["Speak"]}}
"""
from __future__ import annotations

import json
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Settings
from .controller import CallController
from .detection.audio import MusicDetector, ToneDetector
from .detection.events import AmdDetector, ProgressDetector, TextDetector
from .detection.fusion import Fusion
from .detection.signals import Sig, Signal, Verdict
from .detection.vad import VadDetector
from .models import Action, AnsweredBy, CallRequest, Dtmf, Hangup, Speak, SpeechDone, State, Tick
from .tones import mulaw_encode

TTS_SECONDS = 1.2        # assumed playback time of anything we say


@dataclass
class ReplayResult:
    verdicts: list[Verdict] = field(default_factory=list)
    actions: list[tuple[float, Action]] = field(default_factory=list)
    outcome: str | None = None
    state: str = ""
    signals: list[Signal] = field(default_factory=list)

    def timeline(self) -> list[str]:
        rows = [(v.t, f"verdict {v.kind.value:<20} {','.join(v.evidence)}") for v in self.verdicts]
        rows += [(t, f"ACTION  {type(a).__name__} {getattr(a, 'text', getattr(a, 'digits', getattr(a, 'outcome', '')))}")
                 for t, a in self.actions]
        return [f"{t:6.2f}s  {r}" for t, r in sorted(rows, key=lambda x: x[0])]


def load_wav(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, ch)[:, 0]
    if sr != 8000:
        pcm = np.interp(np.linspace(0, len(pcm) - 1, int(len(pcm) * 8000 / sr)), np.arange(len(pcm)), pcm).astype(np.int16)
    return pcm


def replay(pcm: np.ndarray, events: list[dict], settings: Settings | None = None, targets=None,
           extension: str | None = None, tail_s: float = 4.0, step: float = 0.1) -> ReplayResult:
    s = settings or Settings(callback_number="4125550123")
    fusion, ctl = Fusion(s), CallController(CallRequest(to="+10000000000", call_id="replay", extension=extension), s, targets)
    vad, tone, music = VadDetector(), ToneDetector(), MusicDetector()
    text, amd, prog = TextDetector(), AmdDetector(), ProgressDetector()
    res = ReplayResult()
    pcm = np.concatenate([pcm, np.zeros(int(tail_s * 8000), dtype=np.int16)])   # a live stream keeps delivering line silence
    mu = mulaw_encode(pcm)
    chunk = int(step * 8000)
    evs = sorted(events, key=lambda e: e["t"])
    ei, done_at = 0, []
    n_chunks = -(-len(mu) // chunk)

    def push(sig: Signal, t_now: float) -> None:
        res.signals.append(sig)
        for v in fusion.on_signal(sig):
            res.verdicts.append(v)
            for a in ctl.handle(v):
                res.actions.append((v.t, a))
                if isinstance(a, Speak):
                    done_at.append(v.t + TTS_SECONDS)

    for k in range(n_chunks):
        t_end = (k + 1) * step
        sigs: list[Signal] = []
        while ei < len(evs) and evs[ei]["t"] <= t_end:
            e = evs[ei]
            ei += 1
            if e["type"] == "status":
                sigs += prog.feed({"CallStatus": e["status"], **e.get("extra", {})}, e["t"])
            elif e["type"] == "amd":
                sigs += amd.feed(AnsweredBy(e["answered_by"]), e["t"])
            elif e["type"] == "text":
                sigs += text.feed(e["text"], e["t"], e.get("final", True))
        a = mu[k * chunk:(k + 1) * chunk]
        if a:
            for d in (vad, tone, music):
                sigs += d.feed(a)
        sigs.sort(key=lambda x: x.t)
        for sg in sigs:
            push(sg, t_end)
        push(Signal("timer", Sig.TICK, t_end), t_end)
        for dt in [d for d in done_at if d <= t_end]:
            done_at.remove(dt)
            for act in ctl.handle(SpeechDone(dt)):
                res.actions.append((dt, act))
        for act in ctl.handle(Tick(t_end)):
            res.actions.append((t_end, act))
        if ctl.state in (State.ENDED, State.HUMAN):
            break
    res.outcome = ctl.outcome.value if ctl.outcome else None
    res.state = ctl.state.value
    return res


def check(res: ReplayResult, expect: dict) -> list[str]:
    """Return a list of problems (empty = pass)."""
    problems = []
    kinds = [v.kind.value for v in res.verdicts]
    it = iter(kinds)
    for want in expect.get("verdicts", []):
        if want not in it:
            problems.append(f"missing verdict '{want}' (in order); got {kinds}")
            break
    for bad in expect.get("never", []):
        if bad in kinds:
            problems.append(f"unexpected verdict '{bad}'")
    if "outcome" in expect and res.outcome != expect["outcome"]:
        problems.append(f"outcome {res.outcome!r} != {expect['outcome']!r}")
    acts = [type(a).__name__ + (":" + a.digits if isinstance(a, Dtmf) else "") for _, a in res.actions]
    for want in expect.get("actions", []):
        if want not in acts:
            problems.append(f"missing action {want}; got {acts}")
    return problems


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    mpath = Path(argv[0])
    manifest = json.loads(mpath.read_text())
    verbose = "-v" in argv
    bad = 0
    for item in manifest["samples"]:
        res = replay(load_wav(mpath.parent / item["file"]), item["events"], Settings(callback_number="4125550123"))
        probs = check(res, item.get("expect", {}))
        bad += bool(probs)
        print(("PASS " if not probs else "FAIL ") + item["file"] + f"  -> {res.outcome or res.state}")
        for p in probs:
            print("      " + p)
        if verbose or probs:
            for row in res.timeline():
                print("      " + row)
    print(f"\n{len(manifest['samples']) - bad}/{len(manifest['samples'])} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
