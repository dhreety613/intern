"""Sync test rig: Fusion + CallController with an explicit clock (no sleeping)."""
from telephony_agent.config import Settings
from telephony_agent.controller import CallController
from telephony_agent.detection.events import AmdDetector, ProgressDetector, TextDetector
from telephony_agent.detection.fusion import Fusion
from telephony_agent.detection.signals import Sig, Signal
from telephony_agent.models import AnsweredBy, CallRequest, SpeechDone, Tick


class Rig:
    def __init__(self, ext=None, answered=True, **kw):
        self.s = Settings(callback_number="4125550123", **kw)
        self.f = Fusion(self.s)
        self.c = CallController(CallRequest(to="+15551230000", call_id="c1", extension=ext), self.s)
        self.t = 0.0
        self.out = []
        self.verdicts = []
        self._text, self._amd, self._prog = TextDetector(), AmdDetector(), ProgressDetector()
        if answered:
            self.status("in-progress")

    # raw feeding -------------------------------------------------------
    def push(self, sig: Signal):
        acts = []
        for v in self.f.on_signal(sig):
            self.verdicts.append(v)
            acts += self.c.handle(v)
        self.out += acts
        return acts

    def sig(self, kind: Sig, conf=1.0, **data):
        return self.push(Signal("test", kind, self.t, conf, data))

    def say(self, text, final=True):
        acts = []
        for s in self._text.feed(text, self.t, final):
            acts += self.push(s)
        return acts

    def amd(self, ab: AnsweredBy):
        acts = []
        for s in self._amd.feed(ab, self.t):
            acts += self.push(s)
        return acts

    def status(self, st, **kw):
        acts = []
        for s in self._prog.feed({"CallStatus": st, **kw}, self.t):
            acts += self.push(s)
        return acts

    def speech_done(self):
        acts = self.c.handle(SpeechDone(self.t))
        self.out += acts
        return acts

    def talk(self, secs):
        """VAD-style: someone speaks for `secs`."""
        a = self.sig(Sig.SPEECH_START)
        self.t += secs
        return a + self.sig(Sig.SPEECH_END, dur=secs)

    def wait(self, secs, step=0.1):
        acts, end = [], self.t + secs
        while self.t < end - 1e-9:
            self.t += step
            acts += self.push(Signal("timer", Sig.TICK, self.t))
            a = self.c.handle(Tick(self.t))
            self.out += a
            acts += a
        return acts


def kinds(acts):
    return [type(a).__name__ for a in acts]
