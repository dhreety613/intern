"""DetectionHub: every detector is an independent asyncio task with its own input queue, running in
parallel for the whole call (until a human is confirmed / the call ends). Detectors never talk to each
other; they push Signals onto one queue that the fusion task consumes.

   audio ──► vad ─────┐
        ├──► tone ────┤
        └──► music ───┤
   transcript ► text ─┼─► signals ─► fusion ─► verdicts ─► (controller)
   twilio status ► progress ┤                    ▲
   twilio AMD ───► amd ─────┘                    └── ticker (100 ms timers)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from ..classifier import LlmFallback, classify_with_fallback
from ..config import Settings
from ..models import AnsweredBy
from .audio import MusicDetector, ToneDetector
from .events import AmdDetector, ProgressDetector, TextDetector
from .fusion import Fusion
from .signals import Sig, Signal, Verdict
from .vad import VadDetector

log = logging.getLogger("telephony.detection")
VerdictFn = Callable[[Verdict], Awaitable[None]]
FRAME = 160
TICK_S = 0.1


class DetectionHub:
    def __init__(self, settings: Settings, on_verdict: VerdictFn, *, clock: Callable[[], float],
                 llm: LlmFallback | None = None, audio_batch_frames: int = 5, signal_log_cap: int = 5000):
        self.s, self.on_verdict, self.clock, self.llm = settings, on_verdict, clock, llm
        self.fusion = Fusion(settings)
        self.progress, self.amd, self.text = ProgressDetector(), AmdDetector(), TextDetector()
        self.vad, self.tone, self.music = VadDetector(), ToneDetector(), MusicDetector()
        self._audio = {"vad": self.vad, "tone": self.tone, "music": self.music}
        self.q: dict[str, asyncio.Queue] = {n: asyncio.Queue() for n in ("progress", "amd", "text", "vad", "tone", "music")}
        self.sig_q: asyncio.Queue[Signal] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._abuf = bytearray()
        self._batch = audio_batch_frames * FRAME
        self._audio_started = False
        self.running = False
        self.signal_log: list[dict] = []
        self.verdict_log: list[Verdict] = []
        self._cap = signal_log_cap

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        for n in self._audio:
            self._tasks.append(asyncio.create_task(self._audio_worker(n), name=f"det-{n}"))
        self._tasks += [asyncio.create_task(self._event_worker("progress"), name="det-progress"),
                        asyncio.create_task(self._event_worker("amd"), name="det-amd"),
                        asyncio.create_task(self._text_worker(), name="det-text"),
                        asyncio.create_task(self._fusion_worker(), name="fusion"),
                        asyncio.create_task(self._ticker(), name="ticker")]

    async def stop(self) -> None:
        self.running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    def alive(self) -> dict[str, bool]:
        return {t.get_name(): not t.done() for t in self._tasks}

    async def drain(self) -> None:
        """Wait until everything fed so far has been processed end to end (used by tests / webhooks)."""
        if not self.running:
            return
        self._flush_audio()
        for _ in range(3):
            for q in self.q.values():
                await q.join()
            await self.sig_q.join()

    # ---------------------------------------------------------------- inputs
    def feed_audio(self, mulaw: bytes) -> None:
        if not self._audio_started:
            self._audio_started = True
            t0 = self.clock()
            for d in self._audio.values():
                d.t0 = t0
        self._abuf += mulaw
        while len(self._abuf) >= self._batch:
            chunk = bytes(self._abuf[: self._batch])
            del self._abuf[: self._batch]
            for n in self._audio:
                self.q[n].put_nowait(chunk)

    def skip_audio(self, seconds: float) -> None:
        """Audio was not streaming for `seconds` (DTMF stream reconnect): keep detector clocks on call time."""
        for d in self._audio.values():
            d.t0 += max(0.0, seconds)

    def _flush_audio(self) -> None:
        if self._abuf:
            chunk = bytes(self._abuf)
            self._abuf.clear()
            for n in self._audio:
                self.q[n].put_nowait(chunk)

    def feed_status(self, form: dict) -> None:
        self.q["progress"].put_nowait((form, self.clock()))

    def feed_amd(self, answered_by: AnsweredBy) -> None:
        self.q["amd"].put_nowait((answered_by, self.clock()))

    def feed_transcript(self, text: str, final: bool = True) -> None:
        self.q["text"].put_nowait((text, final, self.clock()))

    def feed_signal(self, sig: Signal) -> None:
        """Escape hatch: inject a pre-built signal (tests, other detectors, model backends)."""
        self.sig_q.put_nowait(sig)

    # --------------------------------------------------------------- workers
    def _publish(self, sigs: list[Signal]) -> None:
        for s in sigs:
            if len(self.signal_log) < self._cap and s.kind not in (Sig.TICK, Sig.TEXT_PARTIAL):
                self.signal_log.append({"t": round(s.t, 2), "src": s.src, "kind": s.kind.value,
                                        "conf": round(s.conf, 2),
                                        "data": {k: (v.value if hasattr(v, "value") else v) for k, v in s.data.items()
                                                 if k in ("label", "text", "verdict", "dur", "freq", "cadence", "tone")}})
            self.sig_q.put_nowait(s)

    async def _audio_worker(self, name: str) -> None:
        det, q = self._audio[name], self.q[name]
        while True:
            chunk = await q.get()
            try:
                self._publish(await asyncio.to_thread(det.feed, chunk))     # CPU work off the event loop
            except Exception:
                log.exception("%s detector error", name)
            finally:
                q.task_done()

    async def _event_worker(self, name: str) -> None:
        q, det = self.q[name], getattr(self, name)
        while True:
            payload, t = await q.get()
            try:
                self._publish(det.feed(payload, t))
            except Exception:
                log.exception("%s detector error", name)
            finally:
                q.task_done()

    async def _text_worker(self) -> None:
        q = self.q["text"]
        while True:
            text, final, t = await q.get()
            try:
                label = await classify_with_fallback(text, self.llm) if final else None
                self._publish(self.text.feed(text, t, final, label))
            except Exception:
                log.exception("text detector error")
            finally:
                q.task_done()

    async def _fusion_worker(self) -> None:
        while True:
            sig = await self.sig_q.get()
            try:
                for v in self.fusion.on_signal(sig):
                    self.verdict_log.append(v)
                    await self._enrich(v)
                    await self.on_verdict(v)
            except Exception:
                log.exception("fusion error")
            finally:
                self.sig_q.task_done()

    async def _enrich(self, v) -> None:
        """IVR menu: let the LLM (if configured) pick the key by priority human > voicemail. Rules are the fallback."""
        pick = getattr(self.llm, "choose_key", None)
        if v.kind.value != "ivr_menu_ready" or pick is None or not v.data.get("options"):
            return
        try:
            v.data["llm_pick"] = await asyncio.wait_for(pick(v.data["options"], v.data.get("text", "")), self.s.llm_key_timeout_s)
        except Exception:
            v.data["llm_pick"] = None

    async def _ticker(self) -> None:
        while True:
            await asyncio.sleep(TICK_S)
            if self.sig_q.qsize() < 50:
                self.sig_q.put_nowait(Signal("timer", Sig.TICK, self.clock()))
