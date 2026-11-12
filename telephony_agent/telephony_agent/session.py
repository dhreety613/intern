"""CallSession: one call = DetectionHub (parallel detectors) -> CallController (actions) -> Twilio / TTS."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Awaitable, Callable

from .classifier import LlmFallback
from .config import Settings
from .controller import CallController
from .detection.hub import DetectionHub
from .detection.signals import Sig, Signal, Verdict
from .models import (Action, AnsweredBy, CallRequest, Dtmf, HandoffToBrain, Hangup, Outcome, Speak, SpeechDone,
                      State, Tick)
from .speech import STT, TTS
from .tones import dtmf_mulaw

log = logging.getLogger("telephony.session")
FRAME = 160
TICK_S = 0.25
TICK_FRAME_S = 0.02

HangupFn = Callable[[str], Awaitable[None]]
HandoffFn = Callable[["CallSession", HandoffToBrain], Awaitable[None]]
DtmfFn = Callable[[str, str], Awaitable[None]]      # (call_sid, digits)


class CallSession:
    def __init__(self, req: CallRequest, settings: Settings, *, stt_factory: Callable[[], STT], tts: TTS,
                 hangup: HangupFn, send_dtmf: DtmfFn | None = None, on_handoff: HandoffFn | None = None,
                 llm: LlmFallback | None = None):
        self.req, self.s = req, settings
        self.ctl = CallController(req, settings)
        self.stt_factory, self.tts, self._hangup, self.on_handoff = stt_factory, tts, hangup, on_handoff
        self._send_dtmf = send_dtmf
        self._reconnect_until = 0.0          # while now() < this, a closing media websocket is expected
        self._gap_from: float | None = None
        self.hub = DetectionHub(settings, self._on_verdict, clock=self.now, llm=llm)
        self.call_sid: str | None = None
        self.stream_sid: str | None = None
        self.ws = None
        self.stt: STT | None = None
        self._t0 = time.monotonic()
        self._tasks: list[asyncio.Task] = []
        self._send_lock = asyncio.Lock()
        self._mark_n = 0
        self._out_q: asyncio.Queue = asyncio.Queue()
        self.stats = {"frames_sent": 0, "marks_sent": 0, "send_errors": 0, "last_error": "", "frames_in": 0, "ws_none_drops": 0}
        self.done = asyncio.Event()
        self.status_history: list[str] = []

    def now(self) -> float:
        return time.monotonic() - self._t0

    @property
    def outcome(self) -> Outcome | None:
        return self.ctl.outcome

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        """Detectors run from dial time: status callbacks and AMD arrive before/without audio."""
        self.hub.start()
        self._tasks.append(asyncio.create_task(self._ctl_ticker(), name="ctl-ticker"))

    async def settle(self) -> None:
        await self.hub.drain()

    async def _on_verdict(self, v: Verdict) -> None:
        for a in self.ctl.handle(v):
            await self._exec(a)
        await self._after()

    async def _after(self) -> None:
        """Call over -> stop detection. After a human is confirmed (handoff) the detectors stay open
        (stage 2 re-runs detection on the brain's reason codes); they stop when the call ends."""
        if self.ctl.state is State.ENDED and self.hub.running:
            self.done.set()
            asyncio.get_running_loop().call_soon(lambda: asyncio.ensure_future(self.hub.stop()))
        elif self.ctl.state is State.HUMAN:
            self.done.set()

    async def _ctl_ticker(self) -> None:
        while self.ctl.state not in (State.ENDED, State.HUMAN):
            await asyncio.sleep(TICK_S)
            for a in self.ctl.handle(Tick(self.now())):
                await self._exec(a)
            await self._after()

    # -------------------------------------------------------------- actions
    async def _exec(self, a: Action) -> None:
        if isinstance(a, Speak):
            await self._send_audio(await self.tts.synth(a.text))
            await self._send_mark()
        elif isinstance(a, Dtmf):
            await self._press(a.digits)
        elif isinstance(a, Hangup):
            self.done.set()
            if self.call_sid:
                try:
                    await self._hangup(self.call_sid)
                except Exception:
                    log.exception("hangup failed")
        elif isinstance(a, HandoffToBrain):
            if self.on_handoff:
                await self.on_handoff(self, a)
            else:
                log.info("HANDOFF call=%s (stub; stage 2 wires the brain)", self.req.call_id)

    async def _press(self, digits: str) -> None:
        if self.s.dtmf_mode == "twilio" and self._send_dtmf and self.call_sid:
            self._reconnect_until = self.now() + self.s.reconnect_window_s
            self._gap_from = self.now()
            try:
                await self._send_dtmf(self.call_sid, digits)
                return
            except Exception:
                log.exception("twilio dtmf failed; falling back to in-band tones")
                self._reconnect_until, self._gap_from = 0.0, None
        await self._send_audio(dtmf_mulaw(digits))

    async def _send_json(self, obj: dict) -> None:
        if self.ws is None:
            return
        async with self._send_lock:
            await self.ws.send_text(json.dumps(obj))

    async def _send_audio(self, mulaw: bytes) -> None:
        """Queue audio for the paced sender: 160-byte (20 ms) frames in real time, exactly how call_agent feeds
        Twilio (big bursts are not reliably played, DTMF tones included). Returns at once; marks keep order."""
        if not mulaw or not self.stream_sid:
            return
        if len(mulaw) % FRAME:
            mulaw += bytes([0xFF]) * (FRAME - len(mulaw) % FRAME)   # mu-law silence
        for i in range(0, len(mulaw), FRAME):
            self._out_q.put_nowait(("media", mulaw[i:i + FRAME]))

    async def _send_mark(self) -> None:
        self._mark_n += 1
        self._out_q.put_nowait(("mark", f"speech_done_{self._mark_n}"))

    async def _sender(self) -> None:
        loop = asyncio.get_running_loop()
        nxt = loop.time()
        while True:
            kind, val = await self._out_q.get()
            now = loop.time()
            if nxt < now - 0.1:
                nxt = now                                      # fell behind / was idle: restart the clock
            try:
                if self.ws is None:
                    self.stats["ws_none_drops"] += 1
                if kind == "media":
                    self.stats["frames_sent"] += 1
                    await self._send_json({"event": "media", "streamSid": self.stream_sid,
                                           "media": {"payload": base64.b64encode(val).decode()}})
                    nxt += TICK_FRAME_S
                    await asyncio.sleep(max(0.0, nxt - loop.time()))
                else:
                    self.stats["marks_sent"] += 1
                    await self._send_json({"event": "mark", "streamSid": self.stream_sid, "mark": {"name": val}})
            except Exception as e:
                self.stats["send_errors"] += 1
                self.stats["last_error"] = f"{type(e).__name__}: {e}"[:200]
                log.exception("audio sender error")

    # ------------------------------------------------------------ websocket
    async def run_media(self, ws) -> None:
        self.ws = ws
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                ev = msg.get("event")
                if ev == "start":
                    await self._on_start(msg["start"])
                elif ev == "media":
                    m = msg["media"]
                    if m.get("track", "inbound") == "inbound":
                        audio = base64.b64decode(m["payload"])
                        self.stats["frames_in"] += 1
                        if self.stt:
                            self.stt.feed(audio)
                        self.hub.feed_audio(audio)
                elif ev == "mark":
                    if str(msg.get("mark", {}).get("name", "")).startswith("speech_done"):
                        for a in self.ctl.handle(SpeechDone(self.now())):
                            await self._exec(a)
                        await self._after()
                elif ev == "stop":
                    break
        except Exception as e:
            if type(e).__name__ not in ("WebSocketDisconnect", "ConnectionClosed"):
                log.exception("media loop error")
        finally:
            if self.now() < self._reconnect_until and self.ctl.state not in (State.ENDED, State.HUMAN):
                self.ws = None                                   # DTMF redirect: Twilio re-opens the stream
                self._tasks.append(asyncio.create_task(self._reconnect_watchdog(), name="reconnect-watchdog"))
            else:
                await self._shutdown()

    async def _reconnect_watchdog(self) -> None:
        await asyncio.sleep(max(0.0, self._reconnect_until - self.now()) + 0.2)
        if self.ws is None and self.ctl.state not in (State.ENDED, State.HUMAN):
            self._reconnect_until = 0.0
            await self._shutdown()                               # stream never came back: call is gone

    async def _on_start(self, start: dict) -> None:
        self.stream_sid = start.get("streamSid")
        self.call_sid = start.get("callSid") or self.call_sid
        if self.stt is not None:                                 # reconnect after a DTMF redirect
            if self._gap_from is not None:
                self.hub.skip_audio(self.now() - self._gap_from)
            self._gap_from, self._reconnect_until = None, 0.0
            return
        if not self.hub.running:
            self.hub.start()
        self._tasks.append(asyncio.create_task(self._sender(), name="audio-sender"))
        self.stt = self.stt_factory()
        self.hub.fusion.stt_busy = lambda: bool(getattr(self.stt, "busy", False))
        await self.stt.start()
        self._tasks.append(asyncio.create_task(self._pump_stt(), name="stt-pump"))
        self.hub.feed_status({"CallStatus": "in-progress"})        # media stream start == answered

    async def _pump_stt(self) -> None:
        assert self.stt
        async for text, final in self.stt.results():
            self.hub.feed_transcript(text, final)

    async def _shutdown(self) -> None:
        if self.ctl.state not in (State.ENDED, State.HUMAN) and self.hub.running:
            self.hub.feed_status({"CallStatus": "completed"})      # stream closed == far end hung up
            await self.hub.drain()
        if self.stt and self.ctl.state is not State.HUMAN:         # human: brain takes the stream (stage 2)
            await self.stt.close()
        for t in self._tasks:
            t.cancel()
        await self.hub.stop()
        self.done.set()
