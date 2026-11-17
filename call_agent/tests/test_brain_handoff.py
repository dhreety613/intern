"""The brain only talks to confirmed humans: it takes over a live call from the telephony layer and has no
voicemail / IVR / AMD / DTMF handling of its own. Speech services and the LLM are stubbed; no network."""
import asyncio
import audioop
import base64
import json
import math
import struct

import pytest

from call_agent.app import handoff, media_stream
from call_agent.app.greeting import opening_message


def mulaw(level: int, ms: int) -> bytes:
    n = 8 * ms
    pcm = b"".join(struct.pack("<h", int(level * math.sin(i * 0.3))) for i in range(n))
    return audioop.lin2ulaw(pcm, 2)


def media_event(audio: bytes) -> str:
    return json.dumps({"event": "media", "media": {"track": "inbound", "payload": base64.b64encode(audio).decode()}})


class FakeWS:
    def __init__(self, incoming):
        self.q: asyncio.Queue = asyncio.Queue()
        for m in incoming:
            self.q.put_nowait(m)
        self.sent: list[dict] = []
        self.closed = False

    async def receive_text(self):
        return await self.q.get()

    async def send_text(self, text):
        self.sent.append(json.loads(text))

    async def close(self):
        self.closed = True


class FakeTTS:
    spoken: list[str] = []

    async def text_to_speech_stream(self, text):
        FakeTTS.spoken.append(text)
        yield b"\xff" * 320                      # two 20 ms frames


class FakeSTT:
    async def transcribe_stream(self, audio):
        async for _ in audio:
            pass
        yield "Yes, this is Jim"


@pytest.fixture(autouse=True)
def stubs(monkeypatch):
    FakeTTS.spoken = []
    monkeypatch.setattr(media_stream, "get_stt_engine", lambda *_: FakeSTT())
    monkeypatch.setattr(media_stream, "get_tts_engine", lambda *_: FakeTTS())


def test_brain_has_no_telephony_handling():
    for name in ("handle_voicemail", "_leave_voicemail_after_beep", "_send_dtmf", "_arm_voicemail_key_timer"):
        assert not hasattr(media_stream.TwilioMediaStream, name), name
    from call_agent.app.graph.graph import call_graph
    nodes = set(call_graph.get_graph().nodes)
    assert {"memory", "brain", "brain_fast", "decision", "finalize"} <= nodes
    assert not any("ivr" in n or "voicemail" in n for n in nodes)
    for gone in ("ivr_agent", "voicemail_agent", "dtmf"):
        with pytest.raises(ImportError):
            __import__(f"call_agent.app.{'agents.' if gone != 'dtmf' else ''}{gone}", fromlist=["x"])


def test_start_event_carries_context_as_custom_parameters():
    ev = handoff.build_start_event(call_sid="CA1", stream_sid="MZ1", context={
        "customer_name": "Jim", "calibration_dates": ["September 15, 2026", "October 3, 2026"],
        "products": ["365-HX", "410"], "person_id": 7, "deal_id": "d9", "unused": "x"})
    assert ev["streamSid"] == "MZ1" and ev["start"]["callSid"] == "CA1"
    cp = ev["start"]["customParameters"]
    assert cp["customer_name"] == "Jim" and cp["person_id"] == "7" and cp["deal_id"] == "d9"
    assert cp["calibration_dates"] == "September 15, 2026, October 3, 2026" and cp["products"] == "365-HX, 410"
    assert "unused" not in cp


@pytest.mark.asyncio
async def test_run_brain_takes_over_a_live_call_and_talks_to_the_human(monkeypatch):
    seen = {}

    def fake_process(self, text):
        seen["said"] = text
        return {"response": "Great, thanks Jim.", "decision": {"present_task": "CONTINUE", "response": "Great, thanks Jim."},
                "call_ended": False}

    monkeypatch.setattr(media_stream.CallSession, "process", fake_process)
    loud, quiet = mulaw(9000, 20), mulaw(0, 20)
    ws = FakeWS([])

    task = asyncio.create_task(handoff.run_brain(
        ws, call_id="c1", call_sid="CA123", stream_sid="MZ1", heard=["Hello"],
        context={"customer_name": "Jim", "calibration_dates": "September 15, 2026", "products": "365-HX",
                 "person_id": "p1", "deal_id": "d9"}))
    for _ in range(50):                                     # the brain greets first...
        await asyncio.sleep(0.1)
        if FakeTTS.spoken:
            break
    await asyncio.sleep(0.5)                                # ...and finishes speaking before the human replies
    for m in [media_event(loud)] * 25 + [media_event(quiet)] * 45:
        ws.q.put_nowait(m)                                  # ~0.5 s of speech then a pause = end of turn
        await asyncio.sleep(0.002)                          # Twilio paces frames; never hand the loop 70 at once
    for _ in range(100):                                    # wait for the brain to answer the human's turn
        await asyncio.sleep(0.1)
        if "said" in seen and len(FakeTTS.spoken) >= 2:
            break
    ws.q.put_nowait(json.dumps({"event": "stop"}))
    await asyncio.wait_for(task, 10)

    assert FakeTTS.spoken[0] == opening_message("Jim")       # fresh scripted greeting on handover
    assert seen["said"] == "Yes, this is Jim"                # the human's reply went through STT -> brain
    assert FakeTTS.spoken[1] == "Great, thanks Jim."         # ...and the answer was spoken back
    media = [m for m in ws.sent if m["event"] == "media"]
    assert media and all(m["streamSid"] == "MZ1" for m in media)   # frames go out on the handed-over stream
    assert ws.closed


@pytest.mark.asyncio
async def test_plain_twilio_flow_still_works_without_a_preamble(monkeypatch):
    """handle() with no preamble reads start from the socket exactly as before."""
    ws = FakeWS([json.dumps({"event": "connected"}),
                 json.dumps({"event": "start", "streamSid": "MZ2",
                             "start": {"streamSid": "MZ2", "callSid": "CA9", "customParameters": {"customer_name": "Ann"}}}),
                 ])
    h = media_stream.TwilioMediaStream(websocket=ws)
    t = asyncio.create_task(h.handle())
    for _ in range(50):
        await asyncio.sleep(0.1)
        if FakeTTS.spoken:
            break
    ws.q.put_nowait(json.dumps({"event": "stop"}))
    await asyncio.wait_for(t, 10)
    assert FakeTTS.spoken[0] == opening_message("Ann") and h.session.state["customer_name"] == "Ann"


def test_calibration_dates_are_not_split_at_their_own_comma():
    """Regression: the date-splitting regex held a literal backspace character instead of the word-boundary
    escape, so 'September 15, 2026' reached the brain as ['September 15', '2026']."""
    async def go():
        h = media_stream.TwilioMediaStream(websocket=FakeWS([]))

        async def noop():
            pass
        h._send_opening_greeting = noop
        for raw, want in (("September 15, 2026", ["September 15, 2026"]),
                          ("September 15, 2026, October 3, 2026", ["September 15, 2026", "October 3, 2026"])):
            h._handle_start({"event": "start", "streamSid": "M",
                             "start": {"streamSid": "M", "callSid": "C", "customParameters": {"calibration_dates": raw}}})
            assert h.session.state["calibration_dates"] == want
            h._greeting_task.cancel()
    asyncio.run(go())
