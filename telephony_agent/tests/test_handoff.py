"""Human confirmed -> the brain takes over the SAME live Twilio websocket (no redirect, no audio gap)."""
import base64
import json

import numpy as np
from starlette.testclient import TestClient

from telephony_agent.config import Settings
from telephony_agent.server import create_app
from telephony_agent.tones import mulaw_encode

from test_server import FakeSTT, FakeTTS, twilio_mock, wait_for


def make_app(calls, runner):
    s = Settings(account_sid="ACx", auth_token="tok", from_number="+15550000000",
                 public_base_url="https://agent.example.com", callback_number="4125550123", dtmf_mode="twilio")
    return create_app(s, stt_factory=FakeSTT, tts=FakeTTS(), http=twilio_mock(calls), validate_signatures=False,
                      api_key="", brain_runner=runner)


def frame(level: int) -> str:
    return base64.b64encode(mulaw_encode(np.full(160, level, np.int16))).decode()


def media(level: int) -> str:
    return json.dumps({"event": "media", "media": {"track": "inbound", "payload": frame(level)}})


def test_human_confirmed_hands_the_live_websocket_to_the_brain():
    calls, seen = [], {}

    async def brain(ws, sess):
        seen["ctx"], seen["heard"] = dict(sess.req.context), list(sess.ctl.heard)
        seen["sids"] = (sess.call_sid, sess.stream_sid)
        seen["first"] = json.loads(await ws.receive_text())        # the first frame AFTER the handoff is the brain's
        await ws.send_text(json.dumps({"event": "mark", "mark": {"name": "brain_says_hi"}}))
        while True:
            if json.loads(await ws.receive_text()).get("event") == "stop":
                seen["stopped"] = True
                return

    ctx = {"customer_name": "Jim", "person_id": "p1", "deal_id": "d9", "products": ["365-HX"]}
    with TestClient(make_app(calls, brain)) as c:
        cid = c.post("/calls", json={"to": "+15551230000", "context": ctx}).json()["call_id"]
        with c.websocket_connect(f"/media/{cid}") as ws:
            ws.send_text(json.dumps({"event": "connected"}))
            ws.send_text(json.dumps({"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA123"}}))
            assert wait_for(lambda: FakeSTT.inst is not None and hasattr(FakeSTT.inst, "q"))
            stt = FakeSTT.inst
            ws.send_text(media(0))
            stt.push("Hello, this is Dave")
            assert wait_for(lambda: c.app.state.sessions[cid].handed_off, 4)
            ws.send_text(media(0))                 # telephony consumes one last frame, sees the flag, releases the socket
            ws.send_text(media(1000))              # ...so this one is the brain's
            reply = json.loads(ws.receive_text())
            assert reply == {"event": "mark", "mark": {"name": "brain_says_hi"}}
            ws.send_text(json.dumps({"event": "stop"}))
        assert wait_for(lambda: seen.get("stopped"))

    assert seen["first"]["event"] == "media" and seen["first"]["media"]["payload"] == frame(1000)
    assert seen["ctx"] == ctx and seen["sids"] == ("CA123", "MZ1")
    assert any("Dave" in h for h in seen["heard"])
    # telephony did NOT hang up on the human, and left detection/STT shut down behind it
    assert not any(p.endswith("/Calls/CA123.json") and "Status=completed" in b for _, p, b in calls)
    sess = c.app.state.sessions[cid]
    assert sess.outcome.value == "human_confirmed" and not sess.hub.running


def test_brain_crash_hangs_the_call_up_instead_of_leaving_it_open():
    calls = []

    async def brain(ws, sess):
        raise RuntimeError("brain exploded")

    with TestClient(make_app(calls, brain)) as c:
        cid = c.post("/calls", json={"to": "+15551230000"}).json()["call_id"]
        with c.websocket_connect(f"/media/{cid}") as ws:
            ws.send_text(json.dumps({"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA123"}}))
            assert wait_for(lambda: FakeSTT.inst is not None and hasattr(FakeSTT.inst, "q"))
            FakeSTT.inst.push("Hello, this is Dave")
            assert wait_for(lambda: c.app.state.sessions[cid].handed_off, 4)
            ws.send_text(media(0))
            ws.send_text(media(0))
            assert wait_for(lambda: any(p.endswith("/Calls/CA123.json") and "Status=completed" in b for _, p, b in calls), 4)
