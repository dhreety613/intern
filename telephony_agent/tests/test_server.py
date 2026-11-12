import asyncio
import base64
import json
import time

import httpx
from starlette.testclient import TestClient

from telephony_agent.config import Settings
from telephony_agent.server import create_app, twilio_signature
from telephony_agent.tones import mulaw_encode
import numpy as np


class FakeSTT:
    inst = None

    def __init__(self):
        FakeSTT.inst = self
        self.fed = 0

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.q = asyncio.Queue()

    def feed(self, b):
        self.fed += len(b)

    async def results(self):
        while True:
            item = await self.q.get()
            if item is None:
                return
            yield item

    async def close(self):
        self.q.put_nowait(None)

    def push(self, text, final=True):
        self.loop.call_soon_threadsafe(self.q.put_nowait, (text, final))


class FakeTTS:
    async def synth(self, text):
        return b"\xff" * 1600


def twilio_mock(calls, lookup_valid=True):
    def handler(req: httpx.Request):
        calls.append((req.method, req.url.path, req.content.decode()))
        if "lookups.twilio.com" in str(req.url):
            if not lookup_valid:
                return httpx.Response(200, json={"valid": False, "validation_errors": ["TOO_SHORT"]})
            return httpx.Response(200, json={"valid": True, "phone_number": "+15551230000",
                                             "line_type_intelligence": {"type": "landline"}})
        if req.url.path.endswith("/Calls.json"):
            return httpx.Response(201, json={"sid": "CA123"})
        return httpx.Response(200, json={})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def make(calls, **kw):
    s = Settings(account_sid="ACx", auth_token="tok", from_number="+15550000000",
                 public_base_url="https://agent.example.com", callback_number="4125550123",
                 dtmf_mode="twilio")
    app = create_app(s, stt_factory=FakeSTT, tts=FakeTTS(), http=twilio_mock(calls, **kw),
                     validate_signatures=False, api_key="")
    return app


def wait_for(cond, secs=3):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_invalid_number_short_circuits_before_dialing():
    calls = []
    with TestClient(make(calls, lookup_valid=False)) as c:
        r = c.post("/calls", json={"to": "+1555"}).json()
        assert r["outcome"] == "invalid_number"
        assert not any(p.endswith("/Calls.json") for _, p, _ in calls)


def test_dial_params_and_status_callbacks():
    calls = []
    with TestClient(make(calls)) as c:
        r = c.post("/calls", json={"to": "+15551230000"}).json()
        cid = r["call_id"]
        body = [b for m, p, b in calls if p.endswith("/Calls.json")][0]
        assert "AsyncAmd=true" in body and "MachineDetection=DetectMessageEnd" in body and "Timeout=25" in body
        c.post(f"/status/{cid}", data={"CallStatus": "ringing"})
        c.post(f"/status/{cid}", data={"CallStatus": "no-answer"})
        out = c.get(f"/calls/{cid}").json()
        assert out["outcome"] == "no_answer" and out["retryable"] is True
        assert out["status_history"] == ["ringing", "no-answer"]


def test_twiml_streams_to_media_ws():
    calls = []
    with TestClient(make(calls)) as c:
        cid = c.post("/calls", json={"to": "+15551230000"}).json()["call_id"]
        xml = c.post(f"/twiml/{cid}").text
        assert f"wss://agent.example.com/media/{cid}" in xml and "<Connect><Stream" in xml


def test_signature_enforced():
    calls = []
    s = Settings(account_sid="ACx", auth_token="tok", public_base_url="https://agent.example.com")
    app = create_app(s, stt_factory=FakeSTT, tts=FakeTTS(), http=twilio_mock(calls), validate_signatures=True, api_key="")
    with TestClient(app) as c:
        assert c.post("/status/abc", data={"CallStatus": "ringing"}).status_code == 403
        sig = twilio_signature("tok", "https://agent.example.com/status/abc", {"CallStatus": "ringing"})
        assert c.post("/status/abc", data={"CallStatus": "ringing"}, headers={"X-Twilio-Signature": sig}).status_code == 200


def test_end_to_end_voicemail_over_websocket():
    calls = []
    with TestClient(make(calls)) as c:
        cid = c.post("/calls", json={"to": "+15551230000"}).json()["call_id"]
        with c.websocket_connect(f"/media/{cid}") as ws:
            ws.send_text(json.dumps({"event": "connected"}))
            ws.send_text(json.dumps({"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA123"}}))
            assert wait_for(lambda: FakeSTT.inst is not None and hasattr(FakeSTT.inst, "q"))
            silence = base64.b64encode(mulaw_encode(np.zeros(160, np.int16))).decode()
            ws.send_text(json.dumps({"event": "media", "media": {"track": "inbound", "payload": silence}}))
            assert wait_for(lambda: FakeSTT.inst.fed == 160)
            FakeSTT.inst.push("Hi you've reached John, leave a message after the tone.")
            assert wait_for(lambda: c.get(f"/calls/{cid}").json()["state"] == "voicemail")
            c.post(f"/amd/{cid}", data={"AnsweredBy": "machine_end_beep"})
            frames = []
            m2 = json.loads(ws.receive_text())
            while m2["event"] == "media":                   # paced 20 ms frames of 160 bytes, then the mark
                assert m2["streamSid"] == "MZ1" and len(base64.b64decode(m2["media"]["payload"])) == 160
                frames.append(m2)
                m2 = json.loads(ws.receive_text())
            assert len(frames) == 10 and m2["event"] == "mark"
            ws.send_text(json.dumps({"event": "mark", "mark": {"name": m2["mark"]["name"]}}))
            assert wait_for(lambda: any(p.endswith("/Calls/CA123.json") and "Status=completed" in b for _, p, b in calls))
            out = c.get(f"/calls/{cid}").json()
            assert out["outcome"] == "voicemail_left"
            kinds = [e["kind"] for e in out["log"]]
            assert "speak" in kinds and kinds[-1] == "end"
            assert any(x["kind"] == "amd_end_beep" for x in out["signals"])
            assert any(x["kind"] == "text_label" and x["data"]["label"] == "voicemail_greeting" for x in out["signals"])


def test_ivr_key_press_goes_through_twilio_and_survives_the_stream_reconnect():
    calls = []
    with TestClient(make(calls)) as c:
        cid = c.post("/calls", json={"to": "+15551230000"}).json()["call_id"]
        with c.websocket_connect(f"/media/{cid}") as ws:
            ws.send_text(json.dumps({"event": "start", "start": {"streamSid": "MZ2", "callSid": "CA123"}}))
            assert wait_for(lambda: FakeSTT.inst is not None and hasattr(FakeSTT.inst, "q"))
            stt1 = FakeSTT.inst
            stt1.push("For sales, press 1. For calibration services, press 2.")
            # key press = REST call with <Play digits> + a new <Stream> pointing back at us
            assert wait_for(lambda: any(p.endswith("/Calls/CA123.json") and "Twiml" in b for _, p, b in calls))
            body = [b for _, p, b in calls if p.endswith("/Calls/CA123.json") and "Twiml" in b][0]
            from urllib.parse import parse_qs
            twiml = parse_qs(body)["Twiml"][0]
            assert 'digits="w2"' in twiml and f"wss://agent.example.com/media/{cid}" in twiml and "<Connect><Stream" in twiml
            ws.send_text(json.dumps({"event": "stop"}))           # Twilio closes the old stream on redirect
        # not treated as a hangup
        assert c.get(f"/calls/{cid}").json()["outcome"] is None
        with c.websocket_connect(f"/media/{cid}") as ws2:           # ...and opens the new one
            ws2.send_text(json.dumps({"event": "start", "start": {"streamSid": "MZ3", "callSid": "CA123"}}))
            time.sleep(0.2)
            assert FakeSTT.inst is stt1                              # same STT/detectors continue, nothing restarted
            stt1.push("Hello, this is Dave")
            assert wait_for(lambda: c.get(f"/calls/{cid}").json()["state"] == "human", 4)
            out = c.get(f"/calls/{cid}").json()
            assert out["outcome"] == "human_confirmed"
            assert any(e["kind"] == "dtmf" and e["keys"] == "2" for e in out["log"])


def test_stream_that_never_returns_after_a_key_press_counts_as_hangup():
    calls = []
    app = make(calls)
    app.state.settings.reconnect_window_s = 0.3
    with TestClient(app) as c:
        cid = c.post("/calls", json={"to": "+15551230000"}).json()["call_id"]
        with c.websocket_connect(f"/media/{cid}") as ws:
            ws.send_text(json.dumps({"event": "start", "start": {"streamSid": "MZ4", "callSid": "CA123"}}))
            assert wait_for(lambda: FakeSTT.inst is not None and hasattr(FakeSTT.inst, "q"))
            FakeSTT.inst.push("For sales, press 1. For calibration services, press 2.")
            assert wait_for(lambda: any("Twiml" in b for _, _, b in calls))
            ws.send_text(json.dumps({"event": "stop"}))
        assert wait_for(lambda: c.get(f"/calls/{cid}").json()["outcome"] == "remote_hangup", 3)


def test_inband_mode_still_available():
    calls = []
    s = Settings(account_sid="ACx", auth_token="tok", from_number="+15550000000",
                 public_base_url="https://agent.example.com", dtmf_mode="inband")
    app = create_app(s, stt_factory=FakeSTT, tts=FakeTTS(), http=twilio_mock(calls), validate_signatures=False, api_key="")
    with TestClient(app) as c:
        cid = c.post("/calls", json={"to": "+15551230000"}).json()["call_id"]
        with c.websocket_connect(f"/media/{cid}") as ws:
            ws.send_text(json.dumps({"event": "start", "start": {"streamSid": "MZ5", "callSid": "CA123"}}))
            assert wait_for(lambda: FakeSTT.inst is not None and hasattr(FakeSTT.inst, "q"))
            FakeSTT.inst.push("For sales, press 1. For calibration services, press 2.")
            total = 0
            for _ in range(6):                              # DTMF "1": 250 ms tone + 120 ms gap as 160-byte frames
                m = json.loads(ws.receive_text())
                assert m["event"] == "media"
                total += len(base64.b64decode(m["media"]["payload"]))
            assert total == 960
            assert not any("Twiml" in b for _, _, b in calls)
