"""Place outbound calls: Lookup gate -> Twilio REST with async AMD (DetectMessageEnd)."""
from __future__ import annotations

import httpx

from .config import Settings
from .lookup import lookup_number
from .models import CallRequest, Outcome


class DialResult:
    def __init__(self, sid: str | None = None, outcome: Outcome | None = None, detail: str = ""):
        self.sid, self.outcome, self.detail = sid, outcome, detail


def build_call_params(req: CallRequest, to_e164: str, s: Settings) -> dict:
    base = s.public_base_url
    return {
        "To": to_e164,
        "From": s.from_number,
        "Url": f"{base}/twiml/{req.call_id}",
        "Method": "POST",
        "Timeout": str(req.ring_timeout_s or s.ring_timeout_s),                    # ring timeout -> no-answer
        "StatusCallback": f"{base}/status/{req.call_id}",
        "StatusCallbackMethod": "POST",
        "StatusCallbackEvent": ["initiated", "ringing", "answered", "completed"],
        # async AMD: media stream starts at answer, verdict arrives later
        "MachineDetection": "DetectMessageEnd",
        "AsyncAmd": "true",
        "AsyncAmdStatusCallback": f"{base}/amd/{req.call_id}",
        "AsyncAmdStatusCallbackMethod": "POST",
        "MachineDetectionTimeout": str(s.amd_timeout_s),
        "MachineDetectionSilenceTimeout": "5000",
        **({"Record": "true", "RecordingChannels": "dual"} if s.record_calls else {}),   # as call_agent /test-call
    }


async def place_call(req: CallRequest, s: Settings, client: httpx.AsyncClient | None = None) -> DialResult:
    lk = await lookup_number(req.to, s, client)
    if not lk.valid:
        return DialResult(outcome=Outcome.INVALID_NUMBER, detail=f"lookup:{lk.reason}")

    params = build_call_params(req, lk.e164 or req.to, s)
    url = f"https://api.twilio.com/2010-04-01/Accounts/{s.account_sid}/Calls.json"
    own = client is None
    client = client or httpx.AsyncClient(timeout=10)
    try:
        r = await client.post(url, data=params, auth=(s.account_sid, s.auth_token))
        if r.status_code >= 400:
            code = str(r.json().get("code", "")) if r.headers.get("content-type", "").startswith("application/json") else ""
            if code in {"21211", "21214", "21217", "13224"}:
                return DialResult(outcome=Outcome.INVALID_NUMBER, detail=f"twilio:{code}")
            return DialResult(outcome=Outcome.FAILED, detail=f"http_{r.status_code}:{code}")
        return DialResult(sid=r.json()["sid"])
    finally:
        if own:
            await client.aclose()


async def hangup(call_sid: str, s: Settings, client: httpx.AsyncClient | None = None) -> None:
    url = f"https://api.twilio.com/2010-04-01/Accounts/{s.account_sid}/Calls/{call_sid}.json"
    own = client is None
    client = client or httpx.AsyncClient(timeout=10)
    try:
        await client.post(url, data={"Status": "completed"}, auth=(s.account_sid, s.auth_token))
    finally:
        if own:
            await client.aclose()


async def send_digits(call_sid: str, digits: str, stream_url: str, s: Settings,
                      client: httpx.AsyncClient | None = None) -> None:
    """Press keys with Twilio's own DTMF. Redirects the live call to <Play digits/> then re-attaches the
    media stream (Twilio closes the old websocket and opens a new one to stream_url)."""
    import re
    d = re.sub(r"[^0-9#*wW]", "", digits)
    twiml = f'<Response><Play digits="w{d}"/><Connect><Stream url="{stream_url}"/></Connect></Response>'
    url = f"https://api.twilio.com/2010-04-01/Accounts/{s.account_sid}/Calls/{call_sid}.json"
    own = client is None
    client = client or httpx.AsyncClient(timeout=10)
    try:
        r = await client.post(url, data={"Twiml": twiml}, auth=(s.account_sid, s.auth_token))
        r.raise_for_status()
    finally:
        if own:
            await client.aclose()
