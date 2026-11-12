"""FastAPI app: POST /calls to dial; Twilio hits /twiml, /status, /amd, and the /media websocket."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import uuid
from typing import Callable
from urllib.parse import urlsplit
from xml.sax.saxutils import escape

import httpx
from fastapi import Depends, FastAPI, Form, Header, HTTPException, Request, Response, WebSocket
from pydantic import BaseModel

from . import dialer
from .classifier import LlmFallback
from .config import Settings
from .models import AnsweredBy, CallRequest, HandoffToBrain, State
from .session import CallSession
from .speech import STT, TTS

log = logging.getLogger("telephony.server")


class DialBody(BaseModel):
    to: str
    extension: str | None = None
    context: dict = {}


def twilio_signature(auth_token: str, url: str, params: dict) -> str:
    data = url + "".join(f"{k}{params[k]}" for k in sorted(params))
    return base64.b64encode(hmac.new(auth_token.encode(), data.encode(), hashlib.sha1).digest()).decode()


def create_app(settings: Settings | None = None, *, stt_factory: Callable[[], STT], tts: TTS,
               llm: LlmFallback | None = None, http: httpx.AsyncClient | None = None,
               validate_signatures: bool | None = None, api_key: str | None = None) -> FastAPI:
    s = settings or Settings()
    app = FastAPI(title="telephony-agent")
    sessions: dict[str, CallSession] = {}
    results: dict[str, dict] = {}
    validate = (os.getenv("VALIDATE_SIGNATURES", "1") == "1") if validate_signatures is None else validate_signatures
    key = api_key if api_key is not None else os.getenv("AGENT_API_KEY", "")
    app.state.sessions, app.state.results, app.state.settings = sessions, results, s
    delivered: set[str] = set()
    bg: set[asyncio.Task] = set()

    async def _hangup(sid: str) -> None:
        await dialer.hangup(sid, s, http)

    def _dtmf_fn(call_id: str):
        async def send(sid: str, digits: str) -> None:
            ws = s.public_base_url.replace("https://", "wss://").replace("http://", "ws://") + f"/media/{call_id}"
            await dialer.send_digits(sid, digits, ws, s, http)
        return send

    def _record(call_id: str) -> None:
        sess = sessions.get(call_id)
        if sess:
            results[call_id] = {"call_id": call_id, "call_sid": sess.call_sid,
                                "outcome": sess.outcome.value if sess.outcome else None,
                                "retryable": sess.outcome.retryable if sess.outcome else None,
                                "state": sess.ctl.state.value, "log": sess.ctl.log, "signals": sess.hub.signal_log,
                                "heard": sess.ctl.heard,
                                "status_history": sess.status_history, "stats": sess.stats,
                                "stt": {"busy": getattr(sess.stt, "busy", None), "inflight": getattr(sess.stt, "_inflight", None),
                                        "type": type(sess.stt).__name__}}

    async def _deliver(call_id: str) -> None:
        """End-of-call outcome for the tags agent (n8n): same CALL_RESULT_WEBHOOK_URL / _SECRET as call_agent."""
        sess = sessions.get(call_id)
        if not sess or not sess.outcome or call_id in delivered:
            return
        delivered.add(call_id)
        _record(call_id)
        ctx = sess.req.context or {}
        result = {"source": "telephony_agent", "call_id": call_id, "call_sid": sess.call_sid,
                  "person_id": ctx.get("person_id", ""), "deal_id": ctx.get("deal_id", ""),
                  "phone_number": sess.req.to, "outcome": sess.outcome.value, "retryable": sess.outcome.retryable,
                  "human_confirmed": sess.outcome.value == "human_confirmed"}
        log.info("CALL RESULT: %s", json.dumps(result))
        url = os.getenv("CALL_RESULT_WEBHOOK_URL")
        if not url:
            return
        headers = {"X-Webhook-Secret": os.environ["CALL_RESULT_WEBHOOK_SECRET"]} if os.getenv("CALL_RESULT_WEBHOOK_SECRET") else {}
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.post(url, json=result, headers=headers)
                log.info("call result webhook -> HTTP %s", r.status_code)
        except Exception:
            log.exception("call result webhook failed")

    async def _handoff_stub(sess: CallSession, action: HandoffToBrain) -> None:
        """Stage-2 placeholder for the brain: the person is confirmed, so log it, hold the line briefly, hang up."""
        log.info("HANDOFF call=%s sid=%s heard=%s (brain not wired yet)", sess.req.call_id, sess.call_sid,
                 action.transcript_so_far)

        async def _later() -> None:
            await asyncio.sleep(s.handoff_stub_hold_s)
            if sess.call_sid:
                try:
                    await _hangup(sess.call_sid)
                except Exception:
                    log.exception("handoff-stub hangup failed")
            if sess.stt:
                await sess.stt.close()
        t = asyncio.create_task(_later())
        bg.add(t)
        t.add_done_callback(bg.discard)

    async def verify(request: Request, x_twilio_signature: str = Header(default="")) -> dict:
        form = dict((await request.form()).items())
        if validate:
            # Twilio signs the URL it called. public_base_url already carries any mount prefix
            # (e.g. /telephony-agent), and request.url.path may or may not include it depending on Starlette.
            path = request.url.path
            prefix = urlsplit(s.public_base_url).path
            if prefix and path.startswith(prefix + "/"):
                path = path[len(prefix):]
            url = s.public_base_url + path
            if not hmac.compare_digest(twilio_signature(s.auth_token, url, form), x_twilio_signature):
                raise HTTPException(403, "bad signature")
        return form

    @app.post("/calls")
    async def start_call(body: DialBody, x_api_key: str = Header(default="")):
        if key and not hmac.compare_digest(key, x_api_key):
            raise HTTPException(401)
        req = CallRequest(to=body.to, call_id=uuid.uuid4().hex[:12], extension=body.extension, context=body.context)
        sess = CallSession(req, s, stt_factory=stt_factory, tts=tts, hangup=_hangup, send_dtmf=_dtmf_fn(req.call_id), llm=llm,
                           on_handoff=_handoff_stub)
        sessions[req.call_id] = sess
        await sess.start()                       # detectors run from dial time (status/AMD arrive without audio)
        res = await dialer.place_call(req, s, http)
        if res.outcome:
            sess.ctl.finish_pre_answer(res.outcome)
            await sess.hub.stop()
            await _deliver(req.call_id)
            return {"call_id": req.call_id, "outcome": res.outcome.value, "detail": res.detail}
        sess.call_sid = res.sid
        return {"call_id": req.call_id, "call_sid": res.sid}

    @app.get("/calls/{call_id}")
    async def get_call(call_id: str, x_api_key: str = Header(default="")):
        if key and not hmac.compare_digest(key, x_api_key):
            raise HTTPException(401)
        _record(call_id)
        if call_id not in results:
            raise HTTPException(404)
        return results[call_id]

    @app.post("/twiml/{call_id}")
    async def twiml(call_id: str, form: dict = Depends(verify)):
        if call_id not in sessions:
            return Response("<Response><Hangup/></Response>", media_type="text/xml")
        ws = s.public_base_url.replace("https://", "wss://").replace("http://", "ws://") + f"/media/{call_id}"
        xml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Connect><Stream url="{escape(ws)}"/></Connect></Response>'
        return Response(xml, media_type="text/xml")

    @app.post("/status/{call_id}")
    async def status(call_id: str, form: dict = Depends(verify)):
        sess = sessions.get(call_id)
        if not sess:
            return {"ok": True}
        sess.status_history.append(form.get("CallStatus", ""))
        sess.call_sid = sess.call_sid or form.get("CallSid")
        sess.hub.feed_status(form)
        await sess.settle()
        _record(call_id)
        if sess.ctl.state is State.ENDED:
            await _deliver(call_id)
        return {"ok": True}

    @app.post("/amd/{call_id}")
    async def amd(call_id: str, form: dict = Depends(verify)):
        sess = sessions.get(call_id)
        if sess:
            raw = (form.get("AnsweredBy") or "unknown").lower()
            try:
                ab = AnsweredBy(raw)
            except ValueError:
                ab = AnsweredBy.UNKNOWN
            sess.hub.feed_amd(ab)
            await sess.settle()
            _record(call_id)
        return {"ok": True}

    @app.websocket("/media/{call_id}")
    async def media(ws: WebSocket, call_id: str):
        sess = sessions.get(call_id)
        if not sess:
            await ws.close(code=4404)
            return
        await ws.accept()
        await sess.run_media(ws)
        _record(call_id)
        if sess.ws is None and sess.outcome:        # stream really closed (not a DTMF reconnect)
            await _deliver(call_id)

    return app
