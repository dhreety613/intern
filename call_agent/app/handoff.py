"""Entry point for the telephony layer to hand a live call to Anne (the brain).

The telephony agent dials, works out what answered and handles IVRs / voicemail / hold / screening. Once it has
CONFIRMED A HUMAN it calls `run_brain(...)` with the Twilio media websocket it already owns. From then on this
module's handler owns the call: same websocket, same Twilio stream, no redirect and no audio gap.

The brain has no voicemail / IVR logic of its own any more: anything that is not a confirmed human never gets here.
"""
from __future__ import annotations

import logging
from typing import Any

from call_agent.app.media_stream import TwilioMediaStream

logger = logging.getLogger(__name__)

# keys the brain reads from Twilio's <Parameter> values (see TwilioMediaStream._handle_start)
_BRAIN_KEYS = ("customer_name", "email", "phone_number", "calibration_dates", "products", "person_id", "deal_id")


def _as_text(value: Any) -> str:
    """customParameters are strings; the brain parses lists from a comma-joined string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def build_start_event(*, call_sid: str, stream_sid: str, context: dict[str, Any] | None) -> dict[str, Any]:
    """A Twilio-shaped `start` event carrying the call context as customParameters."""
    ctx = context or {}
    params = {k: _as_text(ctx.get(k)) for k in _BRAIN_KEYS}
    return {
        "event": "start",
        "streamSid": stream_sid,
        "start": {"streamSid": stream_sid, "callSid": call_sid, "customParameters": params},
    }


async def run_brain(ws, *, call_id: str, call_sid: str, stream_sid: str, context: dict[str, Any] | None = None,
                    heard: list[str] | None = None) -> None:
    """Run the brain on an already-answered call until the call ends. `ws` is the live Twilio media websocket."""
    ctx = context or {}
    logger.info("HANDOFF -> brain call_id=%s sid=%s person=%s deal=%s heard=%s", call_id, call_sid,
                ctx.get("person_id", "-") or "-", ctx.get("deal_id", "-") or "-", heard or [])
    handler = TwilioMediaStream(
        websocket=ws,
        customer_name=_as_text(ctx.get("customer_name")),
        email=_as_text(ctx.get("email")),
        phone_number=_as_text(ctx.get("phone_number")),
        calibration_dates=_as_text(ctx.get("calibration_dates")),
        products=_as_text(ctx.get("products")),
    )
    await handler.handle(preamble=[{"event": "connected"},
                                   build_start_event(call_sid=call_sid, stream_sid=stream_sid, context=ctx)])
