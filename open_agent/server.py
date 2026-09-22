"""
open_agent/server.py

FastAPI router exposing the Pipecat-based voice agent:
  POST /twiml   - Twilio calls this when a call connects; returns TwiML
                  that opens a Media Stream WebSocket, carrying the
                  customer's data via <Parameter> tags.
  WS   /ws      - Twilio's Media Streams connection; handed off to
                  open_agent.bot.bot() to run the Pipecat pipeline.
  GET  /health  - basic liveness check.

Mounted into the main app (see main.py) with prefix "/open-agent", the
same pattern call_agent uses.
"""

from xml.sax.saxutils import escape

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

router = APIRouter(tags=["Open Agent"])


@router.get("/health")
def open_agent_health():
    return {
        "service": "open-agent",
        "status": "healthy",
        "engine": "pipecat",
        "build": "open-agent-3",
    }


@router.post("/twiml")
async def open_agent_twiml(request: Request):
    """Convert a Twilio call to a Pipecat Media Streams WebSocket.

    Same shape as call_agent's /twiml: params come from this endpoint's own
    query string (set when the outbound call is placed via Twilio's REST
    API), and are forwarded to the WebSocket both as a URL query string
    (fallback) and as <Parameter> tags (primary - see open_agent/bot.py,
    which reads them from call_data.body).
    """
    params = request.query_params

    customer_name = params.get("customer_name", "Customer")
    email = params.get("email", "")
    phone_number = params.get("phone_number", "")
    calibration_dates = params.get("calibration_dates", "")
    product = params.get("product", "")

    host = request.headers.get("host", request.url.netloc)
    prefix = request.url.path[: -len("/twiml")] if request.url.path.endswith("/twiml") else ""

    media_url = escape(f"wss://{host}{prefix}/ws")

    param_tags = "\n        ".join(
        f'<Parameter name="{escape(key)}" value="{escape(value)}" />'
        for key, value in (
            ("customer_name", customer_name),
            ("email", email),
            ("phone_number", phone_number),
            ("calibration_dates", calibration_dates),
            ("product", product),
        )
    )

    twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{media_url}">
        {param_tags}
        </Stream>
    </Connect>
</Response>'''

    return Response(content=twiml, media_type="text/xml")


@router.websocket("/ws")
async def open_agent_media_stream(websocket: WebSocket):
    """Twilio Media Streams WebSocket - handed off to the Pipecat bot."""
    from pipecat.runner.types import WebSocketRunnerArguments

    from open_agent.bot import bot

    await websocket.accept()

    try:
        runner_args = WebSocketRunnerArguments(websocket=websocket)
        await bot(runner_args)
    except WebSocketDisconnect:
        print("open_agent: WebSocket disconnected")
    except Exception as e:
        print(f"open_agent: WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
