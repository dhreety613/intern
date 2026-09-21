import json
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from xml.sax.saxutils import escape

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from call_agent.app.call import CallSession
from call_agent.app.media_stream import TwilioMediaStream


router = APIRouter(
    tags=["Call Agent"]
)


# ============================================================
# SESSION STORE
# ============================================================

sessions: Dict[str, CallSession] = {}


# ============================================================
# STANDARD API SCHEMAS
# ============================================================

class StartCallRequest(BaseModel):

    session_id: str

    customer_name: str
    email: str
    phone_number: str

    calibration_dates: List[str] = Field(
        default_factory=list
    )


class CustomerMessageRequest(BaseModel):

    session_id: str
    message: str


# ============================================================
# ELEVENLABS / OPENAI COMPATIBLE SCHEMAS
# ============================================================

class ChatMessage(BaseModel):

    role: str
    content: Any


class ChatCompletionRequest(BaseModel):

    model: Optional[str] = "call-layer"

    messages: List[ChatMessage] = Field(
        default_factory=list
    )

    stream: bool = True

    temperature: Optional[float] = None

    max_tokens: Optional[int] = None

    user_id: Optional[str] = None

    # Optional so ElevenLabs can connect even if it doesn't
    # send extra body data.
    custom_llm_extra_body: Dict[str, Any] = Field(
        default_factory=dict
    )

    # Support dynamic variables if ElevenLabs forwards them
    dynamic_variables: Dict[str, Any] = Field(
        default_factory=dict
    )

    # Support the full conversation initiation structure
    conversation_initiation_client_data: Dict[str, Any] = Field(
        default_factory=dict
    )

# Add to call_agent/app/api.py

@router.get("/health/stt")
async def health_stt():
    """Actually call the configured STT engine so credentials/encoding are verified."""
    import os
    import traceback
    engine_name = os.getenv("STT_ENGINE", "elevenlabs")
    try:
        from call_agent.app.speech_services import get_stt_engine
        stt = get_stt_engine(engine_name)
        # 0.5s of mu-law silence - won't transcribe, but exercises auth + config.
        silence = b"\xff" * 4000

        async def _gen():
            yield silence

        got = []
        async for text in stt.transcribe_stream(_gen()):
            got.append(text)
        return {"status": "OK", "component": type(stt).__name__, "transcripts": got}
    except Exception as e:
        return {
            "status": "FAILED",
            "component": engine_name,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-1500:],
        }


@router.get("/health/tts")
async def health_tts(text: str = "Hello, this is a test."):
    """Actually synthesize a phrase and report what the TTS path produced."""
    import traceback
    try:
        from call_agent.app.speech_services import get_tts_engine
        tts = get_tts_engine("elevenlabs")
        total = 0
        chunks = 0
        async for chunk in tts.text_to_speech_stream(text):
            total += len(chunk)
            chunks += 1
        return {
            "status": "OK" if total > 0 else "NO_AUDIO",
            "component": "ElevenLabsTTS",
            "voice_id": getattr(tts, "voice_id", None),
            "model_id": getattr(tts, "model_id", None),
            "output_format": getattr(tts, "output_format", None),
            "bytes": total,
            "chunks": chunks,
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "component": "ElevenLabsTTS",
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-1500:],
        }


@router.get("/health/agent")
def health_agent():
    """Test CallSession/Agent initialization"""
    try:
        from call_agent.app.call import CallSession
        session = CallSession(
            customer_name="Test",
            email="test@test.com",
            phone_number="+1234567890",
            calibration_dates=["September 15, 2026"]
        )
        greeting = session.start()
        return {"status": "OK", "component": "CallSession", "greeting": greeting}
    except Exception as e:
        return {"status": "FAILED", "component": "CallSession", "error": str(e)}
@router.post("/twiml")
async def twilio_twiml(request: Request):
    """Convert Twilio call to a Media Streams WebSocket.

    Twilio validates the response as strict XML, so every `&` in the
    stream URL must be XML-escaped (`&amp;`) and each query value must be
    URL-encoded. The host and route prefix are taken from the inbound
    request so this follows whatever URL Render serves.
    """

    params = request.query_params

    customer_name = params.get("customer_name", "Customer")
    email = params.get("email", "")
    phone_number = params.get("phone_number", "")
    calibration_dates = params.get("calibration_dates", "")

    host = request.headers.get("host", request.url.netloc)

    # Mirror whatever prefix this router is mounted under (e.g. /call-agent).
    prefix = request.url.path[: -len("/twiml")] if request.url.path.endswith("/twiml") else ""

    query = "&".join(
        f"{key}={quote(value, safe='')}"
        for key, value in (
            ("customer_name", customer_name),
            ("email", email),
            ("phone_number", phone_number),
            ("calibration_dates", calibration_dates),
        )
    )

    media_url = escape(f"wss://{host}{prefix}/media?{query}")

    # Also pass data via <Parameter>, delivered in the WebSocket "start"
    # event's customParameters. This is the primary channel media_stream.py
    # reads from - some proxies don't reliably preserve query strings on a
    # WSS upgrade request, so the URL query string above is only a fallback.
    param_tags = "\n        ".join(
        f'<Parameter name="{escape(key)}" value="{escape(value)}" />'
        for key, value in (
            ("customer_name", customer_name),
            ("email", email),
            ("phone_number", phone_number),
            ("calibration_dates", calibration_dates),
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

# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
def call_agent_health():

    return {
        "service": "call-agent",
        "status": "healthy",
        "build": "voice-fix-8",
    }


# ============================================================
# START CALL
# ============================================================

@router.post("/start")
def start_call(
    request: StartCallRequest
):

    if request.session_id in sessions:

        raise HTTPException(
            status_code=409,
            detail="Session already exists."
        )

    session = CallSession(

        customer_name=request.customer_name,

        email=request.email,

        phone_number=request.phone_number,

        calibration_dates=request.calibration_dates,
    )

    sessions[
        request.session_id
    ] = session

    opening = session.start()

    return {

        "session_id": request.session_id,

        "response": opening,

        "call_ended": False,
    }


# ============================================================
# CUSTOMER MESSAGE
# ============================================================

@router.post("/message")
def customer_message(
    request: CustomerMessageRequest
):

    session = sessions.get(
        request.session_id
    )

    if session is None:

        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    if session.state.get(
        "call_ended",
        False
    ):

        raise HTTPException(
            status_code=400,
            detail="Call has already ended."
        )

    if not request.message.strip():

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    result = session.process(
        request.message
    )

    decision = result.get(
        "decision",
        {}
    )

    return {

        "session_id": request.session_id,

        "response": result.get(
            "response",
            ""
        ),

        "present_task": decision.get(
            "present_task"
        ),

        "needs_context": decision.get(
            "needs_context",
            False
        ),

        "topic_diversion": decision.get(
            "topic_diversion",
            False
        ),

        "random_conversation": decision.get(
            "random_conversation",
            False
        ),

        "current_stage": result.get(
            "current_stage",
            ""
        ),

        "progress": result.get(
            "progress",
            {}
        ),

        "call_ended": result.get(
            "call_ended",
            False
        ),
    }


# ============================================================
# GET SESSION
# ============================================================

@router.get("/session/{session_id}")
def get_session(
    session_id: str
):

    session = sessions.get(
        session_id
    )

    if session is None:

        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    return session.state


# ============================================================
# DELETE SESSION
# ============================================================

@router.delete("/session/{session_id}")
def delete_session(
    session_id: str
):

    if session_id not in sessions:

        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    del sessions[
        session_id
    ]

    return {

        "session_id": session_id,

        "deleted": True,
    }


# ============================================================
# ELEVENLABS CUSTOM LLM ADAPTER
# ============================================================

@router.post("/v1/chat/completions")
async def elevenlabs_chat_completions(
    request: Request,
):

    # ========================================================
    # READ RAW ELEVENLABS REQUEST
    # ========================================================

    body = await request.json()

    print("\n================================================")
    print("ELEVENLABS REQUEST RECEIVED")
    print(json.dumps(body, indent=2, default=str))
    print("================================================\n")


    # ========================================================
    # EXTRACT MESSAGES
    # ========================================================

    messages = body.get(
        "messages",
        []
    )


    # ========================================================
    # EXTRACT DYNAMIC VARIABLES
    #
    # Supports multiple possible forwarding structures.
    # ========================================================

    dynamic_variables = {}


    # --------------------------------------------------------
    # Option 1:
    # conversation_initiation_client_data.dynamic_variables
    # --------------------------------------------------------

    initiation_data = body.get(
        "conversation_initiation_client_data",
        {}
    )

    if isinstance(
        initiation_data,
        dict
    ):

        dynamic_variables = initiation_data.get(
            "dynamic_variables",
            {}
        )


    # --------------------------------------------------------
    # Option 2:
    # dynamic_variables directly in body
    # --------------------------------------------------------

    if not dynamic_variables:

        dynamic_variables = body.get(
            "dynamic_variables",
            {}
        )


    # --------------------------------------------------------
    # Option 3:
    # custom_llm_extra_body
    # --------------------------------------------------------

    if not dynamic_variables:

        extra_body = body.get(
            "custom_llm_extra_body",
            {}
        )

        if isinstance(
            extra_body,
            dict
        ):

            dynamic_variables = extra_body.get(
                "dynamic_variables",
                extra_body
            )


    # Make sure it is always a dictionary

    if not isinstance(
        dynamic_variables,
        dict
    ):

        dynamic_variables = {}


    print(
        "DYNAMIC VARIABLES:",
        json.dumps(
            dynamic_variables,
            indent=2,
            default=str
        )
    )


    # ========================================================
    # EXTRACT CUSTOMER DATA FROM YOUR N8N PAYLOAD
    # ========================================================

    customer_name = dynamic_variables.get(
        "customer_name",
        ""
    )

    email = dynamic_variables.get(
        "email_on_file",
        ""
    )

    phone_number = dynamic_variables.get(
        "number_on_file",
        ""
    )

    next_calibration_date = dynamic_variables.get(
        "next_calibration_date",
        ""
    )


    # ========================================================
    # NORMALIZE CALIBRATION DATE
    # ========================================================

    if isinstance(
        next_calibration_date,
        list
    ):

        calibration_dates = next_calibration_date

    elif next_calibration_date:

        calibration_dates = [
            str(next_calibration_date)
        ]

    else:

        calibration_dates = []


    # ========================================================
    # CREATE A SESSION ID
    #
    # Priority:
    # 1. ElevenLabs call_id
    # 2. conversation_id
    # 3. user_id
    # 4. deal_id from n8n
    # ========================================================

    call_id = (

        body.get("call_id")

        or body.get("conversation_id")

        or body.get("user_id")

        or dynamic_variables.get("deal_id")

    )


    # ========================================================
    # LAST-RESORT FALLBACK
    #
    # Only prevents the request from crashing while debugging.
    # ========================================================

    if not call_id:

        call_id = "elevenlabs-test-session"


    call_id = str(
        call_id
    )


    print(
        "CALL / SESSION ID:",
        call_id
    )


    print(
        "CUSTOMER:",
        customer_name
    )

    print(
        "EMAIL:",
        email
    )

    print(
        "PHONE:",
        phone_number
    )

    print(
        "CALIBRATION DATES:",
        calibration_dates
    )


    # ========================================================
    # FIND LATEST USER MESSAGE
    # ========================================================

    user_message = ""

    for message in reversed(messages):

        if not isinstance(
            message,
            dict
        ):

            continue

        if message.get("role") == "user":

            content = message.get(
                "content",
                ""
            )

            if isinstance(
                content,
                str
            ):

                user_message = content.strip()

            else:

                user_message = str(
                    content
                ).strip()

            break


    print(
        "USER MESSAGE:",
        user_message
    )


    # ========================================================
    # FIND OR CREATE CALL SESSION
    # ========================================================

    session = sessions.get(
        call_id
    )


    # ========================================================
    # FIRST REQUEST
    # ========================================================

    if session is None:

        print(
            f"CREATING NEW SESSION: {call_id}"
        )

        session = CallSession(

            customer_name=customer_name,

            email=email,

            phone_number=phone_number,

            calibration_dates=calibration_dates,
        )

        sessions[
            call_id
        ] = session


        # Generate Anne's opening statement

        response_text = session.start()

        call_ended = False


    # ========================================================
    # EXISTING SESSION
    # ========================================================

    else:

        print(
            f"USING EXISTING SESSION: {call_id}"
        )


        # Only run LangGraph when the customer actually speaks

        if user_message:

            result = session.process(
                user_message
            )

            response_text = result.get(
                "response",
                ""
            )

            call_ended = result.get(
                "call_ended",
                False
            )

        else:

            response_text = ""

            call_ended = session.state.get(
                "call_ended",
                False
            )


    print(
        "AGENT RESPONSE:",
        response_text
    )


    # ========================================================
    # SSE RESPONSE REQUIRED BY ELEVENLABS
    # ========================================================

    async def event_stream():

        # ----------------------------------------------------
        # Content chunk
        # ----------------------------------------------------

        if response_text:

            chunk = {

                "id":
                    f"chatcmpl-{call_id}",

                "object":
                    "chat.completion.chunk",

                "choices": [

                    {

                        "index": 0,

                        "delta": {

                            "role":
                                "assistant",

                            "content":
                                response_text,
                        },

                        "finish_reason":
                            None,
                    }
                ],
            }


            yield (

                "data: "

                + json.dumps(chunk)

                + "\n\n"
            )


        # ----------------------------------------------------
        # Final chunk
        # ----------------------------------------------------

        final_chunk = {

            "id":
                f"chatcmpl-{call_id}",

            "object":
                "chat.completion.chunk",

            "choices": [

                {

                    "index": 0,

                    "delta": {},

                    "finish_reason":
                        "stop",
                }
            ],
        }


        yield (

            "data: "

            + json.dumps(final_chunk)

            + "\n\n"
        )


        # ----------------------------------------------------
        # Required stream termination
        # ----------------------------------------------------

        yield "data: [DONE]\n\n"


    return StreamingResponse(

        event_stream(),

        media_type="text/event-stream",

        headers={

            "Cache-Control":
                "no-cache",

            "Connection":
                "keep-alive",
        },
    )


# ============================================================
# TWILIO MEDIA STREAMS (WEBSOCKET - VOICE AGENT)
# ============================================================

@router.websocket("/media")
async def twilio_media_stream(
    websocket: WebSocket,
    customer_name: str = "Customer",
    email: str = "",
    phone_number: str = "",
    calibration_dates: str = "",
    deal_id: str = "",
):
    """
    WebSocket endpoint for Twilio Media Streams.
    
    Handles real-time bidirectional audio:
    - Receive caller audio (8kHz LINEAR16)
    - STT → Google Cloud Speech-to-Text
    - Agent → Your LangGraph CallSession
    - TTS → ElevenLabs API
    - Stream audio back to caller
    
    Connect from Twilio:
    wss://your-server/media?customer_name=Jane&email=john@company.com&phone_number=+1234567890
    """

    await websocket.accept()

    try:
        handler = TwilioMediaStream(
            websocket=websocket,
            customer_name=customer_name or "Customer",
            email=email or "",
            phone_number=phone_number or "",
            calibration_dates=calibration_dates or "",
        )

        await handler.handle()

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {deal_id or 'unknown'}")

    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass