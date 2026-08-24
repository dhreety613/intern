import json
import re

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from pydantic import BaseModel, Field

from call_agent.app.call import CallSession


router = APIRouter(
    tags=["Call Agent"]
)


# ============================================================
# SESSION STORE
# ============================================================

sessions: Dict[str, CallSession] = {}


# ============================================================
# SCHEMAS
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


# ------------------------------------------------------------
# ElevenLabs / OpenAI-compatible request
# ------------------------------------------------------------

class ChatMessage(BaseModel):

    role: str
    content: str


class ChatCompletionRequest(BaseModel):

    model: str = "call-layer"

    user_id: Optional[str] = None

    stream: bool = True

    messages: List[ChatMessage] = Field(
        default_factory=list
    )

    elevenlabs_extra_body: Dict[str, Any] = Field(
        default_factory=dict
    )

# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
def call_agent_health():

    return {
        "service": "call-agent",
        "status": "healthy",
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

        calibration_dates=
            request.calibration_dates,
    )

    sessions[
        request.session_id
    ] = session

    opening = session.start()

    return {

        "session_id":
            request.session_id,

        "response":
            opening,

        "call_ended":
            False,
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

    decision = result["decision"]

    return {

        "session_id":
            request.session_id,

        "response":
            result["response"],

        "present_task":
            decision.get(
                "present_task"
            ),

        "needs_context":
            decision.get(
                "needs_context",
                False
            ),

        "topic_diversion":
            decision.get(
                "topic_diversion",
                False
            ),

        "random_conversation":
            decision.get(
                "random_conversation",
                False
            ),

        "current_stage":
            result["current_stage"],

        "progress":
            result["progress"],

        "call_ended":
            result["call_ended"],
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

        "session_id":
            session_id,

        "deleted":
            True,
    }


# ============================================================
# ELEVENLABS ADAPTER
# ============================================================
#
# THIS IS THE ONLY ENDPOINT ELEVENLABS USES.
#
# ONE call_id
#      ↓
# ONE CallSession
#      ↓
# start()
#      ↓
# process()
#      ↓
# process()
#      ↓
# ...
#      ↓
# call_ended
#
# ============================================================

@router.post("/v1/chat/completions")
async def elevenlabs_chat_completions(
    request: ChatCompletionRequest,
):

    messages = request.messages

    # --------------------------------------------------------
    # CALL ID
    # ElevenLabs user_id becomes our call/session ID.
    # --------------------------------------------------------

    call_id = request.user_id

    # --------------------------------------------------------
    # Extract customer information from system message
    # --------------------------------------------------------

    customer_name = ""
    email = ""
    phone_number = ""
    calibration_dates = []

    for message in messages:

        if message.role != "system":
            continue

        content = message.content

        name_match = re.search(
            r"Name:\s*(.+)",
            content,
            re.IGNORECASE,
        )

        email_match = re.search(
            r"Email:\s*(.+)",
            content,
            re.IGNORECASE,
        )

        phone_match = re.search(
            r"Phone:\s*(.+)",
            content,
            re.IGNORECASE,
        )

        calibration_match = re.search(
            r"Calibration date:\s*(.+)",
            content,
            re.IGNORECASE,
        )

        call_id_match = re.search(
            r"Call ID:\s*(.+)",
            content,
            re.IGNORECASE,
        )

        if name_match:
            customer_name = name_match.group(1).strip()

        if email_match:
            email = email_match.group(1).strip()

        if phone_match:
            phone_number = phone_match.group(1).strip()

        if calibration_match:
            calibration_dates = [
                calibration_match.group(1).strip()
            ]

        # Prefer explicit Call ID from the system prompt
        # if ElevenLabs supplied one.
        if call_id_match:
            call_id = call_id_match.group(1).strip()

    # --------------------------------------------------------
    # Validate call ID
    # --------------------------------------------------------

    if not call_id:

        raise HTTPException(
            status_code=400,
            detail="Missing call ID."
        )

    # --------------------------------------------------------
    # Find latest CUSTOMER message
    # --------------------------------------------------------

    user_message = ""

    for message in reversed(messages):

        if message.role == "user":

            user_message = message.content

            break

    # --------------------------------------------------------
    # FIND EXISTING SESSION
    # --------------------------------------------------------

    session = sessions.get(
        call_id
    )

    # ========================================================
    # FIRST REQUEST
    # ========================================================

    if session is None:

        session = CallSession(

            customer_name=customer_name,

            email=email,

            phone_number=phone_number,

            calibration_dates=calibration_dates,
        )

        sessions[
            call_id
        ] = session

        # First request starts the call.
        # Do NOT run LangGraph yet.

        response_text = session.start()

        call_ended = False

    # ========================================================
    # EXISTING CALL
    # ========================================================

    else:

        if not user_message:

            response_text = ""

            call_ended = session.state.get(
                "call_ended",
                False
            )

        else:

            result = session.process(
                user_message
            )

            response_text = result[
                "response"
            ]

            call_ended = result[
                "call_ended"
            ]

    # ========================================================
    # SSE RESPONSE
    # ========================================================

    async def event_stream():

        chunk = {

            "id":
                f"chatcmpl-{call_id}",

            "object":
                "chat.completion.chunk",

            "choices": [

                {

                    "index":
                        0,

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

        final_chunk = {

            "id":
                f"chatcmpl-{call_id}",

            "object":
                "chat.completion.chunk",

            "choices": [

                {

                    "index":
                        0,

                    "delta":
                        {},

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

        yield "data: [DONE]\n\n"

    return StreamingResponse(

        event_stream(),

        media_type="text/event-stream",

        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )