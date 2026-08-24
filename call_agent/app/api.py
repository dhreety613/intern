import json
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from call_agent.app.call import CallSession


router = APIRouter(
    tags=["Call Agent"]
)


# ============================================================
# IN-MEMORY CALL SESSIONS
# ============================================================

sessions: Dict[str, CallSession] = {}


# ============================================================
# REQUEST SCHEMAS
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

        calibration_dates=request.calibration_dates,
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

    return {

        "session_id":
            request.session_id,

        "response":
            result["response"],

        "present_task":
            result["decision"].get(
                "present_task"
            ),

        "needs_context":
            result["decision"].get(
                "needs_context",
                False
            ),

        "topic_diversion":
            result["decision"].get(
                "topic_diversion",
                False
            ),

        "random_conversation":
            result["decision"].get(
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
# DEBUG SESSION
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
# One call_id = one CallSession.
#
# First request:
#     create CallSession
#     call session.start()
#
# Later requests:
#     retrieve SAME CallSession
#     call session.process()
#
# ============================================================

@router.post(
    "/v1/chat/completions"
)
async def elevenlabs_chat_completions(
    request: Request,
):

    body = await request.json()

    messages = body.get(
        "messages",
        []
    )

    extra = body.get(
        "elevenlabs_extra_body",
        {}
    )

    call_id = extra.get(
        "call_id"
    )

    customer_name = extra.get(
        "customer_name",
        ""
    )

    email = extra.get(
        "email",
        ""
    )

    phone_number = extra.get(
        "phone_number",
        ""
    )

    calibration_dates = extra.get(
        "calibration_dates",
        []
    )

    # --------------------------------------------------------
    # Validate call ID
    # --------------------------------------------------------

    if not call_id:

        raise HTTPException(
            status_code=400,
            detail="Missing call_id."
        )

    # --------------------------------------------------------
    # Find the latest customer message
    # --------------------------------------------------------

    user_message = ""

    for message in reversed(messages):

        if message.get("role") == "user":

            content = message.get(
                "content",
                ""
            )

            if isinstance(
                content,
                str
            ):

                user_message = content

            break

    # --------------------------------------------------------
    # ONE SESSION PER CALL
    # --------------------------------------------------------

    session = sessions.get(
        call_id
    )

    # --------------------------------------------------------
    # FIRST REQUEST = START CALL
    # --------------------------------------------------------

    if session is None:

        session = CallSession(

            customer_name=customer_name,

            email=email,

            phone_number=phone_number,

            calibration_dates=(
                calibration_dates
                if isinstance(
                    calibration_dates,
                    list
                )
                else [str(calibration_dates)]
            ),
        )

        sessions[
            call_id
        ] = session

        response_text = session.start()

        call_ended = False

    # --------------------------------------------------------
    # EXISTING CALL = PROCESS MESSAGE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # OPENAI-COMPATIBLE SSE
    # --------------------------------------------------------

    async def event_stream():

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

        yield "data: [DONE]\n\n"

    return StreamingResponse(

        event_stream(),

        media_type=
            "text/event-stream",

        headers={

            "Cache-Control":
                "no-cache",

            "Connection":
                "keep-alive",
        },
    )