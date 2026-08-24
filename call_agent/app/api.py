from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from call_agent.app.call import CallSession

from call_agent.app.elevenlabs_adapter import (
    ChatCompletionRequest,
    elevenlabs_chat_completion,
)




router = APIRouter(
    tags=["Call Agent"]
)

@router.post("/v1/chat/completions")
async def elevenlabs_llm(
    request: ChatCompletionRequest,
):
    return await elevenlabs_chat_completion(
        request
    )


# ============================================================
# REQUEST / RESPONSE SCHEMAS
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
# TEMPORARY IN-MEMORY SESSION STORE
# ============================================================
#
# This is fine for the first Render deployment/test.
# Later, for production/concurrent calls, we will move state
# to a persistent store keyed by ElevenLabs conversation ID.
#

sessions: Dict[str, CallSession] = {}


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

    opening = (
        "Hi, this is Anne from KTC Instruments. "
        f"Am I speaking with {request.customer_name}?"
    )

    session.add_message(
        "assistant",
        opening
    )

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
# DEBUG / TEST STATE
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