import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from call_agent.app.call import CallSession


# ------------------------------------------------------------
# ElevenLabs / OpenAI-compatible request models
# ------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]

    stream: Optional[bool] = True

    # ElevenLabs may provide this.
    user_id: Optional[str] = None

    # Allows us to receive per-call information later
    # through ElevenLabs extra body / dynamic configuration.
    elevenlabs_extra_body: Optional[Dict[str, Any]] = None


# ------------------------------------------------------------
# Temporary in-memory sessions
# ------------------------------------------------------------

sessions: Dict[str, CallSession] = {}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def get_last_user_message(
    messages: List[ChatMessage],
) -> str:

    for message in reversed(messages):

        if message.role == "user":

            if isinstance(
                message.content,
                str
            ):
                return message.content

            return str(
                message.content
            )

    return ""


def get_call_id(
    request: ChatCompletionRequest,
) -> str:

    if request.user_id:
        return request.user_id

    if request.elevenlabs_extra_body:

        call_id = request.elevenlabs_extra_body.get(
            "call_id"
        )

        if call_id:
            return str(call_id)

    return "elevenlabs-default"


def sse_response(
    text: str,
    model: str,
):

    chunk = {
        "id": "call-layer-response",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": None,
            }
        ],
    }

    final_chunk = {
        "id": "call-layer-response",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }

    async def stream():

        yield (
            f"data: {json.dumps(chunk)}\n\n"
        )

        yield (
            f"data: {json.dumps(final_chunk)}\n\n"
        )

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
    )


# ------------------------------------------------------------
# Main adapter
# ------------------------------------------------------------

async def elevenlabs_chat_completion(
    request: ChatCompletionRequest,
):

    call_id = get_call_id(
        request
    )

    user_message = get_last_user_message(
        request.messages
    )

    if not user_message:
        return sse_response(
            "I'm sorry, I didn't catch that.",
            request.model,
        )

    # --------------------------------------------------------
    # Create session if this is the first turn
    # --------------------------------------------------------

    if call_id not in sessions:

        extra = (
            request.elevenlabs_extra_body
            or {}
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

        session = CallSession(
            customer_name=customer_name,
            email=email,
            phone_number=phone_number,
            calibration_dates=calibration_dates,
        )

        sessions[
            call_id
        ] = session

    session = sessions[
        call_id
    ]

    # --------------------------------------------------------
    # Run your existing Call Layer
    # --------------------------------------------------------

    result = session.process(
        user_message
    )

    response_text = result[
        "response"
    ]

    # --------------------------------------------------------
    # Return OpenAI-compatible SSE
    # --------------------------------------------------------

    return sse_response(
        response_text,
        request.model,
    )