import hashlib
import json
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
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
    #
    # ElevenLabs' custom-LLM webhook re-sends the FULL
    # conversation (system prompt + every prior turn) on
    # EVERY call, OpenAI-chat-completions style. It does NOT
    # forward dynamic_variables / conversation_initiation_
    # client_data as top-level body fields the way the n8n
    # call-start payload might suggest -- those are only used
    # by ElevenLabs to template {{vars}} into the system
    # prompt / first_message BEFORE the request ever reaches
    # us. The one documented pass-through channel is
    # `elevenlabs_extra_body`, which mirrors whatever you set
    # as `custom_llm_extra_body` when the call was started.
    # ========================================================

    messages = body.get(
        "messages",
        []
    )

    system_prompt = ""

    history: List[Dict[str, str]] = []

    for message in messages:

        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content", "")

        if not isinstance(content, str):
            content = str(content)

        if role == "system":
            # There can technically be more than one; keep them all.
            system_prompt += content + "\n"
            continue

        if role in ("user", "assistant"):
            history.append({
                "role": role,
                "content": content.strip(),
            })


    # ========================================================
    # EXTRACT DYNAMIC VARIABLES
    #
    # Priority order reflects how ElevenLabs actually delivers
    # data to a custom LLM server, most-likely-correct first:
    #
    # 1. elevenlabs_extra_body.dynamic_variables
    #    -> what arrives if you set `custom_llm_extra_body`
    #       (containing a `dynamic_variables` key) on the
    #       call-start payload sent to ElevenLabs.
    # 2. custom_llm_extra_body.dynamic_variables
    #    -> defensive: in case ElevenLabs ever echoes the
    #       field back under its outbound name instead.
    # 3. conversation_initiation_client_data.dynamic_variables
    # 4. dynamic_variables (bare, top-level)
    #
    # If ALL of these are empty, we fall back to scraping the
    # already-templated system prompt text for the values --
    # this keeps the agent working even before the n8n /
    # ElevenLabs agent config is updated to forward
    # custom_llm_extra_body (see reply for the config change
    # needed to make option 1 actually fire).
    # ========================================================

    def _dig(source: Any, key: str) -> Dict[str, Any]:
        if isinstance(source, dict):
            value = source.get(key, {})
            if isinstance(value, dict):
                return value
        return {}

    dynamic_variables: Dict[str, Any] = {}

    for candidate in (
        _dig(body.get("elevenlabs_extra_body"), "dynamic_variables"),
        _dig(body.get("custom_llm_extra_body"), "dynamic_variables"),
        _dig(body.get("conversation_initiation_client_data"), "dynamic_variables"),
        body.get("dynamic_variables"),
    ):
        if isinstance(candidate, dict) and candidate:
            dynamic_variables = candidate
            break

    used_system_prompt_fallback = False

    if not dynamic_variables and system_prompt:

        # Best-effort scrape of templated values out of the
        # system prompt, in case dynamic variables never made
        # it into the JSON body at all. Adjust these patterns
        # to match your actual system prompt wording.

        used_system_prompt_fallback = True

        def _scrape(pattern: str) -> str:
            match = re.search(pattern, system_prompt, flags=re.IGNORECASE)
            return match.group(1).strip() if match else ""

        dynamic_variables = {
            "customer_name": _scrape(r"customer(?:'s)? name(?: is)?[:\s]+([^\n,.]+)"),
            "email_on_file": _scrape(r"email(?: on file)?(?: is)?[:\s]+([^\s,]+@[^\s,]+)"),
            "number_on_file": _scrape(r"(?:phone|number)(?: on file)?(?: is)?[:\s]+(\+?[\d\s\-()]{7,})"),
            "next_calibration_date": _scrape(r"calibration date(?: is)?[:\s]+([^\n,.]+)"),
        }


    print(
        "DYNAMIC VARIABLES:",
        json.dumps(
            dynamic_variables,
            indent=2,
            default=str
        ),
        "(system-prompt fallback used)" if used_system_prompt_fallback else "",
    )


    # ========================================================
    # EXTRACT CUSTOMER DATA
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
    # 1. ElevenLabs call_id / conversation_id / user_id, IF
    #    ElevenLabs actually sends one (not guaranteed for
    #    custom LLM webhooks).
    # 2. deal_id from n8n dynamic variables.
    # 3. A stable hash of the (already-templated) system
    #    prompt. This is unique per call, since it embeds the
    #    customer's name/phone/etc, but IDENTICAL across every
    #    turn of the SAME call -- so it survives ElevenLabs not
    #    sending any explicit session identifier at all, and it
    #    never collapses two different real calls into one
    #    shared session the way a single hardcoded fallback did.
    # ========================================================

    call_id = (

        body.get("call_id")

        or body.get("conversation_id")

        or body.get("user_id")

        or dynamic_variables.get("deal_id")

    )

    if not call_id and system_prompt:

        call_id = "sysprompt-" + hashlib.sha1(
            system_prompt.encode("utf-8")
        ).hexdigest()[:16]

    if not call_id:

        # Truly nothing to key on (no messages, no system
        # prompt, no ids). Only happens on malformed requests.
        call_id = "elevenlabs-fallback-" + hashlib.sha1(
            json.dumps(body, default=str, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

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
    # LATEST USER MESSAGE (from the reconstructed history)
    # ========================================================

    user_message = ""

    for turn in reversed(history):
        if turn["role"] == "user":
            user_message = turn["content"]
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
    # FIRST REQUEST FOR THIS CALL
    #
    # IMPORTANT: ElevenLabs speaks its own configured
    # first_message via TTS directly -- it does NOT call your
    # custom LLM to generate the opening line. So by the time
    # your webhook is hit for a brand-new call_id, `history`
    # will usually already contain that spoken opening
    # (role="assistant") followed by the customer's actual
    # reply (role="user"). The old code ignored `history`
    # entirely here and always called session.start(), which
    # made Anne re-introduce herself instead of answering what
    # the customer just said. We now seed the session with
    # whatever ElevenLabs already said, then process the
    # customer's reply immediately if one is present.
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

        # Seed conversation history with everything ElevenLabs
        # already said/heard BEFORE this request, so Brain /
        # Decision have real context on the very first turn we
        # actually process. Exclude the trailing user message
        # (if any) -- session.process() below adds that itself.
        seed_turns = history[:-1] if (history and history[-1]["role"] == "user") else history

        for turn in seed_turns:
            session.add_message(turn["role"], turn["content"])

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

            # No customer turn yet at all -- this is a true
            # cold start where OUR LLM is responsible for the
            # opening line (e.g. ElevenLabs first_message left
            # blank). Generate it exactly as locally.

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