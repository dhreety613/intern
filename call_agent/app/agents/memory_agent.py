import json
import re
from typing import Any, Dict
import os

from google import genai

from call_agent.app.graph.state import CallState
from call_agent.app.schemas.memory import CallMemory


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")
MODEL = "gemini-3.5-flash-lite"


def get_client():
    return genai.Client(
        api_key=GEMINI_API_KEY
    )


def extract_json(text: str) -> Dict[str, Any]:

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```$",
        "",
        text
    )

    return json.loads(text.strip())


def memory_agent(
    state: CallState
) -> Dict[str, Any]:

    client = get_client()

    user_message = state.get(
        "user_message",
        ""
    )

    previous_memory = state.get(
        "memory",
        {}
    )

    context = state.get(
        "context",
        []
    )

    prompt = f"""
You are the memory extractor for a Tensitron call.

Merge the current customer message into previous memory.

Do not respond, decide, execute tasks, or invent facts.

RULES:
- Preserve previous values unless explicitly changed.
- New/corrected name → new_name.
- New/corrected email → new_email.
- New/corrected phone → new_phone_number.
- Later corrections overwrite earlier new_* values.
- Store explicitly mentioned calibration dates/information.
- Calibration email request → email_calibration_confirmation=true.
- Contact-update email request → email_crm_update_confirmation=true.
- "Don't call again"/"stop calling" → do_not_call=true.
- Wrong number/person → wrong_number_so_avoid=true.
- Callback request → call_again_when.
- Human request → wants_human_call=true.
- Named human → wants_who.
- Human callback time → wants_human_call_when.
- Preserve established true flags unless explicitly reversed.
- all_tasks_ended is controlled by the orchestrator; preserve it.
- Never guess.
- Structured fields are authoritative.
-facts you feel are important and might be needed to be known by management later about th call store them in important_facts. NOT every line and NOT facts already being stored in the schema. Things special and edge case facts only.

CURRENT:
{user_message}

PREVIOUS:
{json.dumps(previous_memory)}

CONTEXT:
{json.dumps(context)}

Return ONLY JSON matching this schema:

{{
    "customer_name": null,
    "email": null,
    "phone_number": null,
    "new_name": null,
    "new_email": null,
    "new_phone_number": null,
    "calibration_dates": [],
    "calibration_information": [],
    "email_calibration_confirmation": false,
    "email_crm_update_confirmation": false,
    "do_not_call": false,
    "wrong_number_so_avoid": false,
    "call_again_when": null,
    "wants_human_call": false,
    "wants_who": null,
    "wants_human_call_when": null,
    "all_tasks_ended": false,
    "important_facts": []
}}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    try:

        raw = extract_json(
            response.text
        )

        memory = CallMemory(
            **raw
        )

    except Exception as exc:

        raise ValueError(
            "Memory Agent returned invalid output: "
            f"{response.text}"
        ) from exc

    return {
        "memory": memory.model_dump()
    }