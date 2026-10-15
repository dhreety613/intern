import json
import re
from typing import Any, Dict
import os
from datetime import date

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

    # ---- ORIGINAL MEMORY PROMPT (revert: delete the new prompt below, uncomment this) ----
    # prompt = f"""
    # You are the memory extractor for a Tensitron call.
    #
    # Merge the current customer message into previous memory.
    #
    # Do not respond, decide, execute tasks, or invent facts.
    #
    # RULES:
    # - Preserve previous values unless explicitly changed.
    # - New/corrected name → new_name.
    # - New/corrected email → new_email.
    # - New/corrected phone → new_phone_number.
    # - Later corrections overwrite earlier new_* values.
    # - Store explicitly mentioned calibration dates/information.
    # - Calibration email request → email_calibration_confirmation=true.
    # - Contact-update email request → email_crm_update_confirmation=true.
    # - "Don't call again"/"stop calling" → do_not_call=true.
    # - Wrong number/person → wrong_number_so_avoid=true.
    # - Callback request → call_again_when (their words).
    # - If the customer says when we may call again AND it is a proper, resolvable
    #   date (an explicit date like "after October 20", or a clear relative one
    #   like "in two weeks" / "next Monday"), also set next_call_date to that date
    #   as YYYY-MM-DD, computed from TODAY below. Vague answers ("later", "some
    #   time", "when I'm free") → leave next_call_date null. Never guess a date.
    # - "Do not call"/"stop calling"/"remove me" → do_not_call=true (it stays false
    #   otherwise; never set it true from anything else).
    # - Human request → wants_human_call=true.
    # - Named human → wants_who.
    # - Human callback time → wants_human_call_when.
    # - Preserve established true flags unless explicitly reversed.
    # - all_tasks_ended is controlled by the orchestrator; preserve it.
    # - Never guess.
    # - Structured fields are authoritative.
    # -facts you feel are important and might be needed to be known by management later about th call store them in important_facts. NOT every line and NOT facts already being stored in the schema. Things special and edge case facts only.
    #
    # TODAY: {date.today().isoformat()}
    #
    # CURRENT:
    # {user_message}
    #
    # PREVIOUS:
    # {json.dumps(previous_memory)}
    #
    # CONTEXT:
    # {json.dumps(context)}
    #
    # Return ONLY JSON matching this schema:
    #
    # {{
    # "customer_name": null,
    # "email": null,
    # "phone_number": null,
    # "new_name": null,
    # "new_email": null,
    # "new_phone_number": null,
    # "calibration_dates": [],
    # "calibration_information": [],
    # "email_calibration_confirmation": false,
    # "email_crm_update_confirmation": false,
    # "do_not_call": false,
    # "wrong_number_so_avoid": false,
    # "call_again_when": null,
    # "next_call_date": null,
    # "wants_human_call": false,
    # "wants_who": null,
    # "wants_human_call_when": null,
    # "all_tasks_ended": false,
    # "important_facts": []
    # }}
    # """
    # ---- END ORIGINAL ----

    prompt = f"""
You are the memory extractor for a Tensitron call. Merge the current customer message into previous memory. Don't respond, decide or invent facts; never guess.

RULES:
- Keep previous values unless explicitly changed; keep established true flags unless explicitly reversed. Later corrections overwrite earlier new_* values.
- Corrected name/email/phone → new_name / new_email / new_phone_number.
- Store explicitly mentioned calibration dates/information.
- Calibration email request → email_calibration_confirmation=true. Contact-update email request → email_crm_update_confirmation=true.
- "Don't call again"/"stop calling"/"remove me" → do_not_call=true (never true otherwise).
- Wrong number/person → wrong_number_so_avoid=true.
- Callback request → call_again_when (their words). If they give a proper resolvable date ("after October 20", "in two weeks", "next Monday"), also set next_call_date as YYYY-MM-DD computed from TODAY; vague ("later", "when I'm free") → null.
- Human request → wants_human_call=true; named human → wants_who; callback time → wants_human_call_when.
- all_tasks_ended is controlled by the orchestrator; preserve it.
- important_facts: only special/edge-case facts management may need later; not every line and not anything already in the schema.

TODAY: {date.today().isoformat()}
CURRENT: {user_message}
PREVIOUS: {json.dumps(previous_memory)}
CONTEXT: {json.dumps(context)}

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
    "next_call_date": null,
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

    memory_dict = memory.model_dump()

    # Keep next_call_date only if it is a real future date.
    ncd = memory_dict.get("next_call_date")
    if ncd:
        try:
            if date.fromisoformat(str(ncd)[:10]) <= date.today():
                ncd = None
            else:
                ncd = str(ncd)[:10]
        except ValueError:
            ncd = None
    memory_dict["next_call_date"] = ncd

    return {
        "memory": memory_dict
    }