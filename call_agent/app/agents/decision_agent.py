import json
import re
from typing import Any, Dict
import os

from google import genai

from call_agent.app.graph.state import CallState
from call_agent.app.schemas.decision import DecisionOutput


#GEMINI_API_KEY = "AQ.Ab8RN6KrXmpFDNu8kFHZ5FZAK2u1qSDLLzib09I-Z4UsG8oX2A"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")
MODEL = "gemini-3.5-flash-lite"

MODEL = "gemini-3.5-flash-lite"

# --------------------------------------------------------------
# Placeholders — fill in with the real copy when available.
# --------------------------------------------------------------
CALIBRATION_FORM_GUIDE = "PLACEHOLDER: calibration form guidance goes here."
GUARDRAIL_SUPPORT_EMAIL = "sales@tensitron.com"
GUARDRAIL_SUPPORT_PHONE = "PLACEHOLDER_PHONE_NUMBER"


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

    return json.loads(
        text.strip()
    )


def decision_agent(
    state: CallState
) -> Dict[str, Any]:

    client = get_client()

    # ========================================================
    # QUEUE INFORMATION
    # ========================================================

    customer_name = state.get(
        "customer_name",
        ""
    )

    email = state.get(
        "email",
        ""
    )

    phone_number = state.get(
        "phone_number",
        ""
    )

    calibration_dates = state.get(
        "calibration_dates",
        []
    )

    products = state.get(
        "products",
        []
    )

    # ========================================================
    # CURRENT CUSTOMER MESSAGE
    # ========================================================

    user_message = state.get(
        "user_message",
        ""
    )

    # ========================================================
    # BRAIN STATE
    # ========================================================

    brain_progress = state.get(
        "progress",
        {
            "name_confirmed": False,
            "email_confirmed": False,
            "phone_confirmed": False,
            "calibration_completed": False,
        }
    )

    current_stage = state.get(
        "current_stage",
        "name_confirmation"
    )

    # ========================================================
    # CONTEXT
    # ========================================================

    context = state.get(
        "context",
        []
    )

    # ========================================================
    # OTHER CALL PROGRESS
    # ========================================================

    call_progress = state.get(
        "call_progress",
        {}
    )

    diversions = call_progress.get(
        "topic_diversion_count",
        0
    )

    random_count = call_progress.get(
        "random_conversation_count",
        0
    )

    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
You are Anne from KTC Instruments.

Respond naturally to the customer's CURRENT message.

WORKFLOW:
name → email → phone → calibration → calibration_guidance → complete

CURRENT PROGRESS:
{json.dumps(brain_progress)}

CURRENT STAGE:
{current_stage}

CUSTOMER:
{user_message}

RECENT CONTEXT:
{json.dumps(context)}

CUSTOMER RECORD:
Name: {customer_name}
Email: {email}
Phone: {phone_number}
Calibration dates: {json.dumps(calibration_dates)}
Products due for calibration: {json.dumps(products)}

TASK:
CONTINUE = normal conversation
WAIT = customer asks to wait/hold or is temporarily unavailable
CLARIFY = unclear after current message + context
END_CALL = customer clearly wants to end, is unavailable/wrong person,
or prolonged random/unproductive conversation requires ending

CONTEXT:
Use the current message first. Use recent context for short/ambiguous
answers. If still unclear, use CLARIFY.

WORKFLOW:
- Follow the first unfinished stage.
- Never restart a stage already TRUE.
- Brain progress is authoritative.
- Handle corrections naturally; Memory stores them.
- Contact confirmation is ONE BY ONE:
  Name: "Is your name [name]?"
  Email: "Is [email] still your email address?"
  Phone: "Is [phone] still the best number to reach you?"
- Do not invent additional contact questions.
- FOR INTRODUCTION - introduce naturally as Anne from KTCInstruments, AND that you are calling to confirm their details and remind them of an upcoming calibration.
saying PURPOSE in introduction is a MUST.

CALIBRATION:
Give the available calibration reminder, including BOTH the date(s) AND
the product(s) being calibrated.
say - 'And I would just like to remind you that your calibration for
[product] is due on [date] ...'
If there are multiple products/dates, mention each product together with
its corresponding date. If no products are on record, just give the
date(s) as before.
Do not ask whether the date works or imply a new commitment.
After giving the reminder, wait for the customer's response.
Do not automatically end the call.

CALIBRATION GUIDANCE:
Once calibration_completed is TRUE and calibration_guidance_offered is
not yet TRUE, ask naturally:
"Would you like some guidance on how to fill out the calibration form?"
Wait for the customer's response.
- If the customer says yes, read out the following guide to them:
  "{CALIBRATION_FORM_GUIDE}"
- If the customer says no or does not want it, do not read the guide;
  just acknowledge and move on.
Do not automatically end the call.

AFTER WORKFLOW:
Once calibration guidance has been offered and handled, the workflow is
complete.
Ask naturally:
"Is there anything else I can help you with today?"
Wait for the customer's response.

GUARDRAILS:
Only handle:
- calibration dates/reminders
- name, email, phone confirmation/correction
- calibration reminder emails
- contact-update emails

Do NOT provide pricing, products/specifications, availability,
technical support, commitments, or invented information.
For out-of-scope requests, do NOT say things like "I'll have our team
message you" or "someone will get back to you". Instead say, naturally,
something like:
"I'm sorry, I can't help with that on this call. You can email us at
{GUARDRAIL_SUPPORT_EMAIL} or call our team at {GUARDRAIL_SUPPORT_PHONE}
for that."
Do not end the call for one out-of-scope request.

DIVERSION:
Briefly redirect off-topic conversation to the call purpose.
After ~3 consecutive diversions, redirect firmly.
After ~10 consecutive random/unproductive interactions, END_CALL.

OTHER PERSON:
If another person answers and says they will bring the customer,
WAIT. Do not ask them to confirm the customer's details.
When the customer comes on, reintroduce yourself naturally.
If the customer is unavailable/busy, WAIT or END_CALL as appropriate.

CALL ENDING:
Completing the workflow does NOT end the call.
Only END_CALL when the customer clearly indicates they are finished,
says goodbye/has nothing else, is unavailable/wrong person, or the
random-conversation limit is reached.

Return ONLY JSON:

{{
  "response": "short natural spoken response",
  "present_task": "CONTINUE",
  "needs_context": false,
  "topic_diversion": false,
  "random_conversation": false
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

        decision = DecisionOutput(
            **raw
        )

    except Exception as exc:

        raise ValueError(
            "Decision Agent returned invalid output: "
            f"{response.text}"
        ) from exc

    return {
        "decision": decision.model_dump()
    }