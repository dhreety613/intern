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
CALIBRATION_FORM_GUIDE_STEPS = [
    "PLACEHOLDER STEP 1: Open the calibration form we emailed you.",
    "PLACEHOLDER STEP 2: Enter your equipment's serial number in the first field.",
    "PLACEHOLDER STEP 3: Enter today's date in the calibration date field.",
    "PLACEHOLDER STEP 4: Sign and date the bottom of the form.",
    "PLACEHOLDER STEP 5: Email the completed form back to us at the same address you received it from.",
]
GUARDRAIL_SUPPORT_EMAIL = "sales@tensitron.com"
# Also the number the call gets redirected to when the caller asks for a
# rep (see TwilioMediaStream._redirect_call_to_rep) - for now this is the
# same test number used elsewhere, until a real rep line is wired in.
GUARDRAIL_SUPPORT_PHONE = "+919040239072"


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
            "calibration_guidance_offered": False,
            "calibration_guidance_wanted": False,
            "calibration_guidance_step": 0,
        }
    )

    current_stage = state.get(
        "current_stage",
        "name_confirmation"
    )

    # ========================================================
    # CALIBRATION GUIDANCE STEP (see CALIBRATION_FORM_GUIDE_STEPS)
    # ========================================================

    guidance_step_index = brain_progress.get(
        "calibration_guidance_step",
        0
    )

    total_guidance_steps = len(CALIBRATION_FORM_GUIDE_STEPS)

    if 0 <= guidance_step_index < total_guidance_steps:
        current_guidance_step_text = CALIBRATION_FORM_GUIDE_STEPS[guidance_step_index]
    else:
        current_guidance_step_text = ""

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
WAIT = customer asks to wait/hold or is temporarily unavailable, OR Anne
just asked a yes/no question she needs an answer to before continuing
(the human-handoff redirect offer, or confirming a calibration guidance
step) and is waiting on the customer's answer - this starts a
silence timeout so the call doesn't go dead if they don't respond
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
As soon as the customer acknowledges the reminder, calibration_completed
becomes TRUE - your very next reply is the CALIBRATION GUIDANCE question
below, not a generic "you're welcome" / "anything else" wrap-up.

CALIBRATION GUIDANCE:
This fires the moment CURRENT STAGE above is "calibration_guidance" and
calibration_guidance_offered is FALSE. When that's the case, your ENTIRE
response for this turn MUST be the question below, asked naturally, on
its own - do not merge it with a thank-you, do not skip it, and do not
jump ahead to asking "is there anything else":
"Would you like some guidance on how to fill out the calibration form?"
Wait for the customer's response (present_task = WAIT until they answer).
- If the customer says no or does not want it, acknowledge and move
  straight to AFTER WORKFLOW. Do not read any steps.
- If the customer says yes, they now want guidance - move to the
  step-by-step guidance below, starting at step index 0.

STEP-BY-STEP GUIDANCE (only once the customer has said yes above):
Current step index (0-based): {guidance_step_index}
Total steps: {total_guidance_steps}
- If the current step index is less than the total steps, say ONLY the
  one step below out loud - do not read ahead, do not summarize more
  than this step, do not combine it with the next step:
  "{current_guidance_step_text}"
  Then set present_task to WAIT and wait for the customer to confirm
  they've done it or understood it (e.g. "ok", "got it", "done", "yes")
  before continuing. Do not move to the next step until they confirm
  this one. If they don't respond, the call's own silence-timeout will
  prompt again / end the call as usual - just keep waiting.
- Once the customer has confirmed the last step (current step index
  reaches the total), all guidance has been given - move straight to
  AFTER WORKFLOW.
Do not automatically end the call.

AFTER WORKFLOW:
Only once CURRENT STAGE above is "complete" - meaning calibration
guidance has been asked about, and if wanted, fully walked through -
is the workflow actually finished. Never ask this while CURRENT STAGE
is still "calibration_guidance".
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
in one reply:
"I'm sorry, I can't help with that on this call. Please contact
{GUARDRAIL_SUPPORT_EMAIL} or {GUARDRAIL_SUPPORT_PHONE} for that. Would
you like me to redirect this call to a representative now?"
Wait for the customer's response (present_task = WAIT until they answer).
See HUMAN HANDOFF below for how to handle their yes/no.
Do not end the call for one out-of-scope request.

HUMAN HANDOFF:
This applies whenever a redirect to a representative is on the table -
either because a guardrail just offered it (above), or because the
customer asks at ANY point to speak with a person/human/representative/
someone else, about ANY topic (not just calibration - could be
pricing, support, a complaint, literally anything).
- If the customer confirms yes, set redirect_to_rep to TRUE, set
  redirect_reason to a short natural-language phrase for what they
  actually want help with (e.g. "a pricing question", "an order status
  update", "a technical support issue") based on what they said, and
  reply with a short confirmation such as "Sure, connecting you now."
  Do not call END_CALL for this - the transfer itself takes over the
  call afterward.
- If the customer asks for a human directly (without a guardrail
  having just fired), ask the same way: "Would you like me to redirect
  this call to a representative now?" and wait for their answer before
  setting redirect_to_rep.
- If they say no, set redirect_to_rep to FALSE, acknowledge, and resume
  the workflow from wherever it left off.

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
  "random_conversation": false,
  "redirect_to_rep": false,
  "redirect_reason": ""
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