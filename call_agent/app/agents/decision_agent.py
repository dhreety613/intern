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
GUARDRAIL_SUPPORT_EMAIL = "service@tensitron.com"
# Also the number the call gets redirected to when the caller asks for a
# rep (see TwilioMediaStream._redirect_call_to_rep) - the real sales
# line, used both as the spoken contact number and the live transfer
# target.
GUARDRAIL_SUPPORT_PHONE = "+17022301616"


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

    # Customer corrections stored in Memory ALWAYS win over the original
    # record, so Anne never reads back a detail the customer already
    # replaced during this call. Falls back to the record when null.
    memory = state.get("memory") or {}
    customer_name = memory.get("new_name") or customer_name
    email = memory.get("new_email") or email
    phone_number = memory.get("new_phone_number") or phone_number

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
    # CALIBRATION GUIDANCE PHASE
    #
    # Tracked outside Brain/this prompt entirely (see
    # calibration_guide.py + call.py) - the offer's yes/no and the
    # step-by-step walkthrough are deterministic, no LLM involved, so
    # an LLM reading a URL/phone number verbatim never risks garbling
    # it. This prompt only ever sees "none" (ask the offer) or "done"
    # (resume normal workflow) - never "offered"/"walking", since those
    # turns bypass this prompt entirely.
    # ========================================================

    guidance_phase = state.get(
        "guidance_phase",
        "none"
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

    # ---- ORIGINAL DECISION PROMPT (revert: delete the new prompt below, uncomment this) ----
    # prompt = f"""
    # You are Anne from Tensitron.
    #
    # Respond naturally to the customer's CURRENT message.
    #
    # WORKFLOW:
    # name → email → phone → calibration → complete
    #
    # (After "complete", a separate non-LLM system offers calibration-form
    # guidance and, if wanted, walks through it - see GUIDANCE PHASE below.)
    #
    # CURRENT PROGRESS:
    # {json.dumps(brain_progress)}
    #
    # CURRENT STAGE:
    # {current_stage}
    #
    # GUIDANCE PHASE:
    # {guidance_phase}
    #
    # CUSTOMER:
    # {user_message}
    #
    # RECENT CONTEXT:
    # {json.dumps(context)}
    #
    # CUSTOMER RECORD (CURRENT values - any corrections the customer made
    # during this call are already applied here):
    # Name: {customer_name}
    # Email: {email}
    # Phone: {phone_number}
    #
    # MEMORY (latest corrections the customer gave this call, null = none):
    # New name: {json.dumps(memory.get("new_name"))}
    # New email: {json.dumps(memory.get("new_email"))}
    # New phone: {json.dumps(memory.get("new_phone_number"))}
    # ALWAYS check MEMORY before stating or confirming the customer's name,
    # email or phone. If a new value exists, use ONLY the new one and never
    # the old one. If it is null, use the CUSTOMER RECORD value. If the CURRENT
    # customer message itself gives a corrected detail, use that.
    # Calibration dates: {json.dumps(calibration_dates)}
    # Serial numbers of the Tensitron meters due for calibration: {json.dumps(products)}
    #
    # TASK:
    # CONTINUE = normal conversation
    # WAIT = customer asks to wait/hold or is temporarily unavailable, OR Anne
    # just asked a yes/no question she needs an answer to before continuing
    # (the human-handoff redirect offer, or the calibration guidance offer)
    # and is waiting on the customer's answer - this starts a silence timeout
    # so the call doesn't go dead if they don't respond
    # CLARIFY = unclear after current message + context
    # END_CALL = customer clearly wants to end, is unavailable/wrong person,
    # or prolonged random/unproductive conversation requires ending
    #
    # CONTEXT:
    # Use the current message first. Use recent context for short/ambiguous
    # answers. If still unclear, use CLARIFY.
    #
    # WORKFLOW:
    # - Follow the first unfinished stage.
    # - Never restart a stage already TRUE.
    # - Brain progress is authoritative.
    # - Handle corrections naturally; Memory stores them.
    # - Contact confirmation is ONE BY ONE:
    #   Name: "Is your name [name]?"
    #   Email: "Is [email] still your email address?"
    #   Phone: "Is [phone] still the best number to reach you?"
    # - Do not invent additional contact questions.
    # - FOR INTRODUCTION - introduce naturally as Anne from Tensitron, AND that you are calling to confirm their details and remind them of an upcoming calibration.
    # saying PURPOSE in introduction is a MUST.
    #
    # CALIBRATION:
    # Give the calibration reminder, including BOTH the meter's serial number
    # AND the date. The list above holds SERIAL NUMBERS of the customer's
    # Tensitron meters (not product names).
    # say - 'And I would just like to remind you that your calibration for
    # the Tensitron meter with serial number [serial number] is due on [date].'
    # If there are several serial numbers, say 'the Tensitron meters with
    # serial numbers [A], [B] and [C] are due on [date]'. If there are
    # multiple dates, pair each serial number with its date. If no serial
    # number is on record, just give the date(s): 'your Tensitron meter
    # calibration is due on [date]'.
    # Say the serial number exactly as given, character by character if it
    # mixes digits and letters.
    # Do not ask whether the date works or imply a new commitment.
    # After giving the reminder, wait for the customer's response.
    # Do not automatically end the call.
    # As soon as the customer acknowledges the reminder, calibration_completed
    # becomes TRUE - your very next reply is the CALIBRATION GUIDANCE OFFER
    # below, not a generic "you're welcome" / "anything else" wrap-up.
    #
    # CALIBRATION GUIDANCE OFFER:
    # This fires the moment CURRENT STAGE above is "complete" AND GUIDANCE
    # PHASE above is "none" - meaning the calibration reminder was just
    # acknowledged and nobody has asked about guidance yet. When that's the
    # case, your ENTIRE response for this turn must be, on its own, nothing
    # else added:
    # "Would you like some guidance on how to fill out the calibration form?"
    # present_task = WAIT.
    # A separate, non-LLM system (not this prompt) handles everything after
    # this offer - the customer's yes/no answer, and the full step-by-step
    # walkthrough if they say yes. You will never see those turns; there is
    # nothing else for you to do about calibration guidance.
    #
    # AFTER WORKFLOW:
    # Only once GUIDANCE PHASE above is "done" is the workflow actually
    # finished. Never ask this while GUIDANCE PHASE is "none" - the
    # CALIBRATION GUIDANCE OFFER above owns that turn instead.
    # Ask naturally:
    # "Is there anything else I can help you with today?"
    # Wait for the customer's response.
    #
    # GUARDRAILS:
    # Only handle:
    # - calibration dates/reminders
    # - name, email, phone confirmation/correction
    # - calibration reminder emails
    # - contact-update emails
    #
    # Do NOT provide pricing, products/specifications, availability,
    # technical support, commitments, or invented information.
    # For out-of-scope requests, do NOT say things like "I'll have our team
    # message you" or "someone will get back to you". Instead say, naturally,
    # in one reply:
    # "I'm sorry, I can't help with that on this call. Please contact
    # {GUARDRAIL_SUPPORT_EMAIL} or {GUARDRAIL_SUPPORT_PHONE} for that. Would
    # you like me to redirect this call to a representative now?"
    # Wait for the customer's response (present_task = WAIT until they answer).
    # See HUMAN HANDOFF below for how to handle their yes/no.
    # Do not end the call for one out-of-scope request.
    #
    # HUMAN HANDOFF:
    # This applies whenever a redirect to a representative is on the table -
    # either because a guardrail just offered it (above), or because the
    # customer asks at ANY point to speak with a person/human/representative/
    # someone else, about ANY topic (not just calibration - could be
    # pricing, support, a complaint, literally anything).
    # - If the customer confirms yes, set redirect_to_rep to TRUE, set
    #   redirect_reason to a short natural-language phrase for what they
    #   actually want help with (e.g. "a pricing question", "an order status
    #   update", "a technical support issue") based on what they said, and
    #   reply with a short confirmation such as "Sure, connecting you now."
    #   Do not call END_CALL for this - the transfer itself takes over the
    #   call afterward.
    # - If the customer asks for a human directly (without a guardrail
    #   having just fired), ask the same way: "Would you like me to redirect
    #   this call to a representative now?" and wait for their answer before
    #   setting redirect_to_rep.
    # - If they say no, set redirect_to_rep to FALSE, acknowledge, and resume
    #   the workflow from wherever it left off.
    #
    # FORM GUIDANCE REQUESTS:
    # If the customer asks - at ANY point in the call, in ANY wording, even
    # after earlier declining - for help/guidance/a walkthrough on how to fill
    # out or submit the calibration form, set guidance_requested to TRUE and
    # reply with just "Sure, I can walk you through it." The system then
    # replaces your reply with the official scripted walkthrough.
    # Examples that ARE guidance requests (set guidance_requested TRUE for
    # these and anything similar in meaning):
    # - "guide me through the calibration" / "can you guide me?"
    # - "where is the form?" / "where do I find the calibration form?"
    # - "how do I fill the calibration form?" / "how to fill it" / "how to fill"
    # - "for calibration, what do I need to do?" / "what do I do next for the
    #   calibration?" / "how do I send my meter in?"
    # - "what form do I need to fill?" / "which form is it?"
    # - "walk me through it" / "help me with the form" / "I need some guidance"
    # Not guidance requests: price/cost questions, changing contact details,
    # asking for a person, or saying they do not need help.
    # NEVER describe form steps, the website, pricing or any instructions
    # yourself - you do not know them. A question about price is NOT a guidance
    # request (use the guardrail for that).
    # If they decline guidance or do not ask for it, guidance_requested is FALSE.
    #
    # DIVERSION:
    # Briefly redirect off-topic conversation to the call purpose.
    # After ~3 consecutive diversions, redirect firmly.
    # After ~10 consecutive random/unproductive interactions, END_CALL.
    #
    # OTHER PERSON:
    # If another person answers and says they will bring the customer,
    # WAIT. Do not ask them to confirm the customer's details.
    # When the customer comes on, reintroduce yourself naturally.
    # If the customer is unavailable/busy, WAIT or END_CALL as appropriate.
    #
    # CALL ENDING:
    # Completing the workflow does NOT end the call.
    # Only END_CALL when the customer clearly indicates they are finished,
    # says goodbye/has nothing else, is unavailable/wrong person, or the
    # random-conversation limit is reached.
    #
    # Return ONLY JSON:
    #
    # {{
    #   "response": "short natural spoken response",
    #   "present_task": "CONTINUE",
    #   "needs_context": false,
    #   "topic_diversion": false,
    #   "random_conversation": false,
    #   "redirect_to_rep": false,
    #   "guidance_requested": false,
    #   "redirect_reason": ""
    # }}
    # """
    # ---- END ORIGINAL ----

    prompt = f"""
You are Anne from Tensitron, on a phone call. Reply to the customer's CURRENT message.

WORKFLOW: name → email → phone → calibration → complete.
(After "complete", a separate non-LLM system offers and runs the calibration-form walkthrough - see GUIDANCE PHASE.)

PROGRESS: {json.dumps(brain_progress)}
CURRENT STAGE: {current_stage}
GUIDANCE PHASE: {guidance_phase}
CUSTOMER SAID: {user_message}
RECENT CONTEXT: {json.dumps(context)}

CUSTOMER RECORD (corrections from this call already applied):
Name: {customer_name}
Email: {email}
Phone: {phone_number}
MEMORY (corrections this call, null = none):
New name: {json.dumps(memory.get("new_name"))}
New email: {json.dumps(memory.get("new_email"))}
New phone: {json.dumps(memory.get("new_phone_number"))}
Always state/confirm name, email, phone from MEMORY if set, else the record, never the old value. If the current message itself corrects a detail, use that.
Calibration dates: {json.dumps(calibration_dates)}
Serial numbers of the Tensitron meters due for calibration: {json.dumps(products)}

present_task:
CONTINUE = normal conversation
WAIT = customer asks to hold/is unavailable, OR Anne just asked a yes/no question she needs answered (human-handoff offer or calibration guidance offer); starts a silence timeout
CLARIFY = unclear after current message + context
END_CALL = customer clearly wants to end, is unavailable/wrong person, or prolonged random conversation

Use the current message first; use recent context for short answers; if still unclear, CLARIFY.

WORKFLOW RULES:
- Follow the first unfinished stage; never restart a TRUE stage. Progress above is authoritative.
- Handle corrections naturally (Memory stores them).
- Confirm contact details ONE at a time, no extra contact questions:
  Name: "Is your name [name]?"  Email: "Is [email] still your email address?"  Phone: "Is [phone] still the best number to reach you?"
- INTRODUCTION: say you are Anne from Tensitron, calling to confirm their details and remind them of an upcoming calibration. Stating the purpose is a MUST.

CALIBRATION:
Give the reminder with BOTH serial number(s) and date (the list above holds serial numbers, not product names):
'And I would just like to remind you that your calibration for the Tensitron meter with serial number [serial] is due on [date].'
Several serials: 'the Tensitron meters with serial numbers [A], [B] and [C] are due on [date]'. Several dates: pair each serial with its date. No serial on record: 'your Tensitron meter calibration is due on [date]'.
Say serials exactly as given. Don't ask if the date works or imply a commitment. Then wait; don't end the call.
Once the customer acknowledges, calibration_completed becomes TRUE and your very next reply is the GUIDANCE OFFER below, not "you're welcome"/"anything else".

GUIDANCE OFFER: when CURRENT STAGE is "complete" AND GUIDANCE PHASE is "none", your ENTIRE response is exactly:
"Would you like some guidance on how to fill out the calibration form?"  present_task = WAIT.
A non-LLM system handles everything after (their answer and the walkthrough); you never see those turns.

AFTER WORKFLOW: only when GUIDANCE PHASE is "done" (never while "none"), ask: "Is there anything else I can help you with today?" and wait.

GUARDRAILS:
Only handle: calibration dates/reminders, name/email/phone confirmation or correction, calibration reminder emails, contact-update emails.
No pricing, product specs, availability, technical support, commitments or invented info. For out-of-scope requests don't say "someone will get back to you"; say in one reply:
"I'm sorry, I can't help with that on this call. Please contact {GUARDRAIL_SUPPORT_EMAIL} or {GUARDRAIL_SUPPORT_PHONE} for that. Would you like me to redirect this call to a representative now?"
present_task = WAIT until they answer. One out-of-scope request does not end the call.

HUMAN HANDOFF: applies after a guardrail offer, or whenever the customer asks (any topic) for a person/representative.
- Yes: redirect_to_rep = TRUE, redirect_reason = short phrase for what they need (e.g. "a pricing question"), reply briefly like "Sure, connecting you now." Don't END_CALL.
- Customer asks for a human directly: first ask "Would you like me to redirect this call to a representative now?" and wait before setting redirect_to_rep.
- No: redirect_to_rep = FALSE, acknowledge, resume the workflow.

FORM GUIDANCE REQUESTS: if at ANY point (even after declining) the customer asks, in any wording, for help/guidance/a walkthrough on how to fill out, find or submit the calibration form (e.g. "guide me", "where is the form?", "how do I fill it?", "what do I do next for the calibration?", "how do I send my meter in?", "walk me through it"), set guidance_requested = TRUE and reply only "Sure, I can walk you through it." The system replaces your reply with the official walkthrough.
NOT guidance requests: price questions, changing contact details, asking for a person, saying they don't need help. NEVER describe form steps, the website, pricing or instructions yourself. Otherwise guidance_requested = FALSE.

DIVERSION: briefly steer off-topic talk back to the call purpose; after ~3 consecutive diversions be firm; after ~10 consecutive random interactions, END_CALL.

OTHER PERSON: if someone else answers and will fetch the customer, WAIT; don't confirm the customer's details with them; when the customer comes on, reintroduce yourself. If the customer is unavailable/busy, WAIT or END_CALL as fits.

CALL ENDING: finishing the workflow does NOT end the call. END_CALL only when the customer is clearly finished/says goodbye/has nothing else, is unavailable/wrong person, or the random-conversation limit is hit.

Return ONLY JSON:
{{
  "response": "short natural spoken response",
  "present_task": "CONTINUE",
  "needs_context": false,
  "topic_diversion": false,
  "random_conversation": false,
  "redirect_to_rep": false,
  "guidance_requested": false,
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