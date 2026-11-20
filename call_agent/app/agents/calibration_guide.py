"""Calibration form guidance: offer handling + step-by-step walkthrough.

The WORDING is produced by the LLM (natural, adapts to what the customer says), but the PROGRESSION is owned by
code, so the model can neither skip steps nor invent them:

* The steps are structured data (GUIDE_STEPS): what to tell the customer, the exact tokens that must be spoken
  verbatim (URL, menu names, phone number) and markers that identify a step.
* Code keeps the step counter. A step is only advanced by exactly one, and only when the model classifies the
  customer's reply as "done" for the CURRENT step. The model is only ever shown the current step and the next one.
* Every spoken line is validated: required verbatim tokens present, no marker of a FUTURE step present (that would be
  skipping ahead / hallucinating), length sane. A rejected line, or any LLM failure, falls back to the step's plain
  fallback text and NEVER advances.
* The customer's reply is classified by the LLM (yes/no/unclear, done/repeat/stuck/question/decline/side_request), so
  "Not really, and could you please use a different number" is a "no" plus a side request, not a "yes" because it
  contains the word "please". Side requests are handed back to the normal conversation graph.

This module is only used when call.py's CallSession has put the call into the "offered" or "walking" phase.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MODEL = "gemini-3.5-flash-lite"
TIMEOUT_SENTINEL = "[No response received - timeout]"

OFFER_QUESTION = "Would you like some guidance on how to fill out the calibration form?"
DECLINE_ACK = "No problem! Is there anything else I can help you with today?"
GUIDE_INTRO = "No problem, I can walk you through it."

# Each step: facts to convey (spoken form), tokens that MUST be said verbatim, markers that only belong to this step
# (seeing a later step's marker while speaking an earlier step means the model skipped ahead), and the plain
# fallback line used if the LLM fails or is rejected.
GUIDE_STEPS: List[Dict[str, Any]] = [
    {
        "instruction": "Ask them to open a web browser and go to tensitron dot com, and to tell you once they are there.",
        "must_say": ["tensitron dot com"],
        "markers": ["tensitron dot com"],
        "fallback": "Go ahead and open a browser, then head over to tensitron dot com. Let me know once you're there.",
    },
    {
        "instruction": "On the homepage, ask them to look toward the upper part of the page for the menu bar.",
        "must_say": [],
        "markers": ["menu bar"],
        "fallback": "Once the page is up, look toward the upper part of the homepage for the menu bar.",
    },
    {
        "instruction": "From the menu bar, ask them to click Calibration and Service, and to tell you when that page comes up.",
        "must_say": ["calibration and service"],
        "markers": ["calibration and service"],
        "fallback": "From there, click Calibration and Service. Tell me when that page comes up.",
    },
    {
        "instruction": "Tell them they should see the Calibration or Service Form option on that page, and to click it.",
        "must_say": ["service form"],
        "markers": ["service form"],
        "fallback": "You should see the Calibration or Service Form option there. Go ahead and click that.",
    },
    {
        "instruction": "When the form opens, ask them to read the important note, then click Start.",
        "must_say": ["start"],
        "markers": ["important note"],
        "fallback": "When the form opens, read the important note, then click start.",
    },
    {
        "instruction": "Next they will see the process and pricing information. Ask them to read through it, then click Next.",
        "must_say": ["next"],
        "markers": ["pricing"],
        "fallback": "After that, you'll see the process and pricing information. Take a moment to read through that. Then hit next.",
    },
    {
        "instruction": "Next they will see the Important Service Notes. Ask them to make sure to read it, then click Next.",
        "must_say": ["important service notes"],
        "markers": ["important service notes"],
        "fallback": "Next, you'll see the Important Service Notes. Make sure to read it, then hit next.",
    },
    {
        "instruction": "They should see an option to acknowledge the Service Notes. Ask them to click Yes to confirm.",
        "must_say": ["acknowledge"],
        "markers": ["acknowledge"],
        "fallback": "You should see an option to acknowledge the Service Notes. Go ahead and click yes to confirm.",
    },
    {
        "instruction": "Then they can fill out the form and submit it.",
        "must_say": ["submit"],
        "markers": ["submit it"],
        "fallback": "Then you can go ahead and fill it out and submit it.",
    },
]

# After the LAST step is confirmed. No goodbye here: the normal end-of-call flow says one real goodbye.
CLOSING = {
    "instruction": (
        "Tell them: once they submit it, Tensitron will have their information and will be expecting their meter, so "
        "please remember to send it in. If any questions come up later they can email service at tensitron dot com or "
        "call seven-two-zero, two-three-zero, one-six-one-six."
    ),
    "must_say": ["service at tensitron dot com", "seven-two-zero", "two-three-zero", "one-six-one-six"],
    "markers": ["expecting"],
    "fallback": (
        "After you submit it, we will have your information and we will be expecting your meter. Please remember to "
        "send it to us. If any questions come up later, you can email us at service at tensitron dot com or call us "
        "at seven-two-zero, two-three-zero, one-six-one-six."
    ),
}

_ALL = GUIDE_STEPS + [CLOSING]           # index len(GUIDE_STEPS) is the closing


# --------------------------------------------------------------------- LLM
def llm_json(prompt: str) -> Dict[str, Any]:
    """One JSON-returning Gemini call. Tests replace this function."""
    from call_agent.app.agents.llm_util import generate, make_client

    client = make_client(os.getenv("GEMINI_API_KEY"))
    resp = generate(client, model=MODEL, contents=prompt, config={"response_mime_type": "application/json"})
    text = (resp.text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _call_llm(prompt: str) -> Optional[Dict[str, Any]]:
    try:
        out = llm_json(prompt)
        return out if isinstance(out, dict) else None
    except Exception as exc:                       # never let a model/network error break or advance the walkthrough
        logger.error("guidance LLM call failed: %s", exc)
        return None


def _recent(conversation: Optional[List[Dict[str, str]]], n: int = 6) -> str:
    lines = []
    for m in (conversation or [])[-n:]:
        who = "Anne" if m.get("role") == "assistant" else "Customer"
        lines.append(f"{who}: {m.get('content', '')}")
    return "\n".join(lines) or "(none)"


# -------------------------------------------------------------- validation
def valid_line(text: Any, spoken_idx: int) -> bool:
    """A line about step `spoken_idx` (len(GUIDE_STEPS) = the closing) is acceptable only if it says the step's
    required tokens verbatim and mentions nothing that belongs to a LATER step (skipping ahead / invented steps)."""
    if not isinstance(text, str) or not text.strip() or len(text) > 450:
        return False
    low = text.lower()
    if any(tok.lower() not in low for tok in _ALL[spoken_idx]["must_say"]):
        return False
    for later in _ALL[spoken_idx + 1:]:
        if any(m.lower() in low for m in later["markers"]):
            return False
    return True


# ------------------------------------------------------------------ prompts
def _offer_prompt(message: str, conversation) -> str:
    return f"""You classify a customer's reply on a phone call.
Anne just asked: "{OFFER_QUESTION}"
Recent conversation:
{_recent(conversation)}
Customer's reply: "{message}"

Return JSON: {{"answer": "yes" | "no" | "unclear", "other_request": true | false}}
- answer = whether they want the calibration-form guidance. Judge ONLY the guidance question. Polite words such as
  "please" inside some other request do NOT mean yes.
- "no" for any refusal ("not really", "I don't need it", "no thanks", "I'm fine").
- "unclear" if they did not answer that question.
- other_request = true if the reply ALSO asks for or says something else that needs handling (changing their phone
  number / email / name, asking for a person, a different question), even alongside a yes/no."""


def _walk_prompt(step_idx: int, message: str, conversation) -> str:
    cur = GUIDE_STEPS[step_idx]
    n = len(GUIDE_STEPS)
    if step_idx + 1 < n:
        nxt = GUIDE_STEPS[step_idx + 1]
        nxt_block = f"NEXT STEP ({step_idx + 2} of {n}) - only if the customer finished the current step:\n{nxt['instruction']}\nSay verbatim: {nxt['must_say'] or 'nothing special'}"
    else:
        nxt_block = f"CLOSING - only if the customer finished the current (last) step:\n{CLOSING['instruction']}\nSay verbatim: {CLOSING['must_say']}"
    return f"""You are Anne from Tensitron, guiding a customer by phone through filling out the online calibration form, ONE step at a time.

CURRENT STEP ({step_idx + 1} of {n}):
{cur['instruction']}
Say verbatim (spoken form): {cur['must_say'] or 'nothing special'}

{nxt_block}

Recent conversation:
{_recent(conversation)}
Customer just said: "{message}"

First classify the customer's message as "intent":
- "done": they completed the CURRENT step or are ready to continue ("okay", "I'm there", "done", "yes it's open", "got it").
- "repeat": they ask you to repeat or didn't hear.
- "stuck": they cannot find it / are confused / it is not working.
- "question": they ask something about this step.
- "decline": they no longer want guidance ("no, I don't need guidance", "never mind", "I'll do it myself").
- "side_request": they ask for something unrelated to the form (change their number or email, talk to a person, pricing).
Also set "other_request": true if they ALSO ask for something unrelated alongside the intent.

Then write "response" (spoken aloud, 1-2 short sentences, plain words, no lists, no markdown):
- done: a brief acknowledgement, then ONLY the NEXT STEP / CLOSING above, using its verbatim phrases.
- repeat / stuck / question: ONLY the CURRENT step, said more simply or with a bit more help. If the question cannot be answered from the step text, say you don't have that detail and offer to repeat the step.
- decline / side_request: response is an empty string.
Never describe steps other than those given. Never invent buttons, menu names, prices or page layouts. Never mention how many steps remain.

Return JSON: {{"intent": "...", "other_request": false, "response": "..."}}"""


def _say_prompt(step_idx: int, conversation, lead: str) -> str:
    step = _ALL[step_idx]
    return f"""You are Anne from Tensitron on a phone call guiding a customer through the online calibration form.
{lead}
Say this step, in 1-2 short natural spoken sentences (no lists, no markdown):
{step['instruction']}
Say verbatim (spoken form): {step['must_say'] or 'nothing special'}
Recent conversation:
{_recent(conversation)}
Mention nothing beyond this step. Return JSON: {{"response": "..."}}"""


# ------------------------------------------------------------------- public
def _result(response: str, task: str, phase: str, step_idx: int, run_graph: bool = False) -> Dict[str, Any]:
    return {"response": response, "present_task": task, "phase": phase, "step_idx": step_idx,
            "call_ended": False, "run_graph": run_graph}


def _say_step(step_idx: int, conversation, lead: str = "", intro: bool = False) -> str:
    """LLM wording for one step; validated; plain fallback text if the model fails or is rejected."""
    out = _call_llm(_say_prompt(step_idx, conversation, lead))
    text = out.get("response") if out else None
    if valid_line(text, step_idx):
        return text.strip()
    return (GUIDE_INTRO + " " if intro else "") + _ALL[step_idx]["fallback"]


def start_walkthrough(conversation=None) -> Dict[str, Any]:
    """Turn result for starting the walkthrough from step 1."""
    text = _say_step(0, conversation, lead="Start with a brief friendly 'happy to walk you through it'.", intro=True)
    return _result(text, "WAIT", "walking", 0)


def handle_guidance_turn(phase: str, step_idx: int, user_message: str, conversation=None) -> Dict[str, Any]:
    """`phase` is the state going INTO this turn ("offered" or "walking"); the returned "phase"/"step_idx" are what
    call.py should persist. "run_graph" = also run this message through the normal conversation (phase is already
    updated to "done") because the customer asked for something else."""
    message = (user_message or "").strip()
    is_timeout = message == TIMEOUT_SENTINEL

    if phase == "offered":
        if is_timeout:                                                 # silence: re-ask, never assume an answer
            return _result(OFFER_QUESTION, "WAIT", "offered", 0)
        out = _call_llm(_offer_prompt(message, conversation)) or {}
        answer, other = str(out.get("answer", "unclear")).lower(), bool(out.get("other_request"))
        if answer == "yes":
            return start_walkthrough(conversation)
        if answer == "no":
            if other:
                return _result("", "CONTINUE", "done", 0, run_graph=True)
            return _result(DECLINE_ACK, "CONTINUE", "done", 0)
        if other:                                                      # they asked for something else instead of answering
            return _result("", "CONTINUE", "done", 0, run_graph=True)
        return _result(OFFER_QUESTION, "WAIT", "offered", 0)

    # phase == "walking"
    step_idx = max(0, min(step_idx, len(GUIDE_STEPS) - 1))
    repeat = _result(GUIDE_STEPS[step_idx]["fallback"], "WAIT", "walking", step_idx)
    if is_timeout:
        return _result(_say_step(step_idx, conversation, lead="They went quiet, so gently repeat the step."), "WAIT", "walking", step_idx)

    out = _call_llm(_walk_prompt(step_idx, message, conversation))
    if not out:
        return repeat                                                  # model failed: repeat the current step, never advance
    intent, other, text = str(out.get("intent", "")).lower(), bool(out.get("other_request")), out.get("response")

    if intent == "done":
        nxt = step_idx + 1
        if valid_line(text, nxt):
            if nxt < len(GUIDE_STEPS):
                return _result(text.strip(), "WAIT", "walking", nxt)
            return _result(text.strip(), "CONTINUE", "done", 0)
        # model's wording rejected (skipped ahead / dropped a verbatim token): use the plain text for exactly the next step
        if nxt < len(GUIDE_STEPS):
            return _result(GUIDE_STEPS[nxt]["fallback"], "WAIT", "walking", nxt)
        return _result(CLOSING["fallback"], "CONTINUE", "done", 0)
    if intent in ("repeat", "stuck", "question"):
        if valid_line(text, step_idx):
            return _result(text.strip(), "WAIT", "walking", step_idx)
        return repeat
    if intent == "decline":
        return _result("", "CONTINUE", "done", 0, run_graph=True) if other else _result(DECLINE_ACK, "CONTINUE", "done", 0)
    if intent == "side_request":
        return _result("", "CONTINUE", "done", 0, run_graph=True)
    return repeat                                                      # unrecognised intent: repeat the step, never advance
