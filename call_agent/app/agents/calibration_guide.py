"""Deterministic (no-LLM) calibration form guidance: the offer
question's yes/no answer and the step-by-step walkthrough itself.

Why no Gemini here: the walkthrough reads a URL, a sequence of click
targets, and a phone number verbatim - letting an LLM paraphrase any of
that risks garbling exactly the details a customer needs to actually
follow along. The only "intelligence" this stage needs is a binary
repeat-vs-advance decision, which simple keyword matching handles fine.

This module only speaks for itself when call.py's CallSession has
already put the call into "offered" or "walking" phase (see
CallSession.process) - it never runs through call_graph/decision_agent.
"""

from typing import Any, Dict

TIMEOUT_SENTINEL = "[No response received - timeout]"

OFFER_QUESTION = "Would you like some guidance on how to fill out the calibration form?"

GUIDE_INTRO = "No problem, I can walk you through it."

GUIDE_STEPS = [
    "Go ahead and open a browser, then head over to tensitron dot com. Let me know once you're there.",
    "Once the page is up, look toward the upper part of the homepage for the menu bar.",
    "From there, click Calibration and Service. Tell me when that page comes up.",
    "You should see the Calibration or Service Form option there. Go ahead and click that.",
    "When the form opens, read the important note, then click start.",
    "After that, you'll see the process and pricing information. Take a moment to read "
    "through that. Then hit next.",
    "Next, you'll see the Important Service Notes. Make sure to read it, then hit next.",
    "You should see an option to acknowledge the Service Notes. Go ahead and click yes to confirm.",
    "Then you can go ahead and fill it out and submit it.",
]

# Trimmed vs. the original scripted line: the goodbye ("thank you, have
# a great day") is left out on purpose - the normal end-of-call flow
# says a real goodbye right after this, once via "anything else?" and
# END_CALL, not twice.
GUIDE_CLOSING = (
    "After you submit it, we will have your information and we will be "
    "expecting your meter. Please remember to send it to us. If any "
    "questions come up later, you can email us at service at tensitron dot "
    "com or call us at seven-zero-two, two-three-zero, one-six-one-six."
)

DECLINE_ACK = "No problem! Is there anything else I can help you with today?"

_REPEAT_KEYWORDS = (
    "repeat",
    "again",
    "come again",
    "didn't catch",
    "didnt catch",
    "didn't hear",
    "didnt hear",
    "pardon",
    "sorry what",
    "sorry, what",
    "one more time",
    "say again",
)

_AFFIRMATIVE_KEYWORDS = (
    "yes",
    "yeah",
    "yep",
    "yup",
    "sure",
    "please",
    "okay",
    "ok",
    "go ahead",
    "guide me",
    "walk me",
    "sounds good",
    "that'd help",
    "that would help",
)

_NEGATIVE_KEYWORDS = (
    "no",
    "nah",
    "nope",
    "not really",
    "no thanks",
    "no thank you",
    "i'm good",
    "im good",
    "not now",
    "that's okay",
    "thats okay",
    "i'll figure",
    "ill figure",
)


# Asking for the form walkthrough at ANY point in the call (even after
# declining the offer, or before it was made) starts the scripted
# walkthrough - Gemini must never improvise these steps.
_GUIDANCE_REQUEST_KEYWORDS = (
    "guidance",
    "guide me",
    "guide us",
    "walk me through",
    "walk through",
    "how do i fill",
    "how can i fill",
    "how to fill",
    "help me fill",
    "help me with the form",
    "help with the form",
    "help me with the calibration form",
    "fill out the form",
    "fill out the calibration",
    "fill the form",
    "fill in the form",
    "form guidance",
    "step by step",
    "step-by-step",
    "instructions for the form",
)

_GUIDANCE_NEGATIONS = (
    "don't need",
    "dont need",
    "do not need",
    "no need",
    "no guidance",
    "not need",
    "no thanks",
    "no thank you",
    "don't want",
    "dont want",
    "without",
)


def wants_guidance(text: str) -> bool:
    """True if the customer is asking for the calibration form walkthrough."""
    lowered = (text or "").lower()
    if any(neg in lowered for neg in _GUIDANCE_NEGATIONS):
        return False
    return any(kw in lowered for kw in _GUIDANCE_REQUEST_KEYWORDS)


def start_walkthrough() -> Dict[str, Any]:
    """Turn result for starting the walkthrough from step 1."""
    return {
        "response": f"{GUIDE_INTRO} {GUIDE_STEPS[0]}",
        "present_task": "WAIT",
        "phase": "walking",
        "step_idx": 0,
        "call_ended": False,
    }


def _contains_any(text: str, keywords: tuple) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in keywords)


def _wants_repeat(text: str) -> bool:
    return _contains_any(text, _REPEAT_KEYWORDS)


def _is_affirmative(text: str) -> bool:
    return _contains_any(text, _AFFIRMATIVE_KEYWORDS)


def _is_negative(text: str) -> bool:
    return _contains_any(text, _NEGATIVE_KEYWORDS)


def handle_guidance_turn(phase: str, step_idx: int, user_message: str) -> Dict[str, Any]:
    """`phase` is the state going INTO this turn ("offered" or
    "walking"); the returned "phase"/"step_idx" are what call.py should
    persist for the NEXT turn.
    """

    is_timeout = user_message.strip() == TIMEOUT_SENTINEL

    if phase == "offered":
        if is_timeout:
            # Re-ask rather than assume either answer on silence.
            return {
                "response": OFFER_QUESTION,
                "present_task": "WAIT",
                "phase": "offered",
                "step_idx": 0,
                "call_ended": False,
            }
        if _is_affirmative(user_message):
            return {
                "response": f"{GUIDE_INTRO} {GUIDE_STEPS[0]}",
                "present_task": "WAIT",
                "phase": "walking",
                "step_idx": 0,
                "call_ended": False,
            }
        if _is_negative(user_message):
            return {
                "response": DECLINE_ACK,
                "present_task": "CONTINUE",
                "phase": "done",
                "step_idx": 0,
                "call_ended": False,
            }
        # Ambiguous - ask again rather than guess either way.
        return {
            "response": OFFER_QUESTION,
            "present_task": "WAIT",
            "phase": "offered",
            "step_idx": 0,
            "call_ended": False,
        }

    # phase == "walking"
    if is_timeout or _wants_repeat(user_message):
        return {
            "response": GUIDE_STEPS[step_idx],
            "present_task": "WAIT",
            "phase": "walking",
            "step_idx": step_idx,
            "call_ended": False,
        }

    next_idx = step_idx + 1
    if next_idx < len(GUIDE_STEPS):
        return {
            "response": GUIDE_STEPS[next_idx],
            "present_task": "WAIT",
            "phase": "walking",
            "step_idx": next_idx,
            "call_ended": False,
        }

    # Just confirmed the last step - read the closing line and hand
    # back to the normal workflow for "anything else" + a real goodbye.
    return {
        "response": GUIDE_CLOSING,
        "present_task": "CONTINUE",
        "phase": "done",
        "step_idx": 0,
        "call_ended": False,
    }
