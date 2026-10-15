"""Latency fast path: skip the Brain + Memory Gemini calls on trivial turns.

On a plain "yes" / "uh-huh" / "okay, I see" the Brain's answer is fully
predictable (the current stage was just asked about and got confirmed) and
Memory has nothing to extract, so those two LLM calls are pure delay before
the Decision call. This module decides, with NO LLM, whether a turn is that
simple. Anything else (a "no", a correction, a question, a longer sentence,
a reply that doesn't match what Anne last asked) returns None and the normal
Memory + Brain -> Decision path runs unchanged.

Kill switch: set env ANNE_FAST_PATH=0 on Render to disable without a deploy.
"""

import os
import re
from typing import Any, Dict, Optional

from call_agent.app.agents.brain_agent import get_next_stage

ENABLED = os.getenv("ANNE_FAST_PATH", "1") != "0"

# Clear confirmations (also fine for the name question, e.g. "speaking").
_AFFIRM = {
    "yes", "yeah", "yep", "yup", "sure", "correct", "right", "absolutely",
    "exactly", "yes please", "yes it is", "it is", "thats right",
    "that is right", "thats correct", "that is correct", "yes correct",
    "yes thats right", "yes that is right", "yeah thats right",
    "yeah that is right", "yeah correct", "yes speaking", "speaking",
    "this is he", "this is she", "yes this is he", "yes this is she",
    "yes thats me", "thats me", "yes it is correct",
}
# Soft acknowledgements: enough to confirm contact details when they are
# the "uh-huh" kind, and enough to acknowledge the calibration reminder.
_SOFT_YES = {"uh huh", "uhhuh", "mm hmm", "mmhmm", "mhm", "mm hm"}
_ACK = {
    "okay", "ok", "alright", "all right", "i see", "got it", "oh okay",
    "ah okay", "oh i see", "ah i see", "thank you", "thanks", "okay thank you",
    "okay thanks", "okay great", "great", "sounds good", "noted",
    "understood", "okay got it", "okay i see", "okay noted",
}

# What Anne's LAST message must contain for a plain yes to mean "confirmed
# this stage". If it doesn't match we fall back to the full Brain.
_CUES = {
    "name_confirmation": ("speaking with", "your name"),
    "email_confirmation": ("email",),
    "phone_confirmation": ("number", "phone"),
    "calibration": ("calibration",),
}
_STAGE_FLAG = {
    "name_confirmation": "name_confirmed",
    "email_confirmation": "email_confirmed",
    "phone_confirmation": "phone_confirmed",
    "calibration": "calibration_completed",
}


# Short replies made ONLY of these words count as a plain confirmation
# ("yeah you are", "mhm it is", "okay thank you for the reminder"). Any word
# outside this list ("not", "but", "could", "email", "different"...) sends
# the turn to the full Brain, so negatives, questions and corrections never
# take the fast path.
_CORE = {
    "yes", "yeah", "yep", "yup", "sure", "correct", "right", "absolutely",
    "definitely", "exactly", "indeed", "speaking", "mhm", "mm", "hmm", "uh", "huh",
}
_FILLER = {
    "it", "is", "you", "are", "that", "thats", "this", "i", "am", "he", "she",
    "me", "my", "so", "please", "still", "the", "for", "thank", "thanks",
    "reminder", "okay", "ok", "alright", "great", "see", "got", "noted", "oh",
    "ah", "well", "um", "uhm", "very", "much", "yes", "correct",
}
_MAX_WORDS = 6


def _vocab_only(text: str, need_core: bool) -> bool:
    words = text.split()
    if not words or len(words) > _MAX_WORDS:
        return False
    if not all(w in _CORE or w in _FILLER for w in words):
        return False
    return (not need_core) or any(w in _CORE for w in words)


def _normalize(text: str) -> str:
    text = re.sub(r"[^a-z' ]", " ", (text or "").lower()).replace("'", "")
    return re.sub(r"\s+", " ", text).strip()


def _last_assistant_message(state: Dict[str, Any]) -> str:
    for msg in reversed(state.get("conversation") or []):
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")).lower()
    return ""


def fast_brain_update(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Brain output for a trivial confirming turn, or None to use the LLM."""
    if not ENABLED:
        return None

    progress = dict(state.get("progress") or {})
    stage = get_next_stage(progress)
    if stage not in _STAGE_FLAG:
        return None  # workflow already complete: nothing for Brain to do

    raw = state.get("user_message", "") or ""
    text = _normalize(raw)
    # ORIGINAL exact-phrase matching (kept for revert):
    # if stage == "calibration":
    #     ok = text in _AFFIRM or text in _SOFT_YES or text in _ACK
    # else:
    #     ok = text in _AFFIRM or text in _SOFT_YES
    if stage == "calibration":
        ok = text in _AFFIRM or text in _SOFT_YES or text in _ACK or _vocab_only(text, need_core=False)
    else:
        ok = text in _AFFIRM or text in _SOFT_YES or _vocab_only(text, need_core=True)
    # Non-English script in the transcript (e.g. a Hindi/Tamil word): not
    # sure what was said, so use the full Brain.
    if raw.isascii() is False:
        ok = False
    if not ok:
        return None

    last = _last_assistant_message(state)
    if not any(cue in last for cue in _CUES[stage]):
        return None

    progress[_STAGE_FLAG[stage]] = True
    return {
        "progress": progress,
        "current_stage": get_next_stage(progress),
        "stage_confirmed": stage,
    }
