"""Voicemail agent.

Deterministic (no LLM): if the call was picked up by an answering machine,
leave a fixed message and hang up instead of running the normal workflow.

Detection comes from two places:
  1. Twilio AMD - the `AnsweredBy` value (needs `MachineDetection` on the
     call-creation request; with DetectMessageEnd Twilio waits for the
     greeting + beep to finish before connecting the stream, so the
     message lands on the recording).
  2. A phrase check on the first customer turns, as a safety net when AMD
     said "human"/"unknown" or wasn't requested.
"""

from typing import Any, Dict

VOICEMAIL_MESSAGE = (
    "Hey this is Anne from Tensitron, calling to confirm your contact "
    "details registered with us and also remind you of an upcoming "
    "calibration. Please get back to us when you are available."
)

# Twilio AnsweredBy values that mean a machine/voicemail picked up.
# ("fax" is a machine too, but leaving a spoken message there is pointless.)
_MACHINE_ANSWERED_BY = {
    "machine_start",
    "machine_end_beep",
    "machine_end_silence",
    "machine_end_other",
}

# Specific on purpose - a human saying "I'm not available right now" must
# not trip this.
_VOICEMAIL_PHRASES = (
    "leave a message",
    "leave your message",
    "after the tone",
    "after the beep",
    "voicemail",
    "voice mail",
    "mailbox",
    "not available to take your call",
    "unable to take your call",
    "can't take your call",
    "cannot take your call",
)


def is_machine_answer(answered_by: str) -> bool:
    return (answered_by or "").strip().lower() in _MACHINE_ANSWERED_BY


def looks_like_voicemail(text: str) -> bool:
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in _VOICEMAIL_PHRASES)


def voicemail_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Graph node: the hardcoded voicemail message + end the call."""
    return {
        "decision": {
            "response": VOICEMAIL_MESSAGE,
            "present_task": "END_CALL",
            "needs_context": False,
            "topic_diversion": False,
            "random_conversation": False,
            "redirect_to_rep": False,
            "redirect_reason": "",
        },
        "call_ended": True,
    }
