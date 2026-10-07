"""Voicemail agent.

Deterministic (no LLM): if the call was picked up by an answering machine,
leave a fixed message and hang up instead of running the normal workflow.

Triggered by: Twilio's async AMD flagging a machine (call forwarded to
voicemail), the IVR agent having pressed a voicemail option, or - only
until a human is confirmed on the line (state["human_confirmed"]) - the
customer's speech sounding like a voicemail greeting.
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


def looks_like_voicemail(text: str) -> bool:
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in _VOICEMAIL_PHRASES)


def is_machine_answer(answered_by: str) -> bool:
    return (answered_by or "").strip().lower() in _MACHINE_ANSWERED_BY


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
