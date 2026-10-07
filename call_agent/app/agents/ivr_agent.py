"""IVR agent.

Runs on every customer turn (cheap keyword gate first, LLM only when the
turn sounds like an automated phone menu). When the other end is an IVR,
Anne stays silent and presses keys:

  1. an option that reaches a live person (operator / representative /
     customer service / sales / receptionist), otherwise
  2. a voicemail / leave-a-message option, otherwise
  3. hang up.

The key press itself is sent as DTMF tones over the media stream (see
dtmf.py and TwilioMediaStream._send_dtmf).
"""

import json
import os
import re
from typing import Any, Dict, List

from google import genai

from call_agent.app.greeting import opening_message

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")
MODEL = "gemini-3.5-flash-lite"

# Hard caps so a menu loop can never keep the call alive forever.
MAX_PRESSES = 4
MAX_CONSECUTIVE_WAITS = 3
MAX_IVR_TURNS = 10
HEARD_WINDOW = 4

# Cheap gate: only turns containing one of these (or following an IVR turn)
# are sent to the LLM.
_IVR_CUES = (
    "press ",
    "dial ",
    "extension",
    "main menu",
    "menu option",
    "for sales",
    "for support",
    "for service",
    "for billing",
    "to speak with",
    "to speak to",
    "to repeat",
    "your call is important",
    "your call may be recorded",
    "please hold",
    "stay on the line",
    "para español",
    "company directory",
    "dial by name",
    "operator",
)


def looks_like_ivr(text: str) -> bool:
    lowered = (text or "").lower()
    return any(cue in lowered for cue in _IVR_CUES)


def _new_ivr_state() -> Dict[str, Any]:
    return {
        "active": False,
        "seen": False,
        "turns": 0,
        "waits": 0,
        "heard": [],
        "presses": [],
        "pressed_voicemail": False,
    }


def _get_client():
    return genai.Client(api_key=GEMINI_API_KEY)


def _extract_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text)
    return json.loads(text.strip())


def _classify(heard: List[str], presses: List[str]) -> Dict[str, Any]:
    prompt = f"""
You are listening to a phone call that Anne (an automated caller) placed.
Below is what the other end said, oldest first (speech-to-text, may be
imperfect, a menu may be split over several lines).

HEARD:
{json.dumps(heard, ensure_ascii=False)}

KEYS ALREADY PRESSED ON THIS CALL: {json.dumps(presses)}

Decide whether the other end is an automated phone menu (IVR) and what to do.

Rules:
- is_ivr = false if a real person is talking, or it is a voicemail
  greeting/recording. Then action = "none".
- GOAL 1: reach a live person. Choose the option for an operator,
  representative, receptionist, customer service, sales, or general
  inquiries (including "press 0 for an operator"). Avoid automated options
  (hours, directions, balances, payments, "press 1 for English" is fine only
  as a language step if it is the only thing offered).
- GOAL 2: only if no option reaches a person, choose an option to leave a
  message / voicemail.
- If neither exists in a menu that has been fully read out, action = "hangup".
- If the menu seems unfinished (more options likely coming, or only an
  intro like "Thank you for calling") and nothing suitable has been heard
  yet, action = "wait".
- action = "press" requires digit to be a single key from 0-9, * or #.
- Do not press a key that is already in KEYS ALREADY PRESSED unless the
  menu is clearly a different, new menu.
- If the system asks for an account number, PIN, or anything Anne cannot
  provide with a single key, action = "hangup".

Return ONLY JSON:
{{"is_ivr": true|false, "action": "press"|"wait"|"hangup"|"none", "digit": "", "target": "representative"|"voicemail"|"", "reason": "short"}}
"""
    response = _get_client().models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    return _extract_json(response.text)


def ivr_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Classify the latest customer turn.

    Returns {"ivr": <updated persistent ivr state>,
             "ivr_result": {"is_ivr": bool, "action": ..., "digit": ...}}.
    """
    ivr = dict(state.get("ivr") or _new_ivr_state())
    ivr["heard"] = list(ivr.get("heard", []))
    ivr["presses"] = list(ivr.get("presses", []))

    text = (state.get("user_message") or "").strip()
    not_ivr = {"ivr": ivr, "ivr_result": {"is_ivr": False, "action": "none"}}

    def _no_longer_ivr() -> Dict[str, Any]:
        """The far end stopped being a menu. Decide what that means."""
        was_navigating = bool(ivr["presses"])
        voicemail = ivr["pressed_voicemail"]
        ivr["active"] = False
        ivr["heard"] = []
        ivr["waits"] = 0
        if was_navigating:
            ivr["presses"] = []
            ivr["turns"] = 0
            ivr["pressed_voicemail"] = False
            # We pressed a voicemail option -> this is the voicemail
            # greeting: leave the fixed message. Otherwise a human joined:
            # restart the main agent from the beginning.
            action = "voicemail" if voicemail else "restart"
            return {"ivr": ivr, "ivr_result": {"is_ivr": False, "action": action}}
        return not_ivr

    # Gate: skip the LLM unless this sounds like a menu, or the previous
    # turn already was one (menus arrive in pieces).
    if not text or not (ivr["active"] or looks_like_ivr(text)):
        return _no_longer_ivr() if text else not_ivr

    ivr["heard"] = (ivr["heard"] + [text])[-HEARD_WINDOW:]
    ivr["turns"] += 1

    try:
        verdict = _classify(ivr["heard"], ivr["presses"])
    except Exception:
        # Looks like a menu but the classifier failed: stay silent, keep
        # listening (the turn cap below still ends the call eventually).
        verdict = {"is_ivr": True, "action": "wait", "digit": "", "target": ""}

    if not verdict.get("is_ivr"):
        return _no_longer_ivr()

    ivr["active"] = True
    ivr["seen"] = True

    action = verdict.get("action", "wait")
    digit = str(verdict.get("digit", "") or "").strip()

    if action == "press" and not re.fullmatch(r"[0-9*#]", digit):
        action = "wait"

    if action == "wait":
        ivr["waits"] += 1
    else:
        ivr["waits"] = 0

    # Caps: too many presses/waits/turns means we're stuck in a menu.
    if (
        (action == "press" and len(ivr["presses"]) >= MAX_PRESSES)
        or ivr["waits"] > MAX_CONSECUTIVE_WAITS
        or ivr["turns"] > MAX_IVR_TURNS
    ):
        action = "hangup"

    if action == "press":
        ivr["presses"].append(digit)
        ivr["heard"] = []  # next menu starts fresh
        if verdict.get("target") == "voicemail":
            ivr["pressed_voicemail"] = True

    return {
        "ivr": ivr,
        "ivr_result": {
            "is_ivr": True,
            "action": action,
            "digit": digit if action == "press" else "",
            "target": verdict.get("target", ""),
            "reason": verdict.get("reason", ""),
        },
    }


def ivr_respond(state: Dict[str, Any]) -> Dict[str, Any]:
    """Graph node: turn the IVR verdict into a (silent) decision."""
    result = state.get("ivr_result", {})
    action = result.get("action", "wait")

    # A human joined after the menu: start the call over with the opening.
    response = (
        opening_message(state.get("customer_name", ""))
        if action == "restart"
        else ""
    )

    return {
        "decision": {
            "response": response,
            "present_task": "END_CALL" if action == "hangup" else "CONTINUE",
            "dtmf": result.get("digit", "") if action == "press" else "",
            # Pressed a leave-a-message option: the stream arms a timer so
            # the fixed line still plays if only a beep follows (no greeting).
            "voicemail_key": action == "press" and result.get("target") == "voicemail",
            "ivr": True,
            "needs_context": False,
            "topic_diversion": False,
            "random_conversation": False,
            "redirect_to_rep": False,
            "redirect_reason": "",
        }
    }
