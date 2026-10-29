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
# ORIGINAL: MAX_CONSECUTIVE_WAITS = 3
MAX_CONSECUTIVE_WAITS = 5  # 2026-10-28 operator rule (was 3)
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


# 2026-10-28 OPERATOR RULE: menus arrive one option per turn, so the LLM kept
# saying "wait" and the wait cap hung up before pressing an option it had
# already heard. Deterministic scan for "<live-person option> ... press N" /
# "press N for <live-person option>" in the newest fragment.
_LIVE_WORDS = (
    r"operator|receptionist|representative|customer care|customer service|"
    r"inquiries|enquiries|sales|someone|a person|an agent"
)
_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "star": "*", "pound": "#", "hash": "#",
}
_DIGIT = r"(\d|zero|one|two|three|four|five|six|seven|eight|nine|star|pound|hash)"
# 2026-10-29 (WireCo): after "enter the extension ... otherwise press zero" the
# operator is behind 0.
_ZERO_FALLBACK = re.compile(
    r"(?:otherwise|if you (?:do not|don't) know)[^.]{0,80}?\bpress\s+(zero|0)\b", re.I
)

# 2026-10-29 (DCI Hollow Metal): once we have pressed a key, a greeting like
# "DCI, how can I help you?" means a person picked up.
_HUMAN_GREETING = re.compile(
    r"\b(how (?:can|may) (?:i|we) (?:help|direct|assist)|can i help you|may i help you|"
    r"what can i do for you)\b",
    re.I,
)

_LIVE_PRESS = [
    re.compile(rf"(?:{_LIVE_WORDS})[^.]{{0,40}}?\b(?:press|dial)\s+{_DIGIT}\b", re.I),
    re.compile(rf"\b(?:press|dial)\s+{_DIGIT}\b[^.]{{0,20}}?\bfor\b[^.]{{0,25}}?(?:{_LIVE_WORDS})", re.I),
]


def find_live_person_key(text: str) -> str:
    """Return the key for a live-person menu option named in `text`, or ''."""
    for pat in _LIVE_PRESS:
        m = pat.search(text or "")
        if m:
            tok = m.group(1).lower()
            return _DIGIT_WORDS.get(tok, tok)
    m = _ZERO_FALLBACK.search(text or "")
    if m:
        return "0"
    return ""


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

    # 2026-10-29: after we pressed a key, a person greeting ("how can I help
    # you?") ends menu mode at once, and any other non-menu line is judged on
    # its own - the stale menu text in the heard window made the classifier
    # keep saying "menu" and Anne stayed silent on a live operator.
    if text and ivr["active"] and ivr["presses"] and not looks_like_ivr(text):
        if _HUMAN_GREETING.search(text):
            return _no_longer_ivr()
        ivr["heard"] = []

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

    target = verdict.get("target", "")
    reason = verdict.get("reason", "")

    # Operator rule 1: press a named live-person option at once.
    if action != "press" and len(ivr["presses"]) < MAX_PRESSES:
        live_key = find_live_person_key(text)
        # ORIGINAL condition: only before any key was pressed. Now a key that was
        # not pressed yet may follow (e.g. 1 for customer care, then 0 at the extension prompt).
        if live_key and live_key not in ivr["presses"]:
            action, digit, target, reason = "press", live_key, "representative", "operator rule: live-person option heard"

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
        # Operator rule 2: nothing pressed yet -> try 0 once instead of hanging up.
        if not ivr["presses"] and ivr["waits"] > MAX_CONSECUTIVE_WAITS:
            action, digit, target, reason = "press", "0", "representative", "operator rule: wait cap reached, trying 0"
        else:
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
            # ORIGINAL: "target": verdict.get("target", ""), "reason": verdict.get("reason", "")
            "target": target,
            "reason": reason,
            "heard": list(ivr["heard"]) if action != "press" else [text],
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
            "ivr_log": {k: result.get(k) for k in ("action", "digit", "target", "reason", "heard")},
            "needs_context": False,
            "topic_diversion": False,
            "random_conversation": False,
            "redirect_to_rep": False,
            "redirect_reason": "",
        }
    }
