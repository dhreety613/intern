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

from call_agent.app.agents.voicemail_agent import looks_like_voicemail
from call_agent.app.greeting import opening_message

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")
MODEL = "gemini-3.5-flash-lite"

# 2026-10-29: the LLM decides menu / voicemail / person on every customer turn until
# a human is confirmed (it used to run only when a keyword cue matched). The keyword
# lists below are now just the fallback when the model call fails. ANNE_LLM_IVR=0
# restores the old keyword-gated behaviour without a deploy.
LLM_IVR = os.getenv("ANNE_LLM_IVR", "1") != "0"

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

LATEST LINE (judge this one; the lines before it are only context for what
was already said / pressed): {json.dumps(heard[-1] if heard else "", ensure_ascii=False)}

Decide who is speaking in the LATEST line and what Anne should do.

"speaker":
- "menu" = an automated phone system that offers choices or collects input:
  department options, "for X press N", extension / directory / dial-by-name
  prompts, hold or transfer announcements ("please stay on the line", "your call
  is being transferred"), language selection.
- "person" = a live human is talking now: answering the phone, giving a company
  or own name, asking how they can help or who is calling, replying to Anne -
  EVEN IF earlier lines were a menu and a key was pressed (the menu has now
  connected to a person, e.g. the operator).
- "voicemail_greeting" = a RECORDED answering message for ONE person or business
  mailbox, i.e. the call was not answered live. Signs: "you have reached ...",
  "is not available / can't take your call", "leave a message (after the
  tone/beep)", "record your message", "the message is not recorded, go ahead",
  "forwarded to an automated voice messaging system", "mailbox is full / not set
  up", "subscriber unavailable". Judge by MEANING, not keywords: a mailbox greeting
  that says "press pound / press 1 / hang up when you are finished" is STILL a
  voicemail greeting - those are recording-end instructions, not a department menu.
  A menu, by contrast, offers several routes to different people or departments.
- "unclear" = cannot tell yet (cut off, noise, a lone fragment).

Rules:
- speaker person or voicemail_greeting -> is_ivr = false. For voicemail_greeting
  action = "voicemail"; for person action = "none".
- speaker menu -> is_ivr = true, then choose the action that gets Anne to a LIVE
  PERSON:
  - Prefer an option that clearly names a person: operator, receptionist,
    representative, customer service/care, sales, general inquiries.
  - PRESS 0 WHEN NOTHING BETTER IS OFFERED. Pressing 0 reaches an operator or
    receptionist in most phone systems, so it is the default escape hatch: use it
    when the menu has no live-person option, when it only lists automated or
    unrelated departments, or at an extension / directory / name prompt ("enter
    the extension... otherwise press 0"). Press 0 once per menu level rather than
    hanging up. If 0 is already in KEYS ALREADY PRESSED for this menu, try another
    live-person option or a voicemail option instead.
  - Avoid automated options (hours, directions, balances, payments, "press 1 for
    English" only as a language step if nothing else is offered).
  - GOAL 2: only if no live-person route exists, choose an option to leave a message.
  - If the menu seems unfinished (more options likely coming, or only an intro
    like "Thank you for calling") and nothing suitable has been heard yet,
    action = "wait". Do not wait once a full menu has been read.
  - action = "hangup" only if the system needs an account number, PIN or anything
    Anne cannot give with one key, or nothing can reach a person or a mailbox.
  - action = "press" requires digit to be a single key from 0-9, * or #.
  - Do not press a key that is already in KEYS ALREADY PRESSED unless the menu is
    clearly a different, new menu.
- speaker unclear -> is_ivr = true, action = "wait".

Return ONLY JSON:
{{"speaker": "menu"|"person"|"voicemail_greeting"|"unclear", "is_ivr": true|false, "action": "press"|"wait"|"hangup"|"voicemail"|"none", "digit": "", "target": "representative"|"voicemail"|"", "reason": "short"}}
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
    # ORIGINAL gate (kept as the ANNE_LLM_IVR=0 behaviour):
    #   if not text or not (ivr["active"] or looks_like_ivr(text)): ...
    if LLM_IVR:
        run_llm = bool(text) and (ivr["active"] or not state.get("human_confirmed"))
    else:
        run_llm = bool(text) and (ivr["active"] or looks_like_ivr(text))
    if not run_llm:
        return _no_longer_ivr() if text else not_ivr

    ivr["heard"] = (ivr["heard"] + [text])[-HEARD_WINDOW:]
    ivr["turns"] += 1

    try:
        verdict = _classify(ivr["heard"], ivr["presses"])
    except Exception:
        # Classifier failed: fall back to the old keyword lists. A menu -> stay
        # silent and keep listening (the turn cap below still ends the call);
        # a voicemail phrase -> voicemail; anything else -> treat as a person.
        if looks_like_ivr(text) or ivr["active"]:
            verdict = {"speaker": "menu", "is_ivr": True, "action": "wait", "digit": "", "target": ""}
        elif looks_like_voicemail(text):
            verdict = {"speaker": "voicemail_greeting", "is_ivr": False, "action": "voicemail"}
        else:
            verdict = {"speaker": "person", "is_ivr": False, "action": "none"}

    # A recorded mailbox greeting (even one that says "press pound when finished"):
    # leave the fixed voicemail message. call.py flags the session as voicemail so
    # the stream waits for the beep.
    if verdict.get("speaker") == "voicemail_greeting" or verdict.get("action") == "voicemail":
        ivr["active"] = False
        return {"ivr": ivr, "ivr_result": {"is_ivr": False, "action": "voicemail", "voicemail_greeting": True}}

    # A person (e.g. the operator we just reached): _no_longer_ivr() restarts the
    # whole tested workflow from the opening line if we had been navigating a menu.
    if not verdict.get("is_ivr") or verdict.get("speaker") == "person":
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
