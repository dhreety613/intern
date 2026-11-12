"""Rules classifier for what the far end is saying. Covers what Twilio AMD can't (IVR, hold,
screening, carrier messages). An optional async LLM fallback is wired in the server for UNKNOWN."""
from __future__ import annotations

import asyncio
import re
from typing import Awaitable, Callable

from .ivr import parse_options
from .models import Label

LlmFallback = Callable[[str], Awaitable[Label]]


def _any(pats: list[str]) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in pats))


_norm_re = re.compile(r"[^\w\s']")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", _norm_re.sub(" ", text.lower())).strip()


# -- carrier -----------------------------------------------------------------
DISCONNECTED = _any([r"no longer in service", r"has been disconnected", r"is disconnected",
                     r"been changed,? disconnected", r"disconnected or is no longer"])
INVALID = _any([r"not in service", r"cannot be completed as dialed", r"can't be completed as dialed",
                r"is not a valid (phone )?number", r"number you (have )?(dialed|called) is (incorrect|invalid)",
                r"check the number and (dial|try) again", r"invalid number", r"not a working number",
                r"no such number", r"your call cannot be completed", r"we're sorry,? the number you have dialed",
                r"has not been assigned", r"number is not (currently )?assigned"])
UNREACHABLE = _any([r"(subscriber|wireless customer|cellular customer|customer) (you|that you) (are|have|were) "
                    r"(calling|dialed|called|trying to reach)",
                    r"is (currently )?(switched off|out of (the )?(coverage|service area)|not reachable|unreachable)",
                    r"(temporarily|currently) (unavailable|not available) .*try again later"])

# -- mailbox -----------------------------------------------------------------
MB_FULL = _any([r"mailbox (is|has been|currently is) (currently )?full", r"mail box is full",
                r"(cannot|can't|unable to|not able to) (accept|take|record|receive) (any )?(more )?(messages|new messages)",
                r"mailbox has reached", r"no more messages can be left", r"voice ?mail (box )?is full",
                r"memory is full"])
MB_DISABLED = _any([r"(mailbox|voice ?mail( box)?) (has not|hasn't|is not|isn't) (been )?(set up|setup|activated|enabled|available|active)",
                    r"not (been )?set up", r"not accepting (any )?(messages|voice ?mail)",
                    r"(does not|doesn't) (accept|take) (any )?(messages|voice ?mail)",
                    r"(mailbox|voice ?mail) (is )?(disabled|deactivated|unavailable)",
                    r"no voice ?mail (service|box)", r"cannot be left for this (number|mailbox)"])

# -- screening ---------------------------------------------------------------
SCREEN_PROMPT = _any([r"(state|say|record|tell me) your name", r"your name and (the )?(reason|purpose)",
                      r"screening service", r"call screen", r"who is calling", r"who's calling (please|today)? *(after|at) the (tone|beep)",
                      r"name and reason for (your )?call", r"reason for (your )?call(ing)? after the (tone|beep)",
                      r"after the (tone|beep),? (please )?(say|state|record) your name",
                      r"record your name and"])
SCREEN_REJECT = _any([r"(is|are) not (available|accepting) (to take )?(your )?call", r"isn't taking calls",
                      r"(does not|doesn't) want to be disturbed", r"declined (your|this) call",
                      r"(unable|not able|cannot|can't) to? ?take your call", r"not taking (any )?calls",
                      r"(is|are) unavailable to take your call", r"(unable|not able) to answer",
                      r"is not available right now", r"has declined"])

# -- voicemail ---------------------------------------------------------------
VOICEMAIL = _any([r"leave (me |us |a |your )?(a )?(short |brief )?message", r"after the (tone|beep)", r"at the (tone|beep)",
                  r"voice ?mail", r"mailbox", r"record your message", r"(can't|cannot|unable to) (come|get) to the phone",
                  r"(not|isn't|am not|are not) (available|here|in) (right now|at the moment|to take)",
                  r"when you('re| are) (finished|done) recording", r"press (pound|#|1) (when|for more)",
                  r"you('ve| have) reached the (voice ?mail|mailbox)"])

# -- hold --------------------------------------------------------------------
HOLD = _any([r"please (continue to )?hold", r"hold (on )?(the line|please)", r"(stay|remain) on the line",
             r"your call (is|will be) (important|answered|being)", r"next available (agent|representative|associate|person)",
             r"all (of our )?(agents|representatives|operators|associates|lines|specialists) are (currently )?(busy|assisting)",
             r"thank you for (holding|your patience)", r"estimated wait", r"one moment (please)?", r"transfer(ring)? your call",
             r"please wait", r"hold on", r"connecting you", r"will be with you (shortly|in a moment)", r"hold for the next"])

# -- ivr ---------------------------------------------------------------------
IVR_INVALID = _any([r"(not|isn't) a valid (entry|option|selection|choice|response|extension)",
                    r"invalid (entry|option|selection|choice|response|extension)",
                    r"(i |we )(did not|didn't) (receive|get|hear|understand|catch) (any|that|your)",
                    r"sorry,? (i|we) (didn't|did not) (get|catch|understand)", r"that (was|is) not a valid",
                    r"no (entry|selection|input) (was )?(received|detected)"])
IVR_EXT = _any([r"enter (the |your |his |her |their )?(party's |parties' )?(\d[- ]digit )?extension",
                r"dial (the |your )?(party's )?extension", r"extension of the (person|party)",
                r"(enter|spell|dial) (the )?(first three|last name|first name|first few)",
                r"dial by name", r"if you know (your |the )?(party's )?extension"])
IVR_PHRASE = _any([r"main menu", r"following options", r"menu options (have|has) (recently )?changed",
                   r"listen (carefully|closely) as (our|the) (menu|options)", r"to hear (these|this|the) (options|menu) again",
                   r"using your (touch ?tone|keypad|phone)", r"press (the )?(star|pound|zero|\d)", r"for .{0,40}(press|dial) (\d|zero|one|two|three|four|five|six|seven|eight|nine|star|pound)",
                   r"say or (press|enter)", r"in a few words,? (tell|say)", r"tell me (what|why|how)", r"what can i help you with today",
                   r"briefly (say|tell|describe)", r"para espa[nñ]ol"])

HUMAN_MAX_WORDS = 14

# -- human -------------------------------------------------------------------
HUMAN = _any([r"^(hello|hi|hey|yes|yeah|yep|yup|what|hullo|hola)\b *(who|this|\?)?", r"\bspeaking\b",
              r"\bthis is (?!(?:to|an?|the|your|from|our|my|about|regarding|in|for|call|calling|message|notification|automated|recorded|important|urgent|final)\b)[a-z]+\b(?! (is|will) )", r"how (can|may|could) i (help|assist)", r"can i help you",
              r"what can i do for you", r"who('s| is) (this|calling)\b", r"who am i speaking (with|to)",
              r"may i ask who", r"go ahead", r"(good )?(morning|afternoon|evening),? *[a-z ]{0,30}(this is|speaking)"])
_GREETING_ONLY = _any([r"thank you for calling", r"welcome to", r"you('ve| have) reached"])

# phrases that signal a prerecorded/automated source even if a HUMAN cue matches ("this is", "hello")
_AUTOMATED_CUES = _any([r"this is an? (automated|recorded)", r"your call (may|will) be (monitored|recorded)",
                        r"calls? (may|will) be (monitored|recorded)", r"press ", r"menu", r"voice ?mail", r"mailbox"])


def classify(text: str) -> Label:
    t = norm(text)
    if not t:
        return Label.UNKNOWN

    if DISCONNECTED.search(t):
        return Label.CARRIER_DISCONNECTED
    if INVALID.search(t):
        return Label.CARRIER_INVALID
    if UNREACHABLE.search(t):
        return Label.CARRIER_UNREACHABLE
    if MB_FULL.search(t):
        return Label.MAILBOX_FULL
    if MB_DISABLED.search(t):
        return Label.MAILBOX_DISABLED
    if SCREEN_PROMPT.search(t):
        return Label.SCREEN_PROMPT
    opts = parse_options(text)
    if len(opts) >= 2:
        return Label.IVR_MENU        # >=2 keys is a menu, even if it mentions voicemail
    if VOICEMAIL.search(t):
        return Label.VOICEMAIL_GREETING
    if SCREEN_REJECT.search(t):
        return Label.SCREEN_REJECT
    if IVR_INVALID.search(t):
        return Label.IVR_INVALID_ENTRY
    if IVR_EXT.search(t):
        return Label.IVR_EXTENSION_PROMPT

    if opts or IVR_PHRASE.search(t):
        return Label.IVR_MENU
    if HOLD.search(t):
        return Label.HOLD
    if _AUTOMATED_CUES.search(t):
        return Label.UNKNOWN
    if _GREETING_ONLY.search(t) and not re.search(r"\bspeaking\b|\bthis is [a-z]+", t):
        return Label.AMBIGUOUS_GREETING   # receptionist or recording; controller probes
    if HUMAN.search(t) and len(t.split()) <= HUMAN_MAX_WORDS:   # a person's answer is short; a long monologue is a recording
        return Label.HUMAN
    return Label.UNKNOWN


def is_greeting_only(text: str) -> bool:
    return bool(_GREETING_ONLY.search(norm(text)))


async def classify_with_fallback(text: str, llm: LlmFallback | None, min_words: int = 6, timeout_s: float = 1.6) -> Label:
    label = classify(text)
    if label is Label.UNKNOWN and llm and len(norm(text).split()) >= min_words:
        try:
            return await asyncio.wait_for(llm(text), timeout_s)
        except Exception:
            return Label.UNKNOWN
    return label
