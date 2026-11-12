"""IVR menu parsing + key choice. Pure functions, no I/O."""
from __future__ import annotations

import re
from dataclasses import dataclass

_KEY = r"(?:\d|zero|one|two|three|four|five|six|seven|eight|nine|star|pound|hash)"
_KEYMAP = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
           "six": "6", "seven": "7", "eight": "8", "nine": "9", "star": "*", "pound": "#", "hash": "#"}

# style A: "press 1 for sales"      style B: "for sales, press 1"
_A = re.compile(rf"(?:press|dial|enter)\s+(?P<k>{_KEY})\s*(?:now\s+)?,?\s*(?:for|to|if)\s+(?P<d>[^,.;]+)")
_A2 = re.compile(rf"(?:^|,|\bor\b|\band\b)\s*(?P<k>\d|star|pound)\s+(?:for|to)\s+(?P<d>[^,.;]+)")
_B = re.compile(rf"(?:for|to|if)\s+(?P<d>[^,.;]+?)\s*,?\s*(?:please\s+)?(?:press|dial|enter)\s+(?P<k>{_KEY})\b")
_ZERO_ANYTIME = re.compile(r"(?:press|dial)\s+(?:zero|0)\b")

# intent ranking: earlier = better. Matched against option description.
DEFAULT_TARGETS = [
    # which TEAM to reach when a menu lists several: engineering, then parts & service, then QA
    r"\bengineering\b|\bengineers?\b|\btechnical\b|\btech support\b",
    r"\bparts\b|(?<!customer )\bservice (department|team|and|&)\b|\bparts (and|&) service\b",
    r"\bq\.? ?a\b|\bquality\b",
    # then any route to a person
    r"\b(speak|talk)\b.*\b(person|someone|representative|agent|associate|rep)\b",
    r"\bcalibration\b",
    r"\bcustomer (service|support)\b",
    r"\b(sales|orders?)\b",
    r"\bservice\b",
    r"\b(operator|receptionist|reception|front desk)\b",
    r"\b(representative|agent|associate|officer|live|someone|somebody|person|human)\b",
    r"\b(accounts?|billing)\b",
]
_AVOID = re.compile(r"\b(hear (these|this)|repeat|spanish|espa[nñ]ol|directory|hours|location|fax|"
                    r"leave a message|voicemail|voice mail|pay(ment)?|careers?|employment)\b")


@dataclass(frozen=True)
class IvrOption:
    key: str
    desc: str


def _norm_key(k: str) -> str:
    return _KEYMAP.get(k, k)


def parse_options(text: str) -> list[IvrOption]:
    t = text.lower()
    out: dict[str, IvrOption] = {}
    for sent in re.split(r"[.?!;]\s*", t):
        sent = sent.strip()
        if not sent:
            continue
        if re.match(r"(?:to \w+ )?(?:press|dial)\b", sent) or sent.startswith("press") or sent.startswith("dial"):
            ms = list(_A.finditer(sent)) + list(_A2.finditer(sent))
        else:
            ms = list(_B.finditer(sent)) or list(_A.finditer(sent))
        for m in ms:
            k = _norm_key(m.group("k"))
            out.setdefault(k, IvrOption(k, m.group("d").strip()))
    return sorted(out.values(), key=lambda o: o.key)


def operator_available(text: str) -> bool:
    return bool(_ZERO_ANYTIME.search(text.lower())) or any(o.key == "0" for o in parse_options(text))


def choose_key(options: list[IvrOption], targets: list[str] | None = None) -> IvrOption | None:
    """Best option toward a live person. None => nothing matches."""
    targets = targets or DEFAULT_TARGETS
    for pat in targets:
        for o in options:
            if re.search(pat, o.desc) and not _AVOID.search(o.desc):
                return o
    return None


_VM_OPT = re.compile(r"\b(leave (a |your )?message|voice ?mail|mailbox)\b")


def voicemail_option(options: list[IvrOption]) -> IvrOption | None:
    return next((o for o in options if _VM_OPT.search(o.desc)), None)


def validate_llm_pick(kind: str, key: str, options: list[IvrOption]) -> IvrOption | None:
    """Accept the LLM's pick only if it is a key actually offered (or 0 for operator)."""
    by_key = {o.key: o for o in options}
    if kind == "human" and key in by_key:
        return by_key[key]
    if kind == "voicemail" and key in by_key:
        return by_key[key]
    return None
