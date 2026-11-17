"""Turn spoken contact details (as heard by speech-to-text) into the form the CRM needs.

"plus nine one nine zero four zero two three two three two two" -> "+919040232322"
"D-H-R-E-E-T-Y at Outlook"                                    -> "dhreety@outlook.com"

Deterministic on purpose: the memory model's output is passed through here so that a slip in the model can never write
a garbled number or address into the CRM. If a value cannot be parsed safely it is returned UNCHANGED.
"""

import re
from typing import Optional

_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_REPEAT = {"double": 2, "triple": 3}
# filler that speech-to-text leaves in front of an address ("it's d h r i t i at ...", "my email is ...")
_LEAD = re.compile(r"^(?:(?:um+|uh+|so|okay|ok|yes|yeah|please|use|it'?s|its|it is|that'?s|thats|my|email|address|id|is|to|be|the|now|actually|and)\s+)+")
_PROVIDERS = {"gmail", "outlook", "yahoo", "hotmail", "icloud", "live", "aol", "proton", "protonmail", "rediffmail"}


def normalize_phone(text: Optional[str]) -> Optional[str]:
    """Spoken/typed phone number -> digits with a leading '+' if the speaker said plus. Unparseable -> unchanged."""
    if not text:
        return text
    raw = str(text)
    tokens = re.findall(r"[a-z]+|[0-9]+|\+", raw.lower())
    first_digit = next((i for i, t in enumerate(tokens) if t in _DIGIT_WORDS or t.isdigit()), len(tokens))
    plus = any(t in ("+", "plus") for t in tokens[:first_digit])      # "um, use plus nine one ..." still counts
    digits, repeat = [], 1
    for tok in tokens:
        if tok in _REPEAT:
            repeat = _REPEAT[tok]
        elif tok in _DIGIT_WORDS:
            digits.append(_DIGIT_WORDS[tok] * repeat)
            repeat = 1
        elif tok.isdigit():
            digits.append(tok)
            repeat = 1
        # anything else (plus, filler words such as "my", "number", "is") is ignored
    number = "".join(digits)
    if not 7 <= len(number) <= 15:
        return raw
    return ("+" if plus else "") + number


def _collapse_spelled(part: str) -> str:
    """'d-h-r-e-e-t-y' or 'd h r e e t y' (single characters separated by spaces/hyphens) -> 'dhreety'."""
    if re.fullmatch(r"[a-z0-9]([\s\-][a-z0-9])+", part):
        return re.sub(r"[\s\-]", "", part)
    return part


def normalize_email(text: Optional[str]) -> Optional[str]:
    """Spoken/typed email -> 'name@domain.tld'. Unparseable -> unchanged."""
    if not text:
        return text
    raw = str(text)
    s = raw.strip().lower()
    s = _LEAD.sub("", s)
    if "@" not in s:
        s = re.sub(r"\s+at the rate(?: of)?\s+|\s+at\s+", "@", s, count=1)
    if "@" not in s:
        return raw
    s = re.sub(r"\s+dot\s+|\s*\bdot\b\s*", ".", s)
    s = re.sub(r"\s+underscore\s+|\s*\bunderscore\b\s*", "_", s)
    local, _, domain = s.partition("@")
    local = ".".join(_collapse_spelled(p.strip()) for p in local.split("."))
    local = local.replace(" ", "")
    domain = domain.replace(" ", "").strip(".")
    if domain and "." not in domain and domain in _PROVIDERS:
        domain += ".com"
    result = f"{local}@{domain}"
    if re.fullmatch(r"[a-z0-9._%+\-]+@[a-z0-9\-]+(\.[a-z0-9\-]+)+", result):
        return result
    return raw
