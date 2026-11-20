"""Deterministic output guard. Runs on every sentence BEFORE it reaches TTS.

The prompt tells the model what it may talk about; this layer makes sure a hallucinated price, lead time,
date, phone number or email can never actually be spoken. Anything numeric/entity-like must trace back to
the fact book (call data, form steps, what the caller said, configured support contacts).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .state import CallInput, digits

_DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]
_NUMWORD = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|sixty|ninety|hundred)"
_MONEY = re.compile(r"[$€£₹]|\d\s*(?:dollars?|bucks?|rupees?|cents?|usd|inr|euros?|percent)\b|%|\b(?:\w+)\s+(?:dollars|rupees|percent)\b", re.I)
_TIMEFRAME = re.compile(rf"\b({_NUMWORD})[\s-]+(?:business\s+|working\s+|calendar\s+)?(days?|weeks?|months?|years?|hours?)\b", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_DIGIT_RUN = re.compile(r"\d[\d\s().-]*\d|\d")


@dataclass
class GuardResult:
    ok: bool
    reason: str = ""


def _digit_words_to_digits(text: str) -> str:
    """'seven three seven two two one two one six three' -> '7377221 2163'-style digits so spoken numbers get checked.
    Only runs of >=4 consecutive digit words are converted (keeps 'one or two' as words)."""
    toks = re.findall(r"[A-Za-z']+|\d+|[^\sA-Za-z\d]", text)
    out, buf = [], []

    def flush():
        if len(buf) >= 4:
            out.append("".join(_DIGIT_WORDS[w] for w in buf))
        else:
            out.extend(buf)
        buf.clear()

    for t in toks:
        lt = t.lower()
        if lt in _DIGIT_WORDS and (lt != "oh" or buf):
            buf.append(lt)
        else:
            flush()
            out.append(t)
    flush()
    return " ".join(out)


class FactBook:
    def __init__(self, call: CallInput, support_phone: str = "", support_email: str = ""):
        self._digit_strings: set[str] = set()
        self._emails: set[str] = set()
        self._corpus = ""  # lowercase text that may legitimately contain numbers/timeframes/months
        self._tokens: set[str] = set()
        for s in (call.registered_phone, support_phone):
            self.allow_phone(s)
        for s in (call.registered_email, support_email):
            self.allow_email(s)
        parts = [call.today] + [f"{m.name} {m.due_date}" for m in call.meters]
        parts += [f"{s.id} {s.title} {s.instruction}" for s in call.form_steps]
        self.allow_text(" ".join(parts))

    def allow_phone(self, s: str):
        d = digits(s)
        if len(d) >= 4:
            self._digit_strings.add(d)
            if len(d) == 11 and d[0] == "1":
                self._digit_strings.add(d[1:])

    def allow_email(self, s: str):
        if s:
            self._emails.add(s.strip().lower())

    def allow_text(self, text: str):
        """Whitelist numbers/months/timeframes present in text (call data, or what the caller said)."""
        t = _digit_words_to_digits(text).lower()
        self._corpus += " " + t
        for m in _EMAIL.finditer(text):
            self.allow_email(m.group(0))
        for m in _DIGIT_RUN.finditer(t):
            d = digits(m.group(0))
            if len(d) >= 4:
                self._digit_strings.add(d)
            if d:
                self._tokens.add(d)

    def allow_caller(self, text: str):
        """What the caller says may be read back (phones, emails, ids) but never unlocks prices, months or timeframes."""
        t = _digit_words_to_digits(text)
        for m in _EMAIL.finditer(text):
            self.allow_email(m.group(0))
        for m in _DIGIT_RUN.finditer(t):
            d = digits(m.group(0))
            if len(d) >= 4:
                self._digit_strings.add(d)
            if d:
                self._tokens.add(d)

    # ---- checks ----
    def check(self, sentence: str) -> GuardResult:
        s = _digit_words_to_digits(sentence)
        low = s.lower()

        if _MONEY.search(low) and not _MONEY.search(self._corpus):
            return GuardResult(False, "money/pricing language")

        for m in _EMAIL.finditer(sentence):
            if m.group(0).lower() not in self._emails:
                return GuardResult(False, f"unknown email {m.group(0)}")

        # phone-like / long digit runs must come from known numbers
        for m in _DIGIT_RUN.finditer(s):
            d = digits(m.group(0))
            if len(d) >= 7:
                if not any(d in known or known in d for known in self._digit_strings):
                    return GuardResult(False, f"unknown number {d}")
            elif len(d) >= 3 and d not in self._tokens and not any(d in k for k in self._digit_strings):
                # 3-6 digit numbers (years, ids, prices) only if the source data had them
                return GuardResult(False, f"unknown figure {d}")

        for m in _TIMEFRAME.finditer(low):
            if m.group(0) not in self._corpus and not self._near(m):
                return GuardResult(False, f"unsourced timeframe '{m.group(0)}'")

        for mon in _MONTHS:
            if re.search(rf"\b{mon}\b", low) and mon != "may" and mon not in self._corpus:
                return GuardResult(False, f"unsourced date ({mon})")
        return GuardResult(True)

    def _near(self, m) -> bool:
        return False


_ABBR = {"mr", "mrs", "ms", "dr", "vs", "a.m", "p.m", "e.g", "i.e", "etc", "st", "no"}


class SentenceSplitter:
    """Incremental splitter so TTS can start on the first sentence while the LLM is still streaming."""
    _b = re.compile(r"([.!?…]+[\"')\]]*)(\s+)")

    def __init__(self):
        self.buf = ""

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out, pos = [], 0
        for m in self._b.finditer(self.buf):
            head = self.buf[pos:m.start()]
            last = (head.split() or [""])[-1].lower().rstrip(".")
            if last in _ABBR and m.group(1).startswith("."):
                continue
            out.append(self.buf[pos:m.end(1)].strip())
            pos = m.end()
        self.buf = self.buf[pos:]
        return [s for s in out if s]

    def flush(self) -> str | None:
        s, self.buf = self.buf.strip(), ""
        return s or None


_STAGE = re.compile(r"\*[^*]*\*|\([^)]*\)|\[[^\]]*\]")


def clean_for_speech(s: str) -> str:
    s = _STAGE.sub("", s)
    s = re.sub(r"[*_#`>•]+", "", s)
    s = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", s)
    return re.sub(r"\s+", " ", s).strip()
