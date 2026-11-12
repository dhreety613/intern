"""Detectors fed by Twilio webhooks (call progress, AMD) and by the transcript."""
from __future__ import annotations

import re

from ..classifier import norm
from ..classifier import classify
from ..ivr import parse_options
from ..models import AnsweredBy, Label
from .signals import Sig, Signal

_PROGRESS = {"ringing": Sig.RINGING, "in-progress": Sig.ANSWERED, "busy": Sig.BUSY, "no-answer": Sig.NO_ANSWER,
             "canceled": Sig.CANCELED, "completed": Sig.COMPLETED}
_INVALID_SIP = {"404", "410", "484", "485", "604"}
_UNREACH_SIP = {"480", "486", "503"}
_INVALID_ERR = {"13223", "13224", "13225", "13226", "21211", "21214", "21217", "21401", "32204"}


class ProgressDetector:
    """Twilio status callback -> signal. Ringing / answered / busy / no-answer / invalid / failed / hangup."""
    source = "progress"

    def feed(self, form: dict, t: float) -> list[Signal]:
        status = (form.get("CallStatus") or "").lower()
        sip, err = str(form.get("SipResponseCode") or ""), str(form.get("ErrorCode") or "")
        if status == "failed":
            if sip in _INVALID_SIP or err in _INVALID_ERR:
                kind = Sig.INVALID_NUMBER
            elif sip in _UNREACH_SIP:
                kind = Sig.UNREACHABLE_SIP
            else:
                kind = Sig.FAILED
        else:
            kind = _PROGRESS.get(status)
        return [Signal(self.source, kind, t, 1.0, {"status": status, "sip": sip})] if kind else []


_AMD = {AnsweredBy.HUMAN: Sig.AMD_HUMAN, AnsweredBy.MACHINE_START: Sig.AMD_MACHINE_START,
        AnsweredBy.MACHINE_END_BEEP: Sig.AMD_END_BEEP, AnsweredBy.MACHINE_END_SILENCE: Sig.AMD_END_SILENCE,
        AnsweredBy.MACHINE_END_OTHER: Sig.AMD_END_OTHER, AnsweredBy.FAX: Sig.AMD_FAX,
        AnsweredBy.UNKNOWN: Sig.AMD_UNKNOWN}


class AmdDetector:
    source = "amd"

    def feed(self, answered_by: AnsweredBy, t: float) -> list[Signal]:
        return [Signal(self.source, _AMD[answered_by], t)]


_TONE = re.compile(r"\b(tone|beep)\b")


class TextDetector:
    """Transcript -> Label (rules, optional LLM fallback applied by the hub) + parsed IVR options."""
    source = "text"

    def feed(self, text: str, t: float, final: bool, label: Label | None = None) -> list[Signal]:
        if not text.strip():
            return []
        if not final:
            return [Signal(self.source, Sig.TEXT_PARTIAL, t, 1.0, {"text": text})]
        label = label or classify(text)
        return [Signal(self.source, Sig.TEXT_LABEL, t, 1.0, {
            "label": label, "text": text, "options": parse_options(text),
            "tone": bool(_TONE.search(text.lower())), "words": len(norm(text).split())})]
