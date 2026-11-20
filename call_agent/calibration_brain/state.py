"""Call input + mutable call state. State is the source of truth; the LLM only proposes changes via tools."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def spoken_date(s: str) -> str:
    """'2026-12-05' -> 'December 5th, 2026'. Anything unparseable passes through untouched."""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            d = datetime.strptime(s.strip(), fmt)
            return f"{d.strftime('%B')} {_ordinal(d.day)}, {d.year}"
        except ValueError:
            continue
    return s


def digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")


@dataclass
class Meter:
    name: str
    due_date: str  # ISO or already spoken; normalised on init

    def __post_init__(self):
        self.due_date = spoken_date(self.due_date)


@dataclass
class FormStep:
    id: str
    title: str
    instruction: str


@dataclass
class CallInput:
    registered_name: str
    registered_phone: str
    registered_email: str
    meters: list[Meter]
    form_steps: list[FormStep] = field(default_factory=list)
    today: str = field(default_factory=lambda: spoken_date(date.today().isoformat()))


# speaker status values
UNKNOWN, ASKING, REGISTERED, NEW_CONTACT, UNREACHABLE = (
    "unknown", "asking_for_registered_person", "registered_person", "new_person_in_charge", "unreachable",
)

# phases (derived, never set by the LLM)
P_OPENING, P_CONTACT, P_REMINDER, P_FORM_OFFER, P_FORM_GUIDE, P_WRAP, P_ENDED = (
    "opening", "contact_details", "reminder", "form_offer", "form_guide", "wrap_up", "ended",
)


@dataclass
class CallState:
    call: CallInput
    speaker: str = UNKNOWN
    # contact capture
    name: Optional[str] = None            # only used for new contact
    phone: Optional[str] = None           # captured new/updated phone
    email: Optional[str] = None
    phone_done: bool = False              # confirmed as-is OR updated OR declined
    email_done: bool = False
    name_done: bool = False
    phone_changed: bool = False
    email_changed: bool = False
    declined: set = field(default_factory=set)
    # reminder + form
    reminder_delivered: bool = False
    form_help_wanted: Optional[bool] = None
    form_steps_done: list[str] = field(default_factory=list)
    form_current: Optional[str] = None
    form_finished: bool = False
    # guardrails / routing
    guardrail_hits: int = 0
    guardrail_topics: list[str] = field(default_factory=list)
    reroute_requested: bool = False
    # end
    outcome: Optional[str] = None
    outcome_summary: str = ""
    ended: bool = False
    # telephony hints
    reentry_note: Optional[str] = None
    turns: int = 0

    # ---- derived ----
    def contact_missing(self) -> list[str]:
        if self.speaker == REGISTERED:
            need = [("phone", self.phone_done), ("email", self.email_done)]
        elif self.speaker == NEW_CONTACT:
            need = [("name", self.name_done), ("phone", self.phone_done), ("email", self.email_done)]
        else:
            return []
        return [n for n, done in need if not done]

    def phase(self) -> str:
        if self.ended:
            return P_ENDED
        if self.speaker in (UNKNOWN, ASKING, UNREACHABLE):
            return P_OPENING
        if self.contact_missing():
            return P_CONTACT
        if not self.reminder_delivered:
            return P_REMINDER
        if self.form_help_wanted is None:
            return P_FORM_OFFER
        if self.form_help_wanted and not self.form_finished:
            return P_FORM_GUIDE
        return P_WRAP

    def reset_for_new_person(self, note: str):
        """A transfer / hold-resume / new pickup happened. Nothing learned so far applies to whoever is on now."""
        self.speaker = UNKNOWN
        self.name = self.phone = self.email = None
        self.phone_done = self.email_done = self.name_done = False
        self.phone_changed = self.email_changed = False
        self.declined.clear()
        self.reminder_delivered = False
        self.form_help_wanted = None
        self.form_steps_done.clear()
        self.form_current = None
        self.form_finished = False
        self.reentry_note = note

    def result(self) -> dict:
        """Everything the post-call job / CRM writer needs."""
        reg = self.speaker == REGISTERED
        return {
            "outcome": self.outcome,
            "summary": self.outcome_summary,
            "speaker": self.speaker,
            "contact": {
                "name": self.call.registered_name if reg else self.name,
                "phone": self.phone or (self.call.registered_phone if reg and "phone" not in self.declined else None),
                "email": self.email or (self.call.registered_email if reg and "email" not in self.declined else None),
                "phone_changed": self.phone_changed,
                "email_changed": self.email_changed,
                "is_new_contact": self.speaker == NEW_CONTACT,
                "declined": sorted(self.declined),
            },
            "reminder_delivered": self.reminder_delivered,
            "form_help_wanted": self.form_help_wanted,
            "form_finished": self.form_finished,
            "form_steps_done": list(self.form_steps_done),
            "guardrail_hits": self.guardrail_hits,
            "guardrail_topics": list(self.guardrail_topics),
            "reroute_requested": self.reroute_requested,
        }
