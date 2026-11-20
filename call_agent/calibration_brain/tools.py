"""Tools the LLM calls to change call state. Validation lives here so a bad call becomes a natural re-ask, not bad data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .state import (
    ASKING, CallState, EMAIL_RE, NEW_CONTACT, REGISTERED, UNREACHABLE, digits,
)

TOOL_SPECS = [
    {
        "name": "report_speaker",
        "description": (
            "Record who is on the line. Call it the moment you learn it, and again whenever it changes "
            "(transfer, put on hold then someone new picks up, 'I'm not the right person', etc)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["registered_person", "new_person_in_charge", "asking_for_registered_person", "unreachable"],
                    "description": (
                        "registered_person: they confirmed they ARE the person on file. "
                        "new_person_in_charge: they are a different person who now handles the meters. "
                        "asking_for_registered_person: still trying to reach them (gatekeeper, holding, being transferred). "
                        "unreachable: cannot reach anyone useful (wrong number, person gone, busy, refuses)."
                    ),
                },
            },
            "required": ["status"],
        },
    },
    {
        "name": "confirm_on_file",
        "description": "Registered person confirmed the phone or email we have on file is still correct.",
        "parameters": {
            "type": "object",
            "properties": {"field": {"type": "string", "enum": ["phone", "email"]}},
            "required": ["field"],
        },
    },
    {
        "name": "save_contact",
        "description": (
            "Save details the person gave you: a new/updated phone or email, or the full name of a new person in charge. "
            "Only pass fields they actually said, exactly as they confirmed them."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string"},
                "phone": {"type": "string", "description": "digits, any formatting"},
                "email": {"type": "string", "description": "written normally, e.g. sam.lee@acme.com"},
            },
        },
    },
    {
        "name": "contact_declined",
        "description": "Person does not want to share/update this detail. Use sparingly; try once gently first.",
        "parameters": {
            "type": "object",
            "properties": {"field": {"type": "string", "enum": ["name", "phone", "email"]}},
            "required": ["field"],
        },
    },
    {
        "name": "mark_reminder_delivered",
        "description": "Call right after you have told the person which meters are due for calibration and when.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "form_help_answer",
        "description": "Record whether the person wants guidance filling the calibration form.",
        "parameters": {
            "type": "object",
            "properties": {"wants_help": {"type": "boolean"}},
            "required": ["wants_help"],
        },
    },
    {
        "name": "form_progress",
        "description": "Track form guidance. status 'current' = you are now walking them through this step; 'done' = they finished it; 'all_done' = form finished or they want to stop.",
        "parameters": {
            "type": "object",
            "properties": {
                "step_id": {"type": "string", "description": "id from the FORM STEPS list (omit for all_done)"},
                "status": {"type": "string", "enum": ["current", "done", "all_done"]},
            },
            "required": ["status"],
        },
    },
    {
        "name": "log_guardrail_hit",
        "description": "Call whenever the person asks for something outside what you're allowed to discuss (price, scheduling, technical specs, anything not in your data).",
        "parameters": {
            "type": "object",
            "properties": {"topic": {"type": "string", "description": "short label e.g. 'pricing'"}},
            "required": ["topic"],
        },
    },
    {
        "name": "reroute_to_rep",
        "description": "Transfer the call to a human representative. Use when they ask for one, or after a guardrail refusal when they say yes to being connected. Say a short 'connecting you now' line in the same turn.",
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
    {
        "name": "end_call",
        "description": "Hang up after your goodbye. Say the goodbye in the same turn.",
        "parameters": {
            "type": "object",
            "properties": {
                "outcome": {
                    "type": "string",
                    "enum": ["completed", "not_available", "wrong_number", "busy_call_back_later", "declined_to_talk", "do_not_call", "other"],
                },
                "summary": {"type": "string", "description": "one line for the CRM"},
            },
            "required": ["outcome", "summary"],
        },
    },
]


@dataclass
class ToolResult:
    name: str
    ok: bool
    message: str = ""
    action: Optional[dict] = None  # {"type": "end_call"|"transfer", ...} executed by the telephony layer


def apply_tool(state: CallState, name: str, args: dict) -> ToolResult:
    a = args or {}
    r = lambda ok=True, msg="", action=None: ToolResult(name, ok, msg, action)  # noqa: E731

    if name == "report_speaker":
        status = a.get("status")
        prev = state.speaker
        if status == prev:
            return r()
        # identity changed between two real people -> earlier captures belong to someone else
        if prev in (REGISTERED, NEW_CONTACT) and status in (REGISTERED, NEW_CONTACT):
            state.reset_for_new_person("The person on the line changed mid-call.")
        state.speaker = status
        return r()

    if name == "confirm_on_file":
        if state.speaker != REGISTERED:
            return r(False, "Only the registered person can confirm on-file details. Do not read them to anyone else.")
        if a.get("field") == "phone":
            state.phone_done = True
        elif a.get("field") == "email":
            state.email_done = True
        return r()

    if name == "save_contact":
        if state.speaker not in (REGISTERED, NEW_CONTACT):
            return r(False, "You don't know who you're speaking to yet; establish that first.")
        if a.get("full_name"):
            if state.speaker == NEW_CONTACT:
                state.name = a["full_name"].strip()
                state.name_done = True
        if a.get("phone"):
            d = digits(a["phone"])
            if len(d) < 10:
                return r(False, f"That phone number has only {len(d)} digits. Politely ask them to repeat the full number, area code included.")
            state.phone, state.phone_done = a["phone"].strip(), True
            state.phone_changed = state.speaker == REGISTERED and d != digits(state.call.registered_phone)
        if a.get("email"):
            e = a["email"].strip().lower()
            if not EMAIL_RE.match(e):
                return r(False, "That email doesn't look complete. Ask them to say it again slowly, or spell the part before the @.")
            state.email, state.email_done = e, True
            state.email_changed = state.speaker == REGISTERED and e != state.call.registered_email.lower()
        return r()

    if name == "contact_declined":
        f = a.get("field")
        state.declined.add(f)
        if f == "phone":
            state.phone_done = True
        elif f == "email":
            state.email_done = True
        elif f == "name":
            state.name_done = True
        return r()

    if name == "mark_reminder_delivered":
        if state.speaker not in (REGISTERED, NEW_CONTACT):
            return r(False, "Reminder can only go to the registered person or the new person in charge.")
        state.reminder_delivered = True
        return r()

    if name == "form_help_answer":
        state.form_help_wanted = bool(a.get("wants_help"))
        if not state.form_help_wanted:
            state.form_finished = True
        return r()

    if name == "form_progress":
        st, sid = a.get("status"), a.get("step_id")
        if st == "all_done":
            state.form_finished, state.form_current = True, None
        elif sid:
            if st == "current":
                state.form_current = sid
            elif st == "done":
                if sid not in state.form_steps_done:
                    state.form_steps_done.append(sid)
                state.form_current = None
        return r()

    if name == "log_guardrail_hit":
        state.guardrail_hits += 1
        state.guardrail_topics.append(str(a.get("topic", "unspecified"))[:60])
        return r()

    if name == "reroute_to_rep":
        state.reroute_requested = True
        state.outcome = "rerouted_to_rep"
        state.outcome_summary = str(a.get("reason", ""))[:200]
        return r(action={"type": "transfer", "reason": state.outcome_summary})

    if name == "end_call":
        outcome = a.get("outcome", "other")
        if outcome == "completed":
            missing = state.contact_missing()
            if state.speaker not in (REGISTERED, NEW_CONTACT):
                return r(False, "You can't mark completed: you never reached the right person. Use not_available / wrong_number instead.")
            if not state.reminder_delivered:
                return r(False, "You haven't delivered the calibration reminder yet. Do that before ending.")
            if missing:
                return r(False, f"Still missing contact details: {', '.join(missing)}. Ask for them (or confirm they decline) before ending.")
            if state.form_help_wanted is None:
                return r(False, "You haven't asked whether they want help with the form yet.")
        state.outcome, state.outcome_summary, state.ended = outcome, str(a.get("summary", ""))[:300], True
        if outcome in ("not_available", "wrong_number", "busy_call_back_later", "declined_to_talk", "do_not_call"):
            state.speaker = UNREACHABLE if state.speaker in ("unknown", ASKING) else state.speaker
        return r(action={"type": "end_call", "outcome": outcome})

    return r(False, f"Unknown tool {name}")
