"""Shared enums and event/action types. Controller is pure: events in, actions out."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Outcome(str, Enum):
    # never reached a person (pre-answer)
    NO_ANSWER = "no_answer"
    BUSY = "busy"
    INVALID_NUMBER = "invalid_number"      # lookup invalid, SIP 404/604, "not in service"
    DISCONNECTED = "disconnected"          # "has been disconnected"
    UNREACHABLE = "unreachable"            # "subscriber not available / switched off"
    FAILED = "failed"
    CANCELED = "canceled"
    # answered, no person
    VOICEMAIL_LEFT = "voicemail_left"
    VOICEMAIL_FULL = "voicemail_full"
    VOICEMAIL_DISABLED = "voicemail_disabled"
    IVR_FAILED = "ivr_failed"
    HOLD_TIMEOUT = "hold_timeout"
    SCREENED_OUT = "screened_out"
    REMOTE_HANGUP = "remote_hangup"
    # success: person on the line -> hand off to brain
    HUMAN_CONFIRMED = "human_confirmed"

    @property
    def retryable(self) -> bool:
        return self in {
            Outcome.NO_ANSWER, Outcome.BUSY, Outcome.UNREACHABLE, Outcome.FAILED,
            Outcome.IVR_FAILED, Outcome.HOLD_TIMEOUT, Outcome.REMOTE_HANGUP,
            Outcome.VOICEMAIL_FULL,
        }


class Label(str, Enum):
    """What a stretch of audio/transcript is."""
    CARRIER_INVALID = "carrier_invalid"        # not in service / cannot be completed as dialed
    CARRIER_DISCONNECTED = "carrier_disconnected"
    CARRIER_UNREACHABLE = "carrier_unreachable"
    MAILBOX_FULL = "mailbox_full"
    MAILBOX_DISABLED = "mailbox_disabled"
    SCREEN_PROMPT = "screen_prompt"            # "state your name"
    SCREEN_REJECT = "screen_reject"            # "person is not available"
    VOICEMAIL_GREETING = "voicemail_greeting"
    HOLD = "hold"
    IVR_INVALID_ENTRY = "ivr_invalid_entry"
    IVR_MENU = "ivr_menu"
    IVR_EXTENSION_PROMPT = "ivr_extension_prompt"
    HUMAN = "human"
    AMBIGUOUS_GREETING = "ambiguous_greeting"   # "thank you for calling X, how can I help" - person or recording
    UNKNOWN = "unknown"


class State(str, Enum):
    DIALING = "dialing"
    LISTENING = "listening"        # answered, nothing classified yet
    IVR = "ivr"
    HOLD = "hold"
    SCREENING = "screening"
    VOICEMAIL = "voicemail"        # greeting playing, waiting for beep
    LEAVING_MESSAGE = "leaving_message"
    HUMAN = "human"                # terminal for this phase -> handoff
    ENDED = "ended"


class AnsweredBy(str, Enum):
    HUMAN = "human"
    MACHINE_START = "machine_start"
    MACHINE_END_BEEP = "machine_end_beep"
    MACHINE_END_SILENCE = "machine_end_silence"
    MACHINE_END_OTHER = "machine_end_other"
    FAX = "fax"
    UNKNOWN = "unknown"

    @property
    def is_machine(self) -> bool:
        return self.value.startswith("machine") or self is AnsweredBy.FAX


# ---------- events (inputs) ----------
@dataclass
class Answered:
    t: float


@dataclass
class Amd:
    answered_by: AnsweredBy
    t: float


@dataclass
class Transcript:
    text: str
    t: float
    final: bool = True
    label: "Label | None" = None     # pre-computed (LLM fallback); else controller classifies


@dataclass
class Beep:
    t: float


@dataclass
class SpeechDone:
    """Our TTS finished playing."""
    t: float


@dataclass
class RemoteHangup:
    t: float


@dataclass
class Tick:
    t: float


# ---------- actions (outputs) ----------
@dataclass
class Speak:
    text: str


@dataclass
class Dtmf:
    digits: str


@dataclass
class Hangup:
    outcome: Outcome


@dataclass
class HandoffToBrain:
    """Human confirmed. Stage-2 wires this to the brain agent."""
    transcript_so_far: list[str] = field(default_factory=list)


Action = Speak | Dtmf | Hangup | HandoffToBrain


@dataclass
class CallRequest:
    to: str
    call_id: str
    extension: str | None = None            # for "enter extension" prompts
    context: dict = field(default_factory=dict)  # passed through to brain later
    ring_timeout_s: int | None = None       # per-call override of RING_TIMEOUT_S
