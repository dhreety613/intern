"""Signals = raw detector outputs. Verdicts = fused decisions the controller acts on."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Sig(str, Enum):
    # call progress (Twilio status callbacks)
    RINGING = "ringing"
    ANSWERED = "answered"
    BUSY = "busy"
    NO_ANSWER = "no_answer"
    INVALID_NUMBER = "invalid_number"
    FAILED = "failed"
    CANCELED = "canceled"
    UNREACHABLE_SIP = "unreachable_sip"
    COMPLETED = "completed"            # far end / Twilio ended the call
    # Twilio AMD
    AMD_HUMAN = "amd_human"
    AMD_MACHINE_START = "amd_machine_start"
    AMD_END_BEEP = "amd_end_beep"
    AMD_END_SILENCE = "amd_end_silence"
    AMD_END_OTHER = "amd_end_other"
    AMD_FAX = "amd_fax"
    AMD_UNKNOWN = "amd_unknown"
    # transcript
    TEXT_PARTIAL = "text_partial"
    TEXT_LABEL = "text_label"          # data: label, text, options, tone, words
    # audio events
    BEEP_END = "beep_end"              # data: freq, dur
    TONE_BUSY = "tone_busy"            # data: cadence = busy | reorder
    TONE_FAX = "tone_fax"
    MUSIC_ON = "music_on"
    MUSIC_OFF = "music_off"
    # voice activity
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"          # data: dur
    RHYTHM = "rhythm"                  # data: verdict = human | machine
    # internal
    TICK = "tick"


@dataclass
class Signal:
    src: str
    kind: Sig
    t: float
    conf: float = 1.0
    data: dict = field(default_factory=dict)


class VK(str, Enum):
    PRE_ANSWER = "pre_answer"
    ANSWERED = "answered"
    REMOTE_HANGUP = "remote_hangup"
    FAX = "fax"
    BUSY_TONE = "busy_tone"
    CARRIER = "carrier"                # data: outcome
    MAILBOX = "mailbox"                # data: outcome
    HUMAN = "human"
    PROBE_DUE = "probe_due"            # silent line: say "Hello?"
    AMBIGUOUS_GREETING = "ambiguous_greeting"   # say intro, see who answers
    VM_GREETING = "vm_greeting"
    VM_BEEP_READY = "vm_beep_ready"    # greeting over: speak now
    SCREEN_PROMPT = "screen_prompt"
    SCREEN_SPEAK_READY = "screen_speak_ready"
    SCREEN_REJECT = "screen_reject"
    IVR_MENU_READY = "ivr_menu_ready"  # data: text, options, repeat
    IVR_INVALID = "ivr_invalid"
    IVR_EXTENSION = "ivr_extension"
    HOLD = "hold"
    DEAD_AIR = "dead_air"              # data: amd_human


@dataclass
class Verdict:
    kind: VK
    t: float
    conf: float = 1.0
    evidence: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)
