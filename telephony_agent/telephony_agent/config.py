from __future__ import annotations

import os
from dataclasses import dataclass, field


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass
class Settings:
    # Twilio
    account_sid: str = field(default_factory=lambda: os.getenv("TWILIO_ACCOUNT_SID", ""))
    auth_token: str = field(default_factory=lambda: os.getenv("TWILIO_AUTH_TOKEN", ""))
    from_number: str = field(default_factory=lambda: os.getenv("TWILIO_FROM_NUMBER", ""))
    public_base_url: str = field(default_factory=lambda: os.getenv("PUBLIC_BASE_URL", "").rstrip("/"))

    # dialing
    ring_timeout_s: int = int(os.getenv("RING_TIMEOUT_S", 25))
    amd_timeout_s: int = int(os.getenv("AMD_TIMEOUT_S", 30))
    block_landline_voip_check: bool = False
    lookup_enabled: bool = os.getenv("LOOKUP_ENABLED", "1") == "1"

    # voicemail
    callback_number: str = field(default_factory=lambda: os.getenv("CALLBACK_NUMBER", ""))
    voicemail_max_wait_s: float = _f("VM_MAX_WAIT_S", 45)      # greeting + beep
    voicemail_silence_fallback_s: float = _f("VM_SILENCE_FALLBACK_S", 1.5)

    # IVR
    ivr_settle_s: float = _f("IVR_SETTLE_S", 1.1)    # silence after menu before we press
    ivr_retries: int = int(os.getenv("IVR_RETRIES", 2))
    llm_key_timeout_s: float = float(os.getenv("LLM_KEY_TIMEOUT_S", 2.0))
    ivr_max_menus: int = int(os.getenv("IVR_MAX_MENUS", 5))
    ivr_stuck_timeout_s: float = _f("IVR_STUCK_TIMEOUT_S", 25)

    # hold
    hold_max_s: float = _f("HOLD_MAX_S", 600)

    # screening
    screen_wait_s: float = _f("SCREEN_WAIT_S", 2.0)

    # human probe: after answer w/ AMD=human and total silence
    human_probe_after_s: float = _f("HUMAN_PROBE_AFTER_S", 2.5)
    listen_timeout_s: float = _f("LISTEN_TIMEOUT_S", 20)

    # fusion
    human_confirm_threshold: float = _f("HUMAN_CONFIRM_THRESHOLD", 0.6)
    human_fallback_s: float = _f("HUMAN_FALLBACK_S", 1.2)
    music_hold_s: float = _f("MUSIC_HOLD_S", 4.0)

    # dtmf: "twilio" (default) = Twilio generates the tones: REST <Play digits> + stream reconnect;
    #       "inband" = our own tones over the media stream (did NOT register on a live IVR)
    dtmf_mode: str = os.getenv("DTMF_MODE", "twilio")
    reconnect_window_s: float = _f("RECONNECT_WINDOW_S", 8)

    # recording (call_agent records every call, dual channel, for review)
    record_calls: bool = os.getenv("RECORD_CALLS", "1") == "1"

    # brain handoff (stage 2 placeholder): keep the line open this long after a person is confirmed, then hang up
    handoff_stub_hold_s: float = _f("HANDOFF_STUB_HOLD_S", 6)

    # speaking
    company: str = "Tensitron"
    agent_name: str = "Anne"

    def voicemail_text(self) -> str:
        return (f"Hey, I am {self.agent_name} from {self.company}, contacting to confirm your contact details "
                "registered with us and also remind you of an upcoming calibration. "
                "Please get back to us once you are available. Thank you.")

    def screening_text(self) -> str:
        return f"Hi, this is {self.agent_name} from {self.company}, calling about your calibration reminder."


def _spell_phone(n: str) -> str:
    digits = [c for c in n if c.isdigit()]
    return ", ".join("".join(digits[i:i + 3]) for i in range(0, len(digits), 3)) if digits else n
