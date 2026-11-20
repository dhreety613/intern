"""Runtime config. Everything overridable via env so nothing is hardcoded in code paths."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    agent_name: str
    company: str
    # Fallback contacts offered whenever a guardrail is hit
    support_phone: str
    support_email: str
    rep_transfer_number: str  # live transfer target for reroute_to_rep
    # LLM
    gemini_model: str
    temperature: float
    # ElevenLabs
    eleven_voice_id: str
    eleven_model: str
    eleven_stability: float
    eleven_similarity: float
    eleven_style: float
    eleven_speed: float
    eleven_output_format: str  # ulaw_8000 = Twilio-native, no transcoding


def load_config() -> Config:
    e = os.getenv
    return Config(
        agent_name=e("AGENT_NAME", "Anne"),
        company=e("COMPANY_NAME", "Tensitron"),
        support_phone=e("SUPPORT_PHONE", "+17202301616"),
        support_email=e("SUPPORT_EMAIL", "service@tensitron.com"),
        rep_transfer_number=e("REP_TRANSFER_NUMBER", "+17202301616"),
        gemini_model=e("GEMINI_MODEL", "gemini-3.5-flash"),
        temperature=float(e("LLM_TEMPERATURE", "0.7")),
        # Default = voice + settings of the live "Anne Tensitron v3" ElevenLabs agent.
        eleven_voice_id=e("ELEVEN_VOICE_ID", "ys3XeJJA4ArWMhRpcX1D"),
        eleven_model=e("ELEVEN_MODEL", "eleven_v3"),
        # Lower stability = more expressive pitch movement; style adds warmth; both cost a little latency.
        eleven_stability=float(e("ELEVEN_STABILITY", "0.5")),
        eleven_similarity=float(e("ELEVEN_SIMILARITY", "0.6")),
        eleven_style=float(e("ELEVEN_STYLE", "0.0")),
        eleven_speed=float(e("ELEVEN_SPEED", "1.0")),
        eleven_output_format=e("ELEVEN_OUTPUT_FORMAT", "ulaw_8000"),
    )
