"""ElevenLabs TTS (streaming) + making text *speakable* so phone numbers/emails sound human, not robotic."""
from __future__ import annotations

import os
import re
from typing import AsyncIterator

import httpx

from .config import Config

_D = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _say_digits(d: str) -> str:
    return " ".join(_D[int(c)] for c in d)


def _phone_to_words(m: re.Match) -> str:
    d = re.sub(r"\D", "", m.group(0))
    if len(d) == 11 and d[0] == "1":
        d = d[1:]
    if len(d) == 10:  # 737, 221, 2163 -> natural phone cadence via commas
        return f"{_say_digits(d[:3])}, {_say_digits(d[3:6])}, {_say_digits(d[6:])}"
    if len(d) >= 7:
        return ", ".join(_say_digits(d[i:i + 3]) for i in range(0, len(d), 3))
    return m.group(0)


_COMMON_DOMAINS = {"gmail", "yahoo", "outlook", "hotmail", "icloud", "aol", "live", "msn", "protonmail"}
_DOMAIN_WORDS = {"instruments", "systems", "tech", "technologies", "industries", "controls", "labs", "group", "solutions", "products", "manufacturing", "engineering", "sales"}
_COMMON_WORDS = {"sales", "service", "support", "info", "admin", "office", "contact", "help", "billing", "accounts",
                 "quality", "orders", "hr"}


def _spell(tok: str) -> str:
    return " ".join(_D[int(c)] if c.isdigit() else c.upper() for c in tok)


def _local_to_words(local: str) -> str:
    # separators first (v3: r.hart -> "R dot hart")
    parts = re.split(r"([._+-])", local)
    sep = {".": " dot ", "_": " underscore ", "-": " dash ", "+": " plus "}
    out = []
    for p in parts:
        if p in sep:
            out.append(sep[p])
        elif not p:
            continue
        elif any(c.isdigit() for c in p):
            out.append(_spell(p))          # letters + digits: one by one, never "nineteen seventy nine"
        elif p.lower() in _COMMON_WORDS or (4 <= len(p) <= 10 and re.search(r"[aeiou]", p.lower())):
            out.append(p)                   # plain word / first name: say it naturally
        else:
            out.append(_spell(p))           # initials / odd strings: letter by letter
    return "".join(out)


def _domain_to_words(domain: str) -> str:
    labels = domain.lower().split(".")
    spoken = []
    for i, lab in enumerate(labels):
        if i == len(labels) - 1:
            spoken.append(lab)
        elif lab in _COMMON_DOMAINS:
            spoken.append(lab)
        elif lab.startswith("ktc"):          # v3 rule: always "K T C", then the readable rest
            rest = lab[3:]
            spoken.append("K T C" + (" " + rest if rest else ""))
        else:
            for n in (3, 2):
                if lab[n:] in _DOMAIN_WORDS:
                    spoken.append(f"{_spell(lab[:n])} {lab[n:]}")
                    break
            else:
                spoken.append(lab)
    return " dot ".join(spoken)


def _email_to_words(m: re.Match) -> str:
    local, _, domain = m.group(0).partition("@")
    return re.sub(r"\s+", " ", f"{_local_to_words(local)} at {_domain_to_words(domain)}").strip()


_URL = re.compile(r"\b(?:www\.)?([a-z0-9-]+)\.(com|net|org|io)\b(?!/)", re.I)


_PHONE = re.compile(r"(?<!\d)\+?\d[\d\s().-]{7,}\d(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def speakable(text: str) -> str:
    text = _EMAIL.sub(_email_to_words, text)
    text = _URL.sub(lambda m: f"{m.group(1)} dot {m.group(2)}", text)
    text = _PHONE.sub(_phone_to_words, text)
    text = re.sub(r"\s*@\s*", " at ", text)   # spaced-out emails the model wrote letter by letter ("d r i t i @ gmail dot com")
    text = text.replace("&", " and ").replace("—", ", ").replace(" - ", ", ")
    return re.sub(r"\s+", " ", text).strip()


class ElevenLabsVoice:
    def __init__(self, cfg: Config, api_key: str | None = None):
        self.cfg = cfg
        self.key = api_key or os.environ["ELEVENLABS_API_KEY"]
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0), base_url="https://api.elevenlabs.io")
        self._prev = ""  # spoken context -> continuous prosody/pitch across sentences

    def reset_context(self):
        self._prev = ""

    async def synth(self, text: str, next_text: str | None = None) -> AsyncIterator[bytes]:
        """Yield audio chunks (default ulaw_8000, Twilio-native) for one sentence."""
        c = self.cfg
        spoken = speakable(text)
        body = {
            "text": spoken,
            "model_id": c.eleven_model,
            "voice_settings": {
                "stability": c.eleven_stability,
                "similarity_boost": c.eleven_similarity,
                "style": c.eleven_style,
                "use_speaker_boost": True,
                "speed": c.eleven_speed,
            },
        }
        ctx_ok = not c.eleven_model.startswith("eleven_v3")  # v3 rejects previous/next_text
        if self._prev and ctx_ok:
            body["previous_text"] = self._prev[-250:]
        if next_text and ctx_ok:
            body["next_text"] = speakable(next_text)
        async with self.http.stream(
            "POST", f"/v1/text-to-speech/{c.eleven_voice_id}/stream",
            params={"output_format": c.eleven_output_format},
            headers={"xi-api-key": self.key, "Content-Type": "application/json"},
            json=body,
        ) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes(1600):
                if chunk:
                    yield chunk
        self._prev = (self._prev + " " + spoken).strip()

    async def aclose(self):
        await self.http.aclose()


def ulaw_frames(audio: bytes, frame: int = 160) -> list[bytes]:
    """20ms Twilio media frames (8kHz mulaw = 160 bytes)."""
    return [audio[i:i + frame] for i in range(0, len(audio), frame)]
