"""STT / TTS interfaces + adapters, wired the way call_agent/app/speech_services.py is (that path is proven live):
ElevenLabs TTS (ulaw_8000, REST /stream) and ElevenLabs Scribe for STT. GoogleSTT stays as STT_ENGINE=google."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import struct
from typing import AsyncIterator, Protocol

import httpx
import numpy as np

from .tones import mulaw_decode

log = logging.getLogger("telephony.speech")


class TTS(Protocol):
    async def synth(self, text: str) -> bytes:
        """Return 8 kHz mu-law audio."""


class STT(Protocol):
    async def start(self) -> None: ...
    def feed(self, mulaw: bytes) -> None: ...
    def results(self) -> AsyncIterator[tuple[str, bool]]:
        """Yield (text, is_final)."""
    async def close(self) -> None: ...


class ElevenLabsTTS:
    """Same wiring as call_agent: REST /stream endpoint (the SDK surface changes between releases), ulaw_8000
    straight into Twilio, same env vars and defaults. Caches repeated strings (voicemail / screening /
    'Hello?') so they cost no latency after warm()."""

    API_BASE = "https://api.elevenlabs.io/v1"

    def __init__(self, api_key: str | None = None, voice_id: str | None = None, model_id: str | None = None):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not set")
        self.voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")   # call_agent default
        self.model_id = model_id or os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
        self.output_format = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "ulaw_8000")
        self.voice_settings = {"stability": float(os.getenv("ELEVENLABS_STABILITY", "0.45")),
                               "similarity_boost": float(os.getenv("ELEVENLABS_SIMILARITY", "0.75")),
                               "style": float(os.getenv("ELEVENLABS_STYLE", "0"))}
        self._cache: dict[str, bytes] = {}
        self._http = httpx.AsyncClient(timeout=30)

    async def synth(self, text: str) -> bytes:
        if not text.strip():
            return b""
        if text in self._cache:
            return self._cache[text]
        async with self._http.stream(
                "POST", f"{self.API_BASE}/text-to-speech/{self.voice_id}/stream",
                params={"output_format": self.output_format},
                headers={"xi-api-key": self.api_key, "content-type": "application/json"},
                json={"text": text, "model_id": self.model_id, "voice_settings": self.voice_settings}) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode("utf-8", "replace")
                log.error("ElevenLabs TTS HTTP %s (voice=%s model=%s): %s", r.status_code, self.voice_id,
                          self.model_id, body[:500])
                raise RuntimeError(f"ElevenLabs TTS {r.status_code}: {body[:300]}")
            audio = b"".join([c async for c in r.aiter_bytes() if c])
        self._cache[text] = audio
        return audio

    async def warm(self, texts: list[str]) -> None:
        await asyncio.gather(*(self.synth(t) for t in texts), return_exceptions=True)


def _wrap_mulaw_as_wav(mulaw: bytes, sample_rate: int = 8000) -> bytes:
    """Minimal WAV container (WAVE_FORMAT_MULAW = 7) around raw Twilio audio, as call_agent does for Scribe."""
    fmt = struct.pack("<HHIIHHH", 7, 1, sample_rate, sample_rate, 1, 8, 0)
    riff = 4 + (8 + len(fmt)) + (8 + len(mulaw))
    return (b"RIFF" + struct.pack("<I", riff) + b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(mulaw)) + mulaw)


class ElevenLabsScribeSTT:
    """ElevenLabs Scribe as the transcriber, the engine call_agent uses (STT_ENGINE=elevenlabs).
    Scribe is a batch API, so the caller audio is cut into utterances with an energy VAD (constants from
    call_agent/app/media_stream.py) and each utterance is transcribed in one call, in order. Finals only.
    Audio-event tags ("[phone ringing]") are stripped so they never reach the detectors."""

    API_URL = "https://api.elevenlabs.io/v1/speech-to-text"
    FRAME = 160                    # 20 ms of 8 kHz mu-law
    AMPLITUDE = 350                # decoded peak counted as voice (call_agent VAD_AMPLITUDE_THRESHOLD)
    MIN_SPEECH_MS = 300
    HANGOVER_MS = 700              # pause that ends an utterance
    MAX_SEGMENT_MS = 12000         # a long IVR menu is cut here at the latest
    PREROLL_MS = 300               # audio kept before voice onset so the first word is not clipped

    def __init__(self, api_key: str | None = None, model_id: str | None = None,
                 client: httpx.AsyncClient | None = None):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not set")
        self.model_id = model_id or os.getenv("ELEVENLABS_STT_MODEL_ID", "scribe_v1")
        self._http = client or httpx.AsyncClient(timeout=30)
        self._segs: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._out: asyncio.Queue[tuple[str, bool] | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._tail = bytearray()          # sub-frame remainder
        self._pre = bytearray()           # rolling pre-roll while idle
        self._seg = bytearray()
        self._voiced = self._quiet = 0    # ms
        self._in_speech = False

    async def start(self) -> None:
        self._task = asyncio.create_task(self._worker(), name="scribe-stt")

    def feed(self, mulaw: bytes) -> None:
        self._tail += mulaw
        while len(self._tail) >= self.FRAME:
            frame = bytes(self._tail[:self.FRAME])
            del self._tail[:self.FRAME]
            self._frame(frame)

    def _frame(self, frame: bytes) -> None:
        loud = int(np.abs(mulaw_decode(frame).astype(np.int32)).max()) > self.AMPLITUDE
        if not self._in_speech:
            self._pre += frame
            del self._pre[:max(0, len(self._pre) - self.PREROLL_MS * 8)]
            self._voiced = self._voiced + 20 if loud else 0
            if self._voiced >= 40:                         # two voiced frames in a row open a segment
                self._in_speech, self._quiet = True, 0
                self._seg = bytearray(self._pre)
            return
        self._seg += frame
        if loud:
            self._voiced += 20
            self._quiet = 0
        else:
            self._quiet += 20
        if self._quiet >= self.HANGOVER_MS or len(self._seg) // 8 >= self.MAX_SEGMENT_MS:
            if self._voiced >= self.MIN_SPEECH_MS:
                self._segs.put_nowait(bytes(self._seg))
            self._seg.clear()
            self._pre.clear()
            self._in_speech, self._voiced, self._quiet = False, 0, 0

    async def results(self):
        while True:
            item = await self._out.get()
            if item is None:
                return
            yield item

    async def close(self) -> None:
        if self._in_speech and self._voiced >= self.MIN_SPEECH_MS:
            self._segs.put_nowait(bytes(self._seg))
        self._segs.put_nowait(None)
        if self._task:
            await asyncio.wait([self._task], timeout=8)
        self._out.put_nowait(None)

    async def _worker(self) -> None:
        while True:
            seg = await self._segs.get()
            if seg is None:
                return
            try:
                text = await self._transcribe(seg)
            except Exception:
                log.exception("scribe transcription failed")
                continue
            if text:
                self._out.put_nowait((text, True))

    async def _transcribe(self, mulaw: bytes) -> str:
        r = await self._http.post(
            self.API_URL, headers={"xi-api-key": self.api_key},
            files={"file": ("audio.wav", _wrap_mulaw_as_wav(mulaw), "audio/wav")},
            data={"model_id": self.model_id, "tag_audio_events": "false"})
        if r.status_code != 200:
            raise RuntimeError(f"ElevenLabs STT {r.status_code}: {r.text[:300]}")
        return re.sub(r"\[[^\]]*\]", "", (r.json().get("text") or "")).strip()


class GoogleSTT:
    """Google Cloud streaming STT, mu-law 8k, phone_call model. Re-opens the stream before
    Google's ~5 min cap so long holds keep working."""

    HINTS = ["press", "extension", "operator", "representative", "voicemail", "after the tone",
             "mailbox", "hold", "menu", "star", "pound"]

    def __init__(self, language: str = "en-US", restart_after_s: float = 270):
        self.language, self.restart_after_s = language, restart_after_s
        self._in: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._out: asyncio.Queue[tuple[str, bool] | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._closed = False

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    def feed(self, mulaw: bytes) -> None:
        self._in.put_nowait(mulaw)

    async def results(self):
        while True:
            item = await self._out.get()
            if item is None:
                return
            yield item

    async def close(self) -> None:
        self._closed = True
        self._in.put_nowait(None)
        if self._task:
            await asyncio.wait([self._task], timeout=2)
        self._out.put_nowait(None)

    async def _run(self) -> None:
        from google.cloud import speech_v1 as speech   # lazy: only needed when STT_ENGINE=google
        client = speech.SpeechAsyncClient()
        cfg = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.MULAW, sample_rate_hertz=8000,
                language_code=self.language, model="phone_call", use_enhanced=True,
                enable_automatic_punctuation=True,
                speech_contexts=[speech.SpeechContext(phrases=self.HINTS, boost=10.0)]),
            interim_results=True)
        loop = asyncio.get_running_loop()
        while not self._closed:
            async def reqs():
                yield speech.StreamingRecognizeRequest(streaming_config=cfg)
                t0 = loop.time()
                while loop.time() - t0 < self.restart_after_s:
                    try:
                        chunk = await asyncio.wait_for(self._in.get(), 1.0)
                    except asyncio.TimeoutError:
                        continue
                    if chunk is None:
                        return
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
            try:
                stream = await client.streaming_recognize(requests=reqs())
                async for resp in stream:
                    for r in resp.results:
                        if r.alternatives:
                            self._out.put_nowait((r.alternatives[0].transcript, r.is_final))
            except Exception:
                log.exception("stt stream error; reopening")
                await asyncio.sleep(0.2)


def make_stt_factory(engine: str | None = None):
    """STT_ENGINE=elevenlabs (default, as call_agent) | google."""
    engine = (engine or os.getenv("STT_ENGINE", "elevenlabs")).lower()
    if engine == "google":
        return GoogleSTT
    if engine == "elevenlabs":
        return ElevenLabsScribeSTT
    raise ValueError(f"unknown STT_ENGINE {engine!r}")
