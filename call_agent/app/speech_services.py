import os
import json
import re
import struct
from typing import AsyncIterator
import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)


# ============================================================
# SPEECH-TO-TEXT (Google Cloud)
# ============================================================

class GoogleSTT:
    """
    Streaming Speech-to-Text using Google Cloud. Not used by default
    (STT_ENGINE=elevenlabs is the default - see get_stt_engine below);
    kept as a fallback path. google-cloud-speech pins protobuf<5, which
    conflicts with newer dependencies (e.g. pipecat-ai wants protobuf>=5),
    so it's intentionally not a hard requirement - only imported if this
    class is actually instantiated.
    """

    def __init__(self):
        from google.cloud import speech_v1

        self._speech_v1 = speech_v1

        # Requires GOOGLE_APPLICATION_CREDENTIALS env var
        # pointing to your service account JSON key
        self.client = speech_v1.SpeechClient()
        self.config = speech_v1.RecognitionConfig(
            # Twilio Media Streams sends 8kHz mono G.711 mu-law, not LINEAR16.
            encoding=speech_v1.RecognitionConfig.AudioEncoding.MULAW,
            sample_rate_hertz=8000,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
    ) -> AsyncIterator[str]:
        """
        Collect the buffered audio and transcribe it in one batch call.

        The media handler already accumulates a chunk of caller audio and
        hands it over as a single blob, so a batch ``recognize`` is both
        simpler and far more reliable than juggling the sync streaming
        gRPC client from inside asyncio. The blocking call is pushed to a
        worker thread so the event loop keeps servicing the WebSocket.

        Args:
            audio_chunks: async iterator of raw audio bytes (mu-law, 8kHz)

        Yields:
            Final transcript text (one item, if anything was recognised)
        """

        buffer = bytearray()
        async for chunk in audio_chunks:
            if chunk:
                buffer.extend(chunk)

        if not buffer:
            return

        audio = self._speech_v1.RecognitionAudio(content=bytes(buffer))

        def _recognize():
            return self.client.recognize(config=self.config, audio=audio)

        try:
            response = await asyncio.to_thread(_recognize)
        except Exception as e:
            logger.error(f"Google STT recognize failed: {e}")
            raise

        for result in response.results:
            if not result.alternatives:
                continue
            transcript = result.alternatives[0].transcript
            if transcript and transcript.strip():
                yield transcript


def _wrap_mulaw_as_wav(mulaw_bytes: bytes, sample_rate: int = 8000) -> bytes:
    """Wrap raw 8-bit mu-law samples (what Twilio Media Streams sends) in a
    minimal WAV container (WAVE_FORMAT_MULAW = 0x0007) so STT APIs that
    expect a real audio file can decode it."""
    channels = 1
    bits_per_sample = 8
    block_align = channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align

    fmt_chunk = struct.pack(
        "<HHIIHHH",
        7,  # wFormatTag: WAVE_FORMAT_MULAW
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        0,  # cbSize (no extra format bytes)
    )

    data_size = len(mulaw_bytes)
    riff_size = 4 + (8 + len(fmt_chunk)) + (8 + data_size)

    return (
        b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
        + b"fmt " + struct.pack("<I", len(fmt_chunk)) + fmt_chunk
        + b"data" + struct.pack("<I", data_size) + mulaw_bytes
    )


# ============================================================
# SPEECH-TO-TEXT (ElevenLabs)
# ============================================================

class ElevenLabsSTT:
    """
    Speech-to-Text using ElevenLabs (Scribe). Used in place of GoogleSTT
    since it reuses the already-verified ELEVENLABS_API_KEY instead of a
    Google Cloud project that may not have the Speech-to-Text API enabled.

    ElevenLabs' endpoint is a batch upload (not gRPC streaming), so this
    mirrors GoogleSTT's behavior: buffer the caller's audio for one turn,
    then transcribe it in a single call.
    """

    API_BASE = "https://api.elevenlabs.io/v1"

    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not set")
        self.model_id = os.getenv("ELEVENLABS_STT_MODEL_ID", "scribe_v1")

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
    ) -> AsyncIterator[str]:
        buffer = bytearray()
        async for chunk in audio_chunks:
            if chunk:
                buffer.extend(chunk)

        if not buffer:
            return

        wav_bytes = _wrap_mulaw_as_wav(bytes(buffer))

        url = f"{self.API_BASE}/speech-to-text"
        headers = {"xi-api-key": self.api_key}
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        # tag_audio_events=false: without this, Scribe labels non-speech
        # audio (line noise, silence, clicks) with tags like "[pause]" or
        # "[phone ringing]" - which we'd otherwise treat as real speech and
        # feed straight into the agent.
        data = {"model_id": self.model_id, "tag_audio_events": "false"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, files=files, data=data)

                if resp.status_code != 200:
                    body = resp.text
                    logger.error(f"ElevenLabs STT HTTP {resp.status_code}: {body[:800]}")
                    raise RuntimeError(f"ElevenLabs STT {resp.status_code}: {body[:800]}")

                result = resp.json()

        except Exception as e:
            logger.error(f"ElevenLabs STT error: {e}")
            raise

        text = (result.get("text") or "").strip()
        # Defensive: strip any bracketed audio-event tags Scribe still
        # emits (e.g. "[click]", "[background noise]") even with
        # tag_audio_events off, so they never reach the agent as if they
        # were something the caller said.
        cleaned = re.sub(r"\[[^\]]*\]", "", text).strip()
        if cleaned:
            yield cleaned


# ============================================================
# TEXT-TO-SPEECH (ElevenLabs)
# ============================================================

class ElevenLabsTTS:
    """
    Text-to-Speech using ElevenLabs API.
    Streams audio bytes for low-latency response.
    """

    # ElevenLabs HTTP streaming endpoint. Used directly instead of the SDK
    # because the SDK's client surface changes between releases (the
    # deployed build exposes `text_to_speech` as a realtime client with no
    # `convert_as_stream`), whereas the REST contract is stable.
    API_BASE = "https://api.elevenlabs.io/v1"

    def __init__(self, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        # Default voice: "Rachel" (21m00Tcm4TlvDq8ikWAM)
        # Other voices: see ElevenLabs dashboard
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not set")

        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", voice_id)
        self.model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
        self.output_format = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "ulaw_8000")
        # Lower stability = more natural variation; too low sounds erratic.
        self.voice_settings = {
            "stability": float(os.getenv("ELEVENLABS_STABILITY", "0.45")),
            "similarity_boost": float(os.getenv("ELEVENLABS_SIMILARITY", "0.75")),
            "style": float(os.getenv("ELEVENLABS_STYLE", "0")),
        }

    async def text_to_speech_stream(
        self,
        text: str,
    ) -> AsyncIterator[bytes]:
        """
        Convert text to speech and stream raw audio bytes.

        Requests ``ulaw_8000`` so the bytes can go straight into Twilio
        Media Streams (8kHz mono mu-law) after base64 encoding.

        Args:
            text: The text to synthesize

        Yields:
            Audio chunks (8kHz mono mu-law)
        """

        if not text.strip():
            return

        url = f"{self.API_BASE}/text-to-speech/{self.voice_id}/stream"
        params = {"output_format": self.output_format}
        headers = {
            "xi-api-key": self.api_key,
            "content-type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": self.voice_settings,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST", url, params=params, headers=headers, json=payload
                ) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        logger.error(
                            f"ElevenLabs TTS HTTP {resp.status_code} "
                            f"(voice={self.voice_id} model={self.model_id} "
                            f"format={self.output_format}): {body[:800]}"
                        )
                        raise RuntimeError(
                            f"ElevenLabs {resp.status_code}: {body[:800]}"
                        )

                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk

        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
            raise


# ============================================================
# GOOGLE CLOUD TTS (fallback/alternative)
# ============================================================

class GoogleTTS:
    """
    Text-to-Speech using Google Cloud (alternative to ElevenLabs).
    Lower cost but slightly longer latency than ElevenLabs.
    """

    def __init__(self):
        from google.cloud import texttospeech
        self.client = texttospeech.TextToSpeechClient()

    async def text_to_speech(
        self,
        text: str,
    ) -> bytes:
        """
        Convert text to speech (returns full audio, not streamed).
        
        Args:
            text: The text to synthesize
            
        Returns:
            Raw audio bytes (LINEAR16, 8kHz for Twilio)
        """

        from google.cloud import texttospeech

        if not text.strip():
            return b""

        input_text = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-C",  # High quality neural voice
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=8000,  # Match Twilio
            speaking_rate=1.0,
        )

        request = texttospeech.SynthesizeSpeechRequest(
            input=input_text,
            voice=voice,
            audio_config=audio_config,
        )

        response = self.client.synthesize_speech(request=request)
        return response.audio_content


# ============================================================
# FACTORY: get the right TTS
# ============================================================

def get_tts_engine(engine: str = "elevenlabs"):
    """
    Returns the configured TTS engine.
    
    Args:
        engine: "elevenlabs" or "google"
    
    Returns:
        TTS instance (ElevenLabsTTS or GoogleTTS)
    """
    if engine.lower() == "elevenlabs":
        return ElevenLabsTTS()
    elif engine.lower() == "google":
        return GoogleTTS()
    else:
        raise ValueError(f"Unknown TTS engine: {engine}")


# ============================================================
# FACTORY: get the right STT
# ============================================================

def get_stt_engine(engine: str = "elevenlabs"):
    """
    Returns the configured STT engine.

    Args:
        engine: "elevenlabs" or "google"

    Returns:
        STT instance (ElevenLabsSTT or GoogleSTT)
    """
    if engine.lower() == "elevenlabs":
        return ElevenLabsSTT()
    elif engine.lower() == "google":
        return GoogleSTT()
    else:
        raise ValueError(f"Unknown STT engine: {engine}")
