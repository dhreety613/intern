import os
import json
from typing import AsyncIterator
from google.cloud import speech_v1
from google.cloud import speech_v1p1beta1
from elevenlabs.client import ElevenLabs
import asyncio
import logging

logger = logging.getLogger(__name__)


# ============================================================
# SPEECH-TO-TEXT (Google Cloud)
# ============================================================

class GoogleSTT:
    """
    Streaming Speech-to-Text using Google Cloud.
    Handles real-time audio chunks and returns transcribed text.
    """

    def __init__(self):
        # Requires GOOGLE_APPLICATION_CREDENTIALS env var
        # pointing to your service account JSON key
        self.client = speech_v1.SpeechClient()
        self.config = speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=8000,  # Twilio sends 8kHz
            language_code="en-US",
            enable_automatic_punctuation=True,
        )

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
    ) -> AsyncIterator[str]:
        """
        Stream audio chunks, yield transcribed text as it arrives.
        
        Args:
            audio_chunks: async iterator of raw audio bytes (LINEAR16, 8kHz)
            
        Yields:
            Transcribed text (partial then final)
        """

        def requests():
            """Generator that yields RecognizeRequest objects."""
            yield speech_v1.StreamingRecognizeRequest(
                streaming_config=speech_v1.StreamingRecognitionConfig(
                    config=self.config,
                    interim_results=True,  # Get partial results
                )
            )

            # Yield audio chunks as requests
            async def audio_gen():
                async for chunk in audio_chunks:
                    yield chunk

            # This is awkward: we need sync generator but have async chunks.
            # Use a blocking queue as bridge.
            import queue
            q = queue.Queue()
            error_holder = [None]

            async def feed_queue():
                try:
                    async for chunk in audio_chunks:
                        q.put(("chunk", chunk))
                    q.put(("done", None))
                except Exception as e:
                    error_holder[0] = e
                    q.put(("error", e))

            # Start feeding task (run in background)
            task = asyncio.create_task(feed_queue())

            while True:
                item_type, item = q.get()
                if item_type == "chunk":
                    yield speech_v1.StreamingRecognizeRequest(
                        audio_content=item
                    )
                elif item_type == "done":
                    break
                elif item_type == "error":
                    raise item

        # Call the streaming recognize API
        responses = self.client.streaming_recognize(requests())

        for response in responses:
            if not response.results:
                continue

            # Get the first result (most confident)
            result = response.results[0]

            if result.alternatives:
                transcript = result.alternatives[0].transcript

                # Only yield final results to avoid noise
                if result.is_final:
                    yield transcript


# ============================================================
# TEXT-TO-SPEECH (ElevenLabs)
# ============================================================

class ElevenLabsTTS:
    """
    Text-to-Speech using ElevenLabs API.
    Streams audio bytes for low-latency response.
    """

    def __init__(self, voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        # Default voice: "Rachel" (21m00Tcm4TlvDq8ikWAM)
        # Other voices: see ElevenLabs dashboard
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not set")

        self.client = ElevenLabs(api_key=self.api_key)
        self.voice_id = voice_id

    async def text_to_speech_stream(
        self,
        text: str,
    ) -> AsyncIterator[bytes]:
        """
        Convert text to speech and stream audio bytes.
        
        Args:
            text: The text to synthesize
            
        Yields:
            Audio chunks (mp3 or pcm, depending on ElevenLabs config)
        """

        if not text.strip():
            return

        try:
            # Use ElevenLabs streaming API for low latency
            # This returns an iterator of audio chunks
            audio_stream = self.client.text_to_speech.convert_as_stream(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_monolingual_v1",
                stream=True,
            )

            # Yield each chunk as it arrives
            for chunk in audio_stream:
                if chunk:
                    yield chunk
                    # Small delay to avoid overwhelming the network
                    await asyncio.sleep(0.01)

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
