import json
import base64
import logging
from typing import Dict, Any
from fastapi import WebSocket
import asyncio
from datetime import datetime, timedelta

from call_agent.app.call import CallSession
from call_agent.app.speech_services import GoogleSTT, get_tts_engine

logger = logging.getLogger(__name__)


# ============================================================
# TWILIO MEDIA STREAMS HANDLER
# ============================================================

class TwilioMediaStream:
    """
    Handles bidirectional WebSocket connection to Twilio Media Streams.
    
    Flow:
    1. Receive audio from caller (8kHz LINEAR16)
    2. Convert to text (Google Cloud STT)
    3. Process through CallSession (your LangGraph agent)
    4. Convert response to speech (ElevenLabs TTS)
    5. Stream audio back to caller
    
    TIMEOUT HANDLING:
    - If agent returns WAIT, starts 90-second timer
    - If no customer input within 90 seconds, auto-ends call
    - Prevents indefinite holds
    """

    def __init__(
        self,
        websocket: WebSocket,
        customer_name: str = "",
        email: str = "",
        phone_number: str = "",
        calibration_dates: str = "",
    ):
        self.websocket = websocket
        self.stream_sid = None
        self.call_sid = None

        # Parse calibration_dates from comma-separated string
        dates_list = []
        if calibration_dates:
            dates_list = [d.strip() for d in calibration_dates.split(",") if d.strip()]

        # Create or reuse CallSession
        self.session = CallSession(
            customer_name=customer_name or "Customer",
            email=email or "",
            phone_number=phone_number or "",
            calibration_dates=dates_list,
        )

        # Speech services
        self.stt = GoogleSTT()
        self.tts = get_tts_engine("elevenlabs")

        # Buffers
        # Twilio streams 8kHz mono mu-law => 8000 bytes/sec. Collect roughly
        # 1.5s of caller audio before sending a batch to STT so utterances
        # aren't chopped mid-word.
        self.audio_buffer = bytearray()
        self.min_audio_buffer_size = 12000
        self.silence_duration = 0
        self.silence_threshold = 10

        # TIMEOUT TRACKING
        self.last_agent_task = None
        self.wait_started_at = None
        self.wait_timeout_seconds = 90
        self.wait_timeout_task = None

    async def handle(self):
        """
        Main WebSocket message loop.
        """
        try:
            while True:
                message = await self.websocket.receive_text()
                data = json.loads(message)

                event = data.get("event")

                if event == "connected":
                    self._handle_connected(data)

                elif event == "start":
                    self._handle_start(data)

                elif event == "media":
                    await self._handle_media(data)

                elif event == "stop":
                    self._handle_stop(data)
                    break

        except Exception as e:
            logger.error(f"Media stream error: {e}", exc_info=True)
        finally:
            self._cancel_wait_timeout()
            await self.websocket.close()

    def _handle_connected(self, data: Dict[str, Any]):
        """Handle Twilio connection event."""
        logger.info("Twilio Media Streams connected")

    def _handle_start(self, data: Dict[str, Any]):
        """Handle Twilio start event (stream parameters)."""
        self.stream_sid = data.get("streamSid")
        self.call_sid = data.get("callSid")
        logger.info(f"Stream started: {self.stream_sid}")

        # Send Anne's opening greeting
        asyncio.create_task(self._send_opening_greeting())

    def _handle_stop(self, data: Dict[str, Any]):
        """Handle Twilio stop event."""
        logger.info(f"Stream stopped: {self.stream_sid}")
        self._cancel_wait_timeout()

    async def _handle_media(self, data: Dict[str, Any]):
        """
        Handle incoming audio chunk from Twilio.
        Buffer it and process when ready.
        """
        payload = data.get("media", {})
        audio_chunk = payload.get("payload")

        if not audio_chunk:
            return

        # Customer provided input - cancel any pending wait timeout
        self._cancel_wait_timeout()

        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(audio_chunk)
            self.audio_buffer.extend(audio_bytes)
        except Exception as e:
            logger.error(f"Failed to decode audio: {e}")
            return

        # Check for silence or buffer size
        if len(self.audio_buffer) >= self.min_audio_buffer_size:
            await self._process_audio_buffer()

    async def _process_audio_buffer(self):
        """
        Process accumulated audio buffer:
        1. Transcribe with STT
        2. Run through agent
        3. Synthesize response with TTS
        4. Stream back to Twilio
        """
        if not self.audio_buffer:
            return

        # Convert buffer to audio chunk
        audio_bytes = bytes(self.audio_buffer)
        self.audio_buffer.clear()

        # Transcribe
        transcript = None
        try:
            async for text in self.stt.transcribe_stream(
                self._async_audio_generator(audio_bytes)
            ):
                transcript = text
                logger.info(f"Customer: {transcript}")

        except Exception as e:
            logger.error(f"STT error: {e}")
            return

        if not transcript or not transcript.strip():
            return

        # Process through agent
        try:
            result = self.session.process(transcript)
            response_text = result.get("response", "")
            call_ended = result.get("call_ended", False)
            decision_task = result.get("decision", {}).get("present_task", "CONTINUE")

        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            response_text = "Sorry, I encountered an error. Could you please repeat that?"
            call_ended = False
            decision_task = "CONTINUE"

        logger.info(f"Anne: {response_text}")
        logger.info(f"Decision Task: {decision_task}")

        # Store the decision task for timeout tracking
        self.last_agent_task = decision_task

        # Synthesize and stream response
        if response_text:
            await self._stream_tts_to_twilio(response_text)

        # TIMEOUT HANDLING: If WAIT, start 90-second timer
        if decision_task == "WAIT":
            self._start_wait_timeout()

        # End call if agent decides so
        if call_ended:
            await self._send_stop()

    def _start_wait_timeout(self):
        """Start a 90-second timeout for WAIT state."""
        self.wait_started_at = datetime.now()
        logger.info(f"WAIT started at {self.wait_started_at} - will timeout in {self.wait_timeout_seconds}s")

        # Cancel any existing timeout task
        self._cancel_wait_timeout()

        # Create new timeout task
        self.wait_timeout_task = asyncio.create_task(self._wait_timeout_handler())

    def _cancel_wait_timeout(self):
        """Cancel the pending wait timeout task."""
        if self.wait_timeout_task and not self.wait_timeout_task.done():
            self.wait_timeout_task.cancel()
            logger.info("Wait timeout cancelled (customer provided input)")
            self.wait_timeout_task = None
            self.wait_started_at = None

    async def _wait_timeout_handler(self):
        """
        Handle 90-second timeout during WAIT state.
        If no customer input received, auto-end call politely.
        """
        try:
            # Wait 90 seconds
            await asyncio.sleep(self.wait_timeout_seconds)

            # If we're still here, no customer input was received
            logger.warning(
                f"WAIT timeout triggered - no customer input for {self.wait_timeout_seconds}s"
            )

            # Auto-generate timeout message and process through agent
            timeout_message = "[No response received - timeout]"
            logger.info(f"Auto-processing timeout: {timeout_message}")

            try:
                result = self.session.process(timeout_message)
                response_text = result.get("response", "")
                call_ended = result.get("call_ended", False)

            except Exception as e:
                logger.error(f"Error processing timeout: {e}")
                response_text = "Thank you for your time. Goodbye."
                call_ended = True

            logger.info(f"Timeout Response: {response_text}")

            # Send final response
            if response_text:
                await self._stream_tts_to_twilio(response_text)

            # End the call
            await self._send_stop()

        except asyncio.CancelledError:
            # Timeout was cancelled (customer provided input)
            pass
        except Exception as e:
            logger.error(f"Wait timeout handler error: {e}")

    async def _send_opening_greeting(self):
        """Send Anne's opening greeting to caller."""
        greeting = self.session.start()
        logger.info(f"Anne (greeting): {greeting}")
        await self._stream_tts_to_twilio(greeting)

    async def _stream_tts_to_twilio(self, text: str):
        """
        Synthesize text to speech and stream chunks to Twilio.
        """
        try:
            # Start streaming TTS
            chunk_index = 0

            async for audio_chunk in self.tts.text_to_speech_stream(text):
                # Package as Twilio media message
                payload = base64.b64encode(audio_chunk).decode("utf-8")

                message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {
                        "payload": payload,
                    },
                }

                await self.websocket.send_text(json.dumps(message))
                chunk_index += 1

                # Small delay between chunks
                await asyncio.sleep(0.02)

            logger.info(f"Streamed {chunk_index} audio chunks to Twilio")

        except Exception as e:
            logger.error(f"TTS streaming error: {e}")

    async def _send_stop(self):
        """Tell Twilio to hang up."""
        self._cancel_wait_timeout()

        message = {
            "event": "stop",
            "streamSid": self.stream_sid,
        }
        await self.websocket.send_text(json.dumps(message))
        logger.info("Sent stop event to Twilio")

    async def _async_audio_generator(self, audio_bytes: bytes):
        """Async generator that yields audio bytes."""
        yield audio_bytes