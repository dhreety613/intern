import json
import re
import base64
import logging
from typing import Dict, Any
from urllib.parse import quote
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect
import asyncio
from datetime import datetime, timedelta

import os
import time

from call_agent.app.call import CallSession
from call_agent.app.call_result import build_call_result, deliver_call_result
from call_agent.app.dtmf import dtmf_ulaw
from call_agent.app.speech_services import get_stt_engine, get_tts_engine

logger = logging.getLogger(__name__)


def _build_mulaw_to_linear_table():
    """Standard ITU-T G.711 mu-law -> 16-bit linear PCM decode table."""
    BIAS = 0x84
    table = []
    for i in range(256):
        u = ~i & 0xFF
        sign = u & 0x80
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        sample = ((mantissa << 3) + BIAS) << exponent
        sample -= BIAS
        table.append(-sample if sign else sample)
    return table


_MULAW_TO_LINEAR = _build_mulaw_to_linear_table()


# Live streams by Twilio CallSid, so the async answering-machine callback
# (api.py /amd-callback) can reach the call it is about.
ACTIVE_STREAMS: Dict[str, "TwilioMediaStream"] = {}


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
        products: str = "",
    ):
        self.websocket = websocket
        self.stream_sid = None
        self.call_sid = None

        # Fallback customer data, parsed from the /media WebSocket URL's
        # query string. Some proxies don't reliably preserve query strings
        # on a WSS upgrade request, so this is only used if Twilio's
        # <Parameter> values (delivered in the "start" event) are missing.
        self._fallback_customer_name = customer_name
        self._fallback_email = email
        self._fallback_phone_number = phone_number
        self._fallback_calibration_dates = calibration_dates
        self._fallback_products = products

        # Created once the "start" event arrives and real customer data
        # (from Twilio's customParameters, or the fallback above) is known.
        self.session = None
        self._answered_by = ""
        self._person_id = ""
        self._deal_id = ""
        self._result_sent = False

        # Per-call tracking (see _track). Every "TRACK |" line in the Render
        # logs carries call/deal/person so one call can be grepped end to end.
        self._transcript = []
        self._started_at = None

        # Speech services
        self.stt = get_stt_engine(os.getenv("STT_ENGINE", "elevenlabs"))
        self.tts = get_tts_engine("elevenlabs")

        # Buffers. Twilio streams 8kHz mono mu-law => 8 bytes/ms. Rather than
        # flushing to STT on a fixed byte count (which either grabs leading
        # silence before the caller starts talking, or chops a longer
        # answer mid-word), buffer continuously and use energy-based voice
        # activity detection to flush once the caller actually pauses.
        self.audio_buffer = bytearray()
        self._voiced_ms = 0.0
        self._silence_run_ms = 0.0

        # TIMEOUT TRACKING
        self.last_agent_task = None
        self.wait_started_at = None
        self.wait_timeout_seconds = 90
        self.wait_timeout_task = None

        # CALL WATCHDOG (added 2026-10-16): no call may stay open forever.
        #  - silence: nobody speaks for SILENCE_TIMEOUT_SECONDS after Anne
        #    last finished talking -> short goodbye + hang up.
        #  - max length: MAX_CALL_SECONDS total -> hard hang up (also via
        #    Twilio REST in case closing the socket does not end the call).
        # Both are env-tunable on Render. Set a value to 0 to disable it.
        self._last_activity = time.monotonic()
        self._last_voiced_mono = 0.0   # last time any voiced frame was heard (voicemail waiting)
        self._vm_pending = False       # voicemail spotted from the transcript, waiting for the beep
        self._beep_event = asyncio.Event()
        self._watchdog_task = None
        self.silence_timeout_seconds = int(os.getenv("ANNE_SILENCE_TIMEOUT_SECONDS", "25"))
        self.max_call_seconds = int(os.getenv("ANNE_MAX_CALL_SECONDS", "360"))

        # Serialize audio writes so the greeting and replies don't interleave
        # frames on the socket.
        self._tts_lock = asyncio.Lock()
        self._greeting_task = None
        self._processing = False

        # Barge-in: set once the caller has spoken over Anne for long enough
        # to count as an interruption, checked inside _stream_tts_to_twilio's
        # send loop so it can stop mid-utterance instead of forcing the
        # caller to wait out however long her whole reply takes to play.
        self._interrupt_requested = False

        # session.process() runs off-thread and can't actually be cancelled
        # once started. If a turn times out (see AGENT_TIMEOUT_SECONDS), the
        # thread keeps running in the background and will still mutate
        # self.session.state when it eventually finishes. This lock makes
        # sure the NEXT turn's session.process() call always waits for that
        # orphaned call to truly finish first, so two turns never mutate
        # session state concurrently or land out of order.
        self._agent_lock = asyncio.Lock()

        # Set once Twilio's "stop" event arrives or we close the socket
        # ourselves, so any in-flight background turn (e.g. a filler or a
        # late agent response) knows not to try writing to a closed socket.
        self._call_ended = False

        self._voicemail_key_task = None

        # Set when Twilio's async AMD says a machine picked up; from then on
        # the normal conversation is dropped in favour of the fixed message.
        self._voicemail = False

    # Hard caps so a slow/hung STT or LLM call can't permanently lock the
    # half-duplex gate (self._processing) and go silent for the rest of the
    # call. AGENT_TIMEOUT_SECONDS is generous because real calls have shown
    # ~25s turns that still succeed (chained memory/brain/decision Gemini
    # calls) - a tighter cap just interrupts otherwise-working turns.
    STT_TIMEOUT_SECONDS = 20.0
    AGENT_TIMEOUT_SECONDS = 40.0
    FILLER_DELAY_SECONDS = 7.0
    FILLER_REPEAT_SECONDS = 15.0

    # After pressing a voicemail option there may be only a beep, no spoken
    # greeting. If no menu/greeting takes over within this long (and the far
    # end is quiet), leave the fixed message anyway.
    VOICEMAIL_KEY_WAIT_SECONDS = 5
    VOICEMAIL_KEY_MAX_SECONDS = 15

    # Voice activity detection (energy-based) for deciding when the caller
    # has finished a turn, instead of flushing on a fixed audio duration.
    BYTES_PER_MS = 8  # 8kHz mono mu-law = 1 byte/sample
    VAD_AMPLITUDE_THRESHOLD = 350  # decoded-linear amplitude above this counts as voice
    MIN_SPEECH_MS = 300  # must hear at least this much voice before a turn can end
    SILENCE_HANGOVER_MS = 700  # pause length that marks end-of-turn
    MAX_BUFFER_MS = 8000  # hard cap so one long turn can't buffer forever
    MAX_SILENCE_ONLY_MS = 4000  # drop the buffer if nothing but noise for this long
    BARGE_IN_VOICED_MS = 200  # sustained voice while Anne is speaking = interruption

    def _track(self, event: str, **fields):
        """Greppable per-call log line: `TRACK | EVENT | call=.. deal=.. person=.. | k=v`."""
        extra = " ".join(f"{k}={v!r}" for k, v in fields.items())
        logger.info(
            f"TRACK | {event} | call={self.call_sid} deal={self._deal_id or '-'} "
            f"person={self._person_id or '-'} | {extra}"
        )

    def _say(self, role: str, text: str):
        """Log one conversation turn and keep it for the end-of-call summary."""
        self._transcript.append(f"{role}: {text}")
        self._track("TURN", speaker=role, text=text)

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
                    self._start_watchdog()

                elif event == "media":
                    await self._handle_media(data)

                elif event == "stop":
                    self._handle_stop(data)
                    break

        except WebSocketDisconnect:
            logger.info("Twilio closed the media stream")
        except Exception as e:
            logger.error(f"Media stream error: {e}", exc_info=True)
        finally:
            self._call_ended = True
            self._cancel_wait_timeout()
            if self._watchdog_task:
                self._watchdog_task.cancel()
            # 2026-10-29: a carrier recording ("mailbox is full...") hangs up right
            # after speaking, so its transcription can still be in flight when the
            # stream closes (Jim May, 2026-10-28). Give it up to 3 s so is_voicemail
            # is set before CALL_END / the result are built.
            for _ in range(30):
                if not self._processing:
                    break
                await asyncio.sleep(0.1)
            if any("mailbox" in t.lower() and "full" in t.lower() for t in self._transcript):
                self._track("MAILBOX_FULL", note="voicemail box full - message could not be left")
            self._track_end()
            if self._voicemail_key_task:
                self._voicemail_key_task.cancel()
            await self._send_call_result()
            if self.call_sid and ACTIVE_STREAMS.get(self.call_sid) is self:
                del ACTIVE_STREAMS[self.call_sid]
            try:
                await self.websocket.close()
            except Exception:
                pass

    def _handle_connected(self, data: Dict[str, Any]):
        """Handle Twilio connection event."""
        logger.info("Twilio Media Streams connected")

    def _handle_start(self, data: Dict[str, Any]):
        """Handle Twilio start event (stream parameters)."""
        start = data.get("start", {}) or {}
        self.stream_sid = data.get("streamSid") or start.get("streamSid")
        self.call_sid = data.get("callSid") or start.get("callSid")
        if self.call_sid:
            ACTIVE_STREAMS[self.call_sid] = self

        # Prefer Twilio's <Parameter> values (delivered here, in the "start"
        # event) over the URL query string fallback, since some proxies
        # don't preserve query strings on the WSS upgrade request.
        custom_params = start.get("customParameters", {}) or {}

        customer_name = custom_params.get("customer_name") or self._fallback_customer_name
        email = custom_params.get("email") or self._fallback_email
        phone_number = custom_params.get("phone_number") or self._fallback_phone_number
        calibration_dates = custom_params.get("calibration_dates") or self._fallback_calibration_dates
        products = custom_params.get("products") or self._fallback_products
        self._answered_by = custom_params.get("answered_by", "")
        self._person_id = custom_params.get("person_id", "")
        self._deal_id = custom_params.get("deal_id", "")

        dates_list = []
        if calibration_dates:
            # Don't split "September 15, 2026" at its own comma: only split on
            # commas that aren't followed by a 4-digit year.
            dates_list = [
                d.strip()
                for d in re.split(r",(?!\s*\d{4})", calibration_dates)
                if d.strip()
            ]

        products_list = []
        if products:
            products_list = [p.strip() for p in products.split(",") if p.strip()]

        self.session = CallSession(
            customer_name=customer_name or "Customer",
            email=email or "",
            phone_number=phone_number or "",
            calibration_dates=dates_list,
            products=products_list,
        )

        logger.info(
            f"Stream started: sid={self.stream_sid} call={self.call_sid} "
            f"customer={customer_name!r} (source={'customParameters' if custom_params.get('customer_name') else 'query-string-fallback'})"
        )

        self._started_at = datetime.now()
        self._track(
            "CALL_START",
            customer=customer_name,
            phone=phone_number,
            products=products_list,
            dates=dates_list,
            answered_by=self._answered_by,
        )

        # Send Anne's opening greeting. Keep a reference so the task isn't
        # garbage-collected and its exceptions surface.
        self._greeting_task = asyncio.create_task(self._send_opening_greeting())

    def _handle_stop(self, data: Dict[str, Any]):
        """Handle Twilio stop event."""
        logger.info(f"Stream stopped: {self.stream_sid}")
        self._call_ended = True
        self._cancel_wait_timeout()

    def _reset_vad_state(self):
        self._voiced_ms = 0.0
        self._silence_run_ms = 0.0

    def _frame_is_voiced(self, frame: bytes) -> bool:
        if not frame:
            return False
        total = 0
        for b in frame:
            total += abs(_MULAW_TO_LINEAR[b])
        return (total / len(frame)) >= self.VAD_AMPLITUDE_THRESHOLD

    async def _handle_media(self, data: Dict[str, Any]):
        """
        Handle incoming audio chunk from Twilio.

        Buffers continuously and uses simple energy-based voice activity
        detection to decide when the caller has actually finished a turn
        (real speech followed by a pause), rather than flushing after a
        fixed duration regardless of what's in it.
        """
        payload = data.get("media", {})
        audio_chunk = payload.get("payload")

        if not audio_chunk:
            return

        if self._voicemail:
            if self._vm_pending:
                # Keep listening for the end of the greeting / the beep.
                try:
                    if self._frame_is_voiced(base64.b64decode(audio_chunk)):
                        self._last_voiced_mono = time.monotonic()
                except Exception:
                    pass
            return

        anne_speaking = self._tts_lock.locked()

        # While a turn is still being processed (STT/agent), or a previous
        # (possibly slow/timed-out-from-the-caller's-view) agent call is
        # still actually running, there's nothing playing to barge into and
        # starting a second competing turn is what caused every later turn
        # to also "time out" while queued behind it - drop audio as before.
        if not anne_speaking and (self._processing or self._agent_lock.locked()):
            self.audio_buffer.clear()
            self._reset_vad_state()
            return

        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(audio_chunk)
        except Exception as e:
            logger.error(f"Failed to decode audio: {e}")
            return

        frame_ms = len(audio_bytes) / self.BYTES_PER_MS
        voiced = self._frame_is_voiced(audio_bytes)

        if anne_speaking and self._voiced_ms == 0.0 and not voiced:
            # Not an interruption in progress - just line noise/echo while
            # she's talking normally. Don't buffer it.
            return

        self.audio_buffer.extend(audio_bytes)

        if voiced:
            self._last_activity = time.monotonic()
            # Customer is actively speaking - cancel any pending wait timeout.
            self._cancel_wait_timeout()
            self._voiced_ms += frame_ms
            self._silence_run_ms = 0.0
        else:
            self._silence_run_ms += frame_ms

        if anne_speaking:
            # Barge-in: sustained voice while she's still talking means the
            # caller is interrupting, not just a stray noise. Once flagged,
            # _stream_tts_to_twilio's send loop stops and clears Twilio's
            # buffer. Everything buffered here carries over as the start of
            # the caller's next turn once her stream actually stops.
            if self._voiced_ms >= self.BARGE_IN_VOICED_MS:
                self._interrupt_requested = True
            return

        buffer_ms = len(self.audio_buffer) / self.BYTES_PER_MS

        has_enough_speech = self._voiced_ms >= self.MIN_SPEECH_MS
        end_of_turn = has_enough_speech and self._silence_run_ms >= self.SILENCE_HANGOVER_MS
        hit_max_duration = buffer_ms >= self.MAX_BUFFER_MS

        if not has_enough_speech and buffer_ms >= self.MAX_SILENCE_ONLY_MS:
            # Nothing but noise/silence for a while - drop it rather than
            # growing the buffer or wasting an STT call on it.
            self.audio_buffer.clear()
            self._reset_vad_state()
            return

        # Flush and process the turn in the background so the receive loop
        # keeps draining the socket.
        if end_of_turn or hit_max_duration:
            self._processing = True
            self._reset_vad_state()
            asyncio.create_task(self._process_turn())

    async def _process_turn(self):
        """Run one STT -> agent -> TTS turn, then re-open the mic."""
        try:
            await self._process_audio_buffer()
        finally:
            self._processing = False

    async def _process_audio_buffer(self):
        """
        Process accumulated audio buffer:
        1. Transcribe with STT
        2. Run through agent
        3. Synthesize response with TTS
        4. Stream back to Twilio
        """
        if not self.audio_buffer or self.session is None or self._voicemail:
            return

        # Convert buffer to audio chunk
        audio_bytes = bytes(self.audio_buffer)
        self.audio_buffer.clear()

        # Transcribe. A hard timeout here matters more than usual: if this
        # hangs instead of raising, `self._processing` never resets and the
        # half-duplex gate in _handle_media silently drops every future
        # audio chunk for the rest of the call - total, permanent silence.
        async def _run_stt():
            last = None
            async for text in self.stt.transcribe_stream(
                self._async_audio_generator(audio_bytes)
            ):
                last = text
                logger.info(f"Customer: {last}")
                self._say("Customer", last)
            return last

        transcript = None
        _t = time.monotonic()
        try:
            transcript = await asyncio.wait_for(_run_stt(), timeout=self.STT_TIMEOUT_SECONDS)
            self._track("LATENCY", stage="stt", ms=int((time.monotonic() - _t) * 1000))

        except asyncio.TimeoutError:
            logger.error(f"STT timed out after {self.STT_TIMEOUT_SECONDS}s")
            await self._stream_tts_to_twilio(
                "Sorry, I couldn't hear that clearly. Could you say that again?"
            )
            return

        except Exception as e:
            logger.error(f"STT error: {e}")
            # Don't leave the caller in dead air - let them know and let
            # them try again rather than silently dropping the turn.
            await self._stream_tts_to_twilio(
                "Sorry, I couldn't hear that clearly. Could you say that again?"
            )
            return

        if not transcript or not transcript.strip():
            logger.info("STT returned no transcript for this turn")
            if self.session.state.get("ivr", {}).get("active"):
                return
            await self._stream_tts_to_twilio(
                "Sorry, I didn't catch that. Could you repeat that, please?"
            )
            return

        # Process through agent. call_graph.invoke() is synchronous, can
        # block for seconds on LLM calls (real calls have shown 25-70s for
        # some turns - the chained memory/brain/decision Gemini calls add
        # up), and has no bounded timeout of its own, so run it off the
        # event loop (to keep reading/writing the Twilio socket) and cap
        # how long we'll wait for the same reason as the STT call above.
        # A filler plays periodically while it's still going so the caller
        # isn't met with dead air while it thinks.
        asyncio.create_task(self._speak_filler_while_busy())
        try:
            _t = time.monotonic()
            result = await self._call_agent(transcript)
            self._track("LATENCY", stage="agent", ms=int((time.monotonic() - _t) * 1000))
            response_text = result.get("response", "")
            call_ended = result.get("call_ended", False)
            decision_task = result.get("decision", {}).get("present_task", "CONTINUE")

        except asyncio.TimeoutError:
            # Don't tell the caller to repeat themselves - their answer is
            # still being processed (see _call_agent/_deliver_late_response,
            # which will speak the real response once it's actually ready).
            logger.error(
                f"Agent timed out after {self.AGENT_TIMEOUT_SECONDS}s "
                "- will deliver the real answer once it's ready"
            )
            return

        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            response_text = "Sorry, I encountered an error. Could you please repeat that?"
            call_ended = False
            decision_task = "CONTINUE"
            result = {}

        if self._voicemail:
            return

        logger.info(f"Anne: {response_text}")
        logger.info(f"Decision Task: {decision_task}")
        self._say("Anne", response_text)
        self._track("DECISION", task=decision_task, call_ended=call_ended)

        # Store the decision task for timeout tracking
        self.last_agent_task = decision_task

        decision_data = result.get("decision", {})
        redirect_to_rep = decision_data.get("redirect_to_rep", False)
        redirect_reason = decision_data.get("redirect_reason", "")

        # Voicemail recognised from the transcript: do NOT talk over the
        # recorded greeting (the message was cut off after 0 frames on the
        # Neal Riebel call, 2026-10-15). Wait for the beep, then speak.
        if (
            response_text
            and self.session.state.get("is_voicemail")
            and not self._voicemail
        ):
            await self._leave_voicemail_after_beep(response_text)
            return

        # Synthesize and stream response
        if response_text:
            await self._stream_tts_to_twilio(response_text)

        if decision_data.get("ivr_log"):
            self._track("IVR_DECISION", **decision_data["ivr_log"])

        # IVR navigation: press the chosen key (Anne stays silent).
        if decision_data.get("dtmf"):
            await self._send_dtmf(decision_data["dtmf"])
            if decision_data.get("voicemail_key"):
                self._arm_voicemail_key_timer()

        # The customer confirmed a handoff to a rep (could follow a
        # guardrail miss, or just the customer asking for a person,
        # about any topic) - hand the live call off to Twilio and stop
        # driving it here.
        if redirect_to_rep:
            await self._redirect_call_to_rep(redirect_reason)
            return

        # TIMEOUT HANDLING: If WAIT, start 90-second timer
        if decision_task == "WAIT":
            self._start_wait_timeout()

        # End call if agent decides so
        if call_ended:
            await self._send_stop()

    async def _call_agent(self, text: str) -> Dict[str, Any]:
        """Run session.process() off-thread, bounded by AGENT_TIMEOUT_SECONDS.

        asyncio.to_thread() can't actually be cancelled once the thread
        starts, so a timed-out call keeps running in the background. Using
        asyncio.shield() here (instead of awaiting the thread directly)
        lets `asyncio.wait_for` give up waiting and raise TimeoutError to
        the caller without killing the underlying task - it keeps running,
        holding self._agent_lock until it truly finishes. That guarantees
        the *next* call to this method blocks until the orphaned one is
        done, so session.state is never mutated by two overlapping turns.

        On timeout, the caller's real answer isn't thrown away: a
        background task keeps waiting for it and speaks the actual response
        once it's ready (see _deliver_late_response), instead of just
        asking the caller to repeat something we're still processing.
        """
        async def _run():
            async with self._agent_lock:
                return await asyncio.to_thread(self.session.process, text)

        task = asyncio.create_task(_run())
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=self.AGENT_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            asyncio.create_task(self._deliver_late_response(task))
            raise

    async def _deliver_late_response(self, task: "asyncio.Task[Dict[str, Any]]"):
        """Speak the real response from a turn that outlasted AGENT_TIMEOUT_SECONDS.

        _handle_media gates new turns on self._agent_lock, so no newer turn
        can start (and no fresher response can exist) while this is still
        pending - it's always safe to deliver once it arrives.
        """
        try:
            result = await task
        except Exception as e:
            logger.error(f"Late agent response failed: {e}", exc_info=True)
            return

        if self._call_ended or self._voicemail:
            return

        response_text = result.get("response", "")
        call_ended = result.get("call_ended", False)
        decision_data = result.get("decision", {})
        decision_task = decision_data.get("present_task", "CONTINUE")
        redirect_to_rep = decision_data.get("redirect_to_rep", False)
        redirect_reason = decision_data.get("redirect_reason", "")

        logger.info(f"Anne (late): {response_text}")
        logger.info(f"Decision Task (late): {decision_task}")
        self._say("Anne", response_text)
        self._track("DECISION", task=decision_task, call_ended=call_ended, late=True)
        self.last_agent_task = decision_task

        if response_text:
            await self._stream_tts_to_twilio(response_text)

        if decision_data.get("dtmf"):
            await self._send_dtmf(decision_data["dtmf"])
            if decision_data.get("voicemail_key"):
                self._arm_voicemail_key_timer()

        if redirect_to_rep:
            await self._redirect_call_to_rep(redirect_reason)
            return

        if decision_task == "WAIT":
            self._start_wait_timeout()

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
                result = await self._call_agent(timeout_message)
                response_text = result.get("response", "")
                call_ended = result.get("call_ended", False)

            except Exception as e:
                logger.error(f"Error processing timeout: {e}")
                response_text = "Thank you for your time. Goodbye."
                call_ended = True

            logger.info(f"Timeout Response: {response_text}")
            self._track("WAIT_TIMEOUT", seconds=self.wait_timeout_seconds)
            self._say("Anne", response_text)

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

    def _start_watchdog(self):
        if self._watchdog_task is None:
            self._started_mono = time.monotonic()
            self._watchdog_task = asyncio.create_task(self._watchdog())

    async def _watchdog(self):
        """Ends the call on prolonged silence or when it runs too long."""
        try:
            while not self._call_ended:
                await asyncio.sleep(1)
                now = time.monotonic()

                if self.max_call_seconds and now - self._started_mono >= self.max_call_seconds:
                    self._track("MAX_DURATION", seconds=self.max_call_seconds)
                    await self._force_hangup("max_duration")
                    return

                # Anne talking / a turn being processed counts as activity.
                if self._tts_lock.locked() or self._processing or self._agent_lock.locked():
                    self._last_activity = now
                    continue
                # The 90 s WAIT timer (customer asked us to hold) has its own clock.
                if self.wait_timeout_task is not None or self._voicemail:
                    self._last_activity = now
                    continue

                if self.silence_timeout_seconds and now - self._last_activity >= self.silence_timeout_seconds:
                    self._track("SILENCE_TIMEOUT", seconds=self.silence_timeout_seconds)
                    line = "Hello? I'm having trouble hearing you. I'll try again another time. Goodbye."
                    try:
                        self._say("Anne", line)
                        await asyncio.wait_for(self._stream_tts_to_twilio(line), timeout=15)
                    except Exception as e:
                        logger.error(f"Silence goodbye failed: {e}")
                    await self._force_hangup("silence")
                    return
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Watchdog error: {e}", exc_info=True)

    async def _force_hangup(self, reason: str):
        """Close the stream AND ask Twilio to complete the call."""
        self._track("FORCE_HANGUP", reason=reason)
        await self._send_stop()
        sid = self.call_sid
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        if not (sid and account_sid and auth_token):
            return
        try:
            from twilio.rest import Client
            await asyncio.to_thread(
                lambda: Client(account_sid, auth_token).calls(sid).update(status="completed")
            )
        except Exception as e:
            logger.warning(f"Twilio hangup after {reason} failed (call may already be over): {e}")

    async def _send_opening_greeting(self):
        """Send Anne's opening greeting to caller."""
        greeting = self.session.start(answered_by=self._answered_by)
        logger.info(
            f"Anne (greeting): {greeting} (answered_by={self._answered_by!r})"
        )
        self._say("Anne", greeting)
        await self._stream_tts_to_twilio(greeting)

        # Voicemail: the fixed message is the whole call - hang up.
        if self.session.state.get("is_voicemail"):
            await self._send_stop()

    def _arm_voicemail_key_timer(self):
        if self._voicemail_key_task and not self._voicemail_key_task.done():
            self._voicemail_key_task.cancel()
        self._voicemail_key_task = asyncio.create_task(
            self._voicemail_key_timer()
        )

    async def _voicemail_key_timer(self):
        """After a voicemail key press: if nothing else takes over (no menu,
        no spoken greeting consumed the press), say the fixed line."""
        try:
            waited = 0
            while waited < self.VOICEMAIL_KEY_MAX_SECONDS and not self._call_ended:
                await asyncio.sleep(1)
                waited += 1
                quiet = (
                    not self._processing
                    and self._voiced_ms == 0.0
                    and not self._tts_lock.locked()
                )
                if waited >= self.VOICEMAIL_KEY_WAIT_SECONDS and quiet:
                    break

            if self._call_ended or self.session is None:
                return
            # A spoken greeting/menu turn already handled it (flag reset).
            if not self.session.state.get("ivr", {}).get("pressed_voicemail"):
                return

            logger.info("Voicemail key pressed, no greeting heard - leaving message")
            self._track("VOICEMAIL_KEY_TIMEOUT", note="key pressed, no greeting heard, leaving message")
            await self.handle_voicemail(force=True)
        except asyncio.CancelledError:
            pass

    def _track_end(self):
        """One-line end-of-call summary with the full transcript."""
        if self._started_at is None:
            return
        state = self.session.state if self.session else {}
        memory = state.get("memory") or {}
        self._track(
            "CALL_END",
            duration_s=int((datetime.now() - self._started_at).total_seconds()),
            turns=len(self._transcript),
            human=bool(state.get("human_confirmed")),
            voicemail=bool(state.get("is_voicemail")) or self._voicemail,
            stage=state.get("current_stage"),
            guidance=state.get("guidance_phase"),
            wrong_number=bool(memory.get("wrong_number_so_avoid")),
            do_not_call=bool(memory.get("do_not_call")),
            transcript=" || ".join(self._transcript),
        )

    async def _send_call_result(self):
        """Once per call, when the stream ends: report the outcome."""
        if self._result_sent or self.session is None:
            return
        self._result_sent = True
        try:
            result = build_call_result(
                self.session.state,
                self.call_sid or "",
                self._person_id,
                self._deal_id,
            )
            await deliver_call_result(result)
        except Exception as e:
            logger.error(f"Failed to send call result: {e}", exc_info=True)

    async def _leave_voicemail_after_beep(self, message: str):
        """Voicemail spotted from the transcript: wait until the recorded
        greeting is over (Twilio says the beep ended, or the line has been
        quiet for VM_QUIET_SECONDS; hard cap VM_MAX_WAIT_SECONDS), then say
        the message with barge-in off and hang up."""
        VM_QUIET_SECONDS = 1.5
        VM_MAX_WAIT_SECONDS = 20
        self._track("VOICEMAIL_DETECTED", via="transcript", note="waiting for beep")
        self._voicemail = True       # stops normal turns + barge-in
        self._vm_pending = True
        self._cancel_wait_timeout()
        self.audio_buffer.clear()
        self._last_voiced_mono = time.monotonic()
        started = time.monotonic()
        reason = "max_wait"
        while time.monotonic() - started < VM_MAX_WAIT_SECONDS and not self._call_ended:
            if self._beep_event.is_set():
                reason = "beep"
                await asyncio.sleep(0.3)   # let the beep tone finish
                break
            if (
                time.monotonic() - started >= 2.0
                and time.monotonic() - self._last_voiced_mono >= VM_QUIET_SECONDS
            ):
                reason = "quiet"
                break
            await asyncio.sleep(0.1)
        self._vm_pending = False
        if self._call_ended:
            return
        self._track("VOICEMAIL_BEEP_WAIT", reason=reason, waited_s=round(time.monotonic() - started, 1))
        self._interrupt_requested = False
        _t = time.monotonic()
        await self._stream_tts_to_twilio(message)
        self._track("VOICEMAIL_LEFT", spoken_s=round(time.monotonic() - _t, 1), text=message)
        await self._send_stop()

    async def handle_voicemail(self, force: bool = False, greeting_over: bool = True):
        """Twilio's async AMD says a machine answered: cut off whatever
        Anne is saying, leave the fixed message and hang up."""
        from call_agent.app.agents.voicemail_agent import VOICEMAIL_MESSAGE

        if self._vm_pending:
            # Already waiting for the beep (transcript detection): a later
            # AMD "...end_*" verdict means the greeting/beep is over.
            if greeting_over:
                self._beep_event.set()
            return
        if self._voicemail or self._call_ended:
            return

        # An IVR menu is also "a machine" to Twilio's AMD - if we've already
        # heard one on this call, keep navigating it instead of leaving the
        # voicemail line and hanging up.
        if (
            not force
            and self.session is not None
            and self.session.state.get("ivr", {}).get("seen")
            # ORIGINAL guard was just the line above. Now only keep
            # navigating if we actually pressed keys; a "menu" we never
            # pressed anything on is a voicemail greeting ("press pound...").
            and self.session.state.get("ivr", {}).get("presses")
        ):
            logger.info("AMD says machine but an IVR was detected - ignoring")
            self._track("VOICEMAIL_IGNORED", reason="IVR already detected, keep navigating menu")
            return

        logger.info(f"Voicemail detected for call {self.call_sid}")
        self._track("VOICEMAIL_DETECTED", forced=force)
        self._voicemail = True
        self._cancel_wait_timeout()
        self.audio_buffer.clear()

        # Stop any greeting/reply in progress, then wait for the TTS lock.
        self._interrupt_requested = True
        if self.session is not None:
            self.session.state["is_voicemail"] = True
            self.session.state["call_ended"] = True

        self._say("Anne (voicemail message)", VOICEMAIL_MESSAGE)
        _t = time.monotonic()
        await self._stream_tts_to_twilio(VOICEMAIL_MESSAGE)
        self._track("VOICEMAIL_LEFT", spoken_s=round(time.monotonic() - _t, 1), text=VOICEMAIL_MESSAGE)
        await self._send_stop()

    async def _speak_filler_while_busy(self):
        """Periodically reassure the caller while self._agent_lock is held.

        Keyed off the lock itself (not a fixed one-shot delay) so it keeps
        going for as long as the underlying agent call actually takes -
        including well past AGENT_TIMEOUT_SECONDS, where the caller would
        otherwise hear nothing until the real answer (see
        _deliver_late_response) finally arrives.
        """
        try:
            await asyncio.sleep(self.FILLER_DELAY_SECONDS)
            while (
                self._agent_lock.locked()
                and not self._call_ended
                and not self._voicemail
            ):
                await self._stream_tts_to_twilio("Just a moment, I'm still working on that.")
                await asyncio.sleep(self.FILLER_REPEAT_SECONDS)
        except asyncio.CancelledError:
            pass

    # 8kHz mu-law => 160 bytes == 20ms. Twilio plays inbound media as
    # 20ms frames, so slice the TTS stream to that size and pace delivery.
    TWILIO_FRAME_BYTES = 160
    TWILIO_FRAME_SECONDS = 0.02

    async def _send_media_frame(self, frame: bytes):
        await self.websocket.send_text(json.dumps({
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": base64.b64encode(frame).decode("utf-8")},
        }))

    async def _send_dtmf(self, digits: str):
        """Press keys on the far end by sending DTMF tones as call audio."""
        if not self.stream_sid or self._call_ended:
            return

        async with self._tts_lock:
            for digit in digits:
                logger.info(f"IVR: pressing {digit!r}")
                self._track("IVR_PRESS", key=digit)
                audio = dtmf_ulaw(digit)
                for i in range(0, len(audio), self.TWILIO_FRAME_BYTES):
                    frame = audio[i:i + self.TWILIO_FRAME_BYTES]
                    if len(frame) < self.TWILIO_FRAME_BYTES:
                        frame += b"\xff" * (self.TWILIO_FRAME_BYTES - len(frame))
                    await self._send_media_frame(frame)
                    await asyncio.sleep(self.TWILIO_FRAME_SECONDS)

    async def _stream_tts_to_twilio(self, text: str):
        """
        Synthesize text to speech and stream 20ms mu-law frames to Twilio.
        """
        if not self.stream_sid:
            logger.warning("No stream_sid yet - dropping TTS output")
            return

        if self._call_ended:
            logger.info("Call already ended - dropping TTS output")
            return

        async with self._tts_lock:
            self._interrupt_requested = False
            try:
                leftover = b""
                frames_sent = 0
                interrupted = False

                _t = time.monotonic()
                async for audio_chunk in self.tts.text_to_speech_stream(text):
                    if frames_sent == 0 and _t:
                        self._track("LATENCY", stage="tts_first_byte", ms=int((time.monotonic() - _t) * 1000))
                        _t = None
                    if self._call_ended:
                        logger.info("Call ended mid-stream - stopping TTS output")
                        return

                    if self._interrupt_requested:
                        interrupted = True
                        break

                    buf = leftover + audio_chunk
                    offset = 0

                    while len(buf) - offset >= self.TWILIO_FRAME_BYTES:
                        if self._interrupt_requested:
                            interrupted = True
                            break
                        frame = buf[offset:offset + self.TWILIO_FRAME_BYTES]
                        offset += self.TWILIO_FRAME_BYTES
                        await self._send_media_frame(frame)
                        frames_sent += 1
                        await asyncio.sleep(self.TWILIO_FRAME_SECONDS)

                    if interrupted:
                        break

                    leftover = buf[offset:]

                if interrupted:
                    # Tell Twilio to discard whatever we already sent ahead
                    # of real-time that hasn't played yet, so the caller
                    # doesn't keep hearing Anne after they started talking.
                    logger.info(
                        f"Caller interrupted after {frames_sent} frames - clearing Twilio playback buffer"
                    )
                    await self.websocket.send_text(json.dumps({
                        "event": "clear",
                        "streamSid": self.stream_sid,
                    }))
                    self._interrupt_requested = False
                    return

                if leftover:
                    frame = leftover + b"\xff" * (self.TWILIO_FRAME_BYTES - len(leftover))
                    await self._send_media_frame(frame)
                    frames_sent += 1

                # Mark end of this utterance (lets Twilio report playback).
                await self.websocket.send_text(json.dumps({
                    "event": "mark",
                    "streamSid": self.stream_sid,
                    "mark": {"name": f"tts-{frames_sent}"},
                }))

                logger.info(f"Streamed {frames_sent} audio frames to Twilio")

            except Exception as e:
                logger.error(f"TTS streaming error: {e}", exc_info=True)

    async def _redirect_call_to_rep(self, reason: str = ""):
        """Hand the live call off to a support rep via Twilio's REST API.

        `reason` is whatever the caller actually needed (set by the
        Decision Agent - a guardrail miss, or the caller just asking for
        a person, about any topic at all, not just calibration). It's
        passed through so the rep's whisper announcement can say what
        the call is about instead of a fixed guess.

        Media Streams has no in-band way to bridge into <Dial> - the only
        way to do it is to replace the live call's TwiML from outside
        using calls(sid).update(). That tears down this <Connect><Stream>
        (Twilio will send a "stop" event / close this socket shortly
        after), so this is the last thing this handler does.
        """
        from call_agent.app.agents.decision_agent import GUARDRAIL_SUPPORT_PHONE

        if not self.call_sid:
            logger.error("Cannot redirect to rep - no call_sid for this call")
            return

        self._track("REDIRECT_TO_REP", reason=reason)

        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        if not account_sid or not auth_token:
            logger.error(
                "Cannot redirect to rep - TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN not set"
            )
            return

        customer_name = (
            self.session.state.get("customer_name", "the customer")
            if self.session else "the customer"
        )

        host = self.websocket.headers.get("host", "")
        path = self.websocket.url.path  # e.g. "/call-agent/media"
        prefix = path[: -len("/media")] if path.endswith("/media") else ""
        redirect_url = (
            f"https://{host}{prefix}/redirect-twiml"
            f"?customer_name={quote(customer_name, safe='')}"
            f"&reason={quote(reason, safe='')}"
        )

        logger.info(
            f"Redirecting call {self.call_sid} to rep {GUARDRAIL_SUPPORT_PHONE} "
            f"via {redirect_url}"
        )

        try:
            from twilio.rest import Client

            client = Client(account_sid, auth_token)
            await asyncio.to_thread(
                client.calls(self.call_sid).update,
                url=redirect_url,
                method="POST",
            )
        except Exception as e:
            logger.error(f"Failed to redirect call to rep: {e}", exc_info=True)
            return

        await self._send_stop()

    async def _send_stop(self):
        """End the call by closing the Media Stream socket.

        Twilio Media Streams has no server-sent "stop" event; closing the
        WebSocket ends the <Connect><Stream> and the call hangs up.
        """
        self._call_ended = True
        self._cancel_wait_timeout()
        logger.info("Ending call - closing media stream")
        try:
            await self.websocket.close()
        except Exception as e:
            logger.error(f"Error closing websocket: {e}")

    async def _async_audio_generator(self, audio_bytes: bytes):
        """Async generator that yields audio bytes."""
        yield audio_bytes