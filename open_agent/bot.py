"""
open_agent/bot.py

Pipecat-based voice pipeline for the Anne / KTC Instruments calibration
confirmation call.

Same four external services as call_agent (Twilio, Gemini, ElevenLabs TTS,
ElevenLabs STT), but built on Pipecat's own turn-detection, VAD, and
LLM-call retry/timeout handling instead of the hand-rolled versions in
call_agent/app/media_stream.py. This is a fresh, standalone pipeline - it
does not call into call_agent's LangGraph brain.
"""

import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.elevenlabs.stt import ElevenLabsSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="INFO")


def _build_system_instruction(
    customer_name: str, email: str, phone_number: str, calibration_dates: str
) -> str:
    return (
        "You are Anne, a phone agent for KTC Instruments calling to confirm a "
        "customer's contact details and remind them of an upcoming calibration. "
        "Your replies are spoken aloud over the phone by a TTS engine, so keep "
        "every reply short and conversational - one or two sentences, no "
        "markdown, no lists, no special characters.\n\n"
        f"Customer on file: name={customer_name!r}, email={email!r}, "
        f"phone={phone_number!r}, next calibration date={calibration_dates!r}.\n\n"
        "Follow this flow, one step at a time, waiting for the customer's reply "
        "before moving to the next step:\n"
        "1. Greet them by name and ask you're speaking with them.\n"
        "2. Once name is confirmed (or corrected), confirm the email on file.\n"
        "3. Once email is confirmed (or corrected), confirm the phone number on file.\n"
        "4. Once phone is confirmed (or corrected), remind them of the calibration date.\n"
        "5. Ask if there's anything else, then thank them and say goodbye.\n\n"
        "If they correct a detail, acknowledge the correction and use the "
        "corrected value from then on. If they go off topic, answer briefly and "
        "gently steer back to the current step. Never skip a step, and never "
        "re-ask something already confirmed."
    )


async def run_bot(
    transport: BaseTransport,
    handle_sigint: bool,
    customer_name: str,
    email: str,
    phone_number: str,
    calibration_dates: str,
):
    llm = GoogleLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
        settings=GoogleLLMService.Settings(
            model=os.getenv("OPEN_AGENT_GEMINI_MODEL", "gemini-3.5-flash-lite"),
            system_instruction=_build_system_instruction(
                customer_name, email, phone_number, calibration_dates
            ),
        ),
    )

    aiohttp_session = aiohttp.ClientSession()

    stt = ElevenLabsSTTService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        aiohttp_session=aiohttp_session,
        params=ElevenLabsSTTService.InputParams(tag_audio_events=False),
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),  # Twilio audio in
            stt,  # ElevenLabs speech-to-text
            user_aggregator,
            llm,  # Gemini
            tts,  # ElevenLabs text-to-speech
            transport.output(),  # Twilio audio out
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    runner = WorkerRunner(handle_sigint=handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Twilio call connected - Anne is starting")
        # No user message yet - LLMRunFrame triggers the LLM against the
        # system instruction alone, so Anne speaks first with the greeting.
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Call ended")
        await aiohttp_session.close()
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point called for every new Twilio Media Stream connection."""

    transport_params = {
        "twilio": lambda: FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    }

    # create_transport auto-detects Twilio from the WebSocket handshake,
    # builds the TwilioFrameSerializer (using TWILIO_ACCOUNT_SID /
    # TWILIO_AUTH_TOKEN), and exposes the <Parameter> values from our TwiML
    # as runner_args.call_data.body.
    transport = await create_transport(runner_args, transport_params)

    call_data = runner_args.call_data
    body = call_data.body if call_data else {}

    customer_name = body.get("customer_name") or "Customer"
    email = body.get("email") or ""
    phone_number = body.get("phone_number") or ""
    calibration_dates = body.get("calibration_dates") or ""

    logger.info(
        f"Call metadata - customer={customer_name!r} email={email!r} "
        f"phone={phone_number!r} calibration={calibration_dates!r}"
    )

    await run_bot(
        transport,
        runner_args.handle_sigint,
        customer_name,
        email,
        phone_number,
        calibration_dates,
    )
