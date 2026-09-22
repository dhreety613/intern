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

import logging as std_logging
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger

# Pipecat uses loguru; the underlying google-genai/httpx calls use the
# stdlib logging module. These are independent systems - make sure
# httpx's request-level logs (the concrete evidence of whether a Gemini
# call was even attempted) aren't silently suppressed by whatever level
# some other part of the stack leaves the "httpx" logger at.
std_logging.getLogger("httpx").setLevel(std_logging.INFO)
if not std_logging.getLogger().handlers:
    std_logging.basicConfig(
        level=std_logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndWorkerFrame, LLMRunFrame
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
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="INFO")


def _build_system_instruction(
    customer_name: str,
    email: str,
    phone_number: str,
    calibration_dates: str,
    product: str,
) -> str:
    product_clause = f" for the {product}" if product else ""
    return (
        "You are Anne, a phone agent for KTC Instruments calling to confirm a "
        "customer's contact details and remind them of an upcoming calibration. "
        "Your replies are spoken aloud over the phone by a TTS engine, so keep "
        "every reply short and conversational - one or two sentences, no "
        "markdown, no lists, no special characters.\n\n"
        f"Customer on file: name={customer_name!r}, email={email!r}, "
        f"phone={phone_number!r}, next calibration date={calibration_dates!r}, "
        f"product={product!r}.\n\n"
        "Follow this flow, one step at a time, waiting for the customer's reply "
        "before moving to the next step:\n"
        "1. Greet them and ask if you're speaking with "
        f"{customer_name}.\n"
        "   - If they confirm, move to step 2.\n"
        "   - If they say you have the wrong person, ask once who you're "
        "actually speaking with. If they give a name, treat that as the "
        "corrected name and continue to step 2. If they still can't confirm "
        "they're the right person (e.g. this really is a wrong number), "
        "apologize for the inconvenience, say a polite goodbye, and call "
        "end_call.\n"
        "2. Once name is confirmed, confirm the email on file.\n"
        "3. Once email is confirmed (or corrected), confirm the phone number on file.\n"
        f"4. Once phone is confirmed (or corrected), remind them of the "
        f"calibration date{product_clause}.\n"
        "5. Ask if there's anything else. When they say no / that's all, "
        "thank them, say goodbye, and call end_call.\n\n"
        "If they correct a detail (name, email, or phone), acknowledge the "
        "correction and use the corrected value from then on - never re-ask "
        "something already confirmed, and never skip a step.\n\n"
        "If they go off topic (small talk, unrelated questions, etc.), answer "
        "briefly and gently steer back to the current step - this counts as "
        "one redirect. If you have redirected the caller back on topic 5 or 6 "
        "times over the course of the call and they keep going off topic, stop "
        "trying: say a polite goodbye explaining you'll try again another time, "
        "and call end_call.\n\n"
        "Always say your goodbye line out loud BEFORE calling end_call - "
        "end_call is how you actually hang up once you've said it, in all "
        "three cases above (wrong person, too many redirects, or normal "
        "completion)."
    )


async def end_call(params: FunctionCallParams):
    """End the phone call now.

    Call this immediately after you have said your goodbye line out loud -
    whether that's because the call finished normally, the caller was
    confirmed to be the wrong person, or you've redirected them back on
    topic too many times without success.
    """
    await params.result_callback({"status": "call ending"})
    # "downstream" (the default) flushes already-queued frames first, so the
    # goodbye TTS audio finishes playing before the pipeline actually closes
    # the Twilio connection.
    await params.pipeline_worker.queue_frame(EndWorkerFrame())


async def run_bot(
    transport: BaseTransport,
    handle_sigint: bool,
    customer_name: str,
    email: str,
    phone_number: str,
    calibration_dates: str,
    product: str,
):
    llm = GoogleLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
        settings=GoogleLLMService.Settings(
            model=os.getenv("OPEN_AGENT_GEMINI_MODEL", "gemini-3.5-flash-lite"),
            system_instruction=_build_system_instruction(
                customer_name, email, phone_number, calibration_dates, product
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

    context = LLMContext(tools=[end_call])
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
        logger.info("on_client_connected fired")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Call ended")
        await aiohttp_session.close()
        await runner.cancel()

    # Kick off the conversation before starting the runner, so there's no
    # ambiguity about whether this frame is queued before the pipeline's
    # consumer loop exists (two prior real-call attempts, queuing from
    # on_client_connected instead, produced zero LLM/TTS activity and zero
    # errors - total silence until the caller gave up and hung up).
    #
    # A bare system_instruction with zero messages also gives the LLM
    # nothing to respond to, so add an actual message before LLMRunFrame.
    logger.info("Queuing initial LLMRunFrame before starting the runner")
    context.add_message({"role": "user", "content": "(The call has just connected. Begin.)"})
    await worker.queue_frames([LLMRunFrame()])
    logger.info("Initial LLMRunFrame queued - starting runner")

    await runner.run()
    logger.info("runner.run() returned - call is over")


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
    product = body.get("product") or ""

    logger.info(
        f"Call metadata - customer={customer_name!r} email={email!r} "
        f"phone={phone_number!r} calibration={calibration_dates!r} product={product!r}"
    )

    await run_bot(
        transport,
        runner_args.handle_sigint,
        customer_name,
        email,
        phone_number,
        calibration_dates,
        product,
    )
