"""Prod entrypoint: uvicorn telephony_agent.main:app --host 0.0.0.0 --port $PORT

Voice layer is wired like call_agent: ElevenLabs TTS (ulaw_8000) + ElevenLabs Scribe STT, Twilio Media Streams.
Same env vars as call_agent (TWILIO_*, ELEVENLABS_*, GEMINI_API_KEY, CALL_RESULT_WEBHOOK_*)."""
import logging
import os

from dotenv import load_dotenv

load_dotenv()                      # before Settings(): several settings read env at import time
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

from .config import Settings          # noqa: E402
from .llm import GeminiFallback       # noqa: E402
from .server import create_app        # noqa: E402
from .speech import ElevenLabsTTS, make_stt_factory   # noqa: E402

log = logging.getLogger("telephony.main")
settings = Settings()

_missing = [k for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER", "PUBLIC_BASE_URL",
                        "ELEVENLABS_API_KEY", "AGENT_API_KEY") if not os.getenv(k)]
if _missing:
    # fail closed: without AGENT_API_KEY POST /calls would let anyone dial from our Twilio number
    raise RuntimeError(f"telephony_agent: missing required env vars: {', '.join(_missing)}")

tts = ElevenLabsTTS()
llm = GeminiFallback.from_env()          # None until GEMINI_API_KEY is set -> rules only
log.info("STT=%s TTS=elevenlabs(%s/%s) LLM=%s DTMF=%s", os.getenv("STT_ENGINE", "elevenlabs"), tts.voice_id,
         tts.model_id, "gemini" if llm else "rules-only", settings.dtmf_mode)


def _load_brain_runner():
    """Once a human is confirmed the call goes to the brain (call_agent) on the same websocket.
    BRAIN_HANDOFF=0, or call_agent not importable (standalone telephony deploy) -> the handoff stays a stub."""
    if os.getenv("BRAIN_HANDOFF", "1") != "1":
        log.info("BRAIN_HANDOFF=0: human handoff is a stub")
        return None
    try:
        from call_agent.app.handoff import run_brain
    except Exception as e:                     # noqa: BLE001 - brain deps missing must never stop telephony booting
        log.warning("brain not importable (%s: %s): human handoff is a stub", type(e).__name__, e)
        return None

    async def runner(ws, sess):
        await run_brain(ws, call_id=sess.req.call_id, call_sid=sess.call_sid or "", stream_sid=sess.stream_sid or "",
                        context=sess.req.context, heard=sess.ctl.heard)
    log.info("human handoff -> brain (call_agent.app.handoff.run_brain)")
    return runner


brain_runner = _load_brain_runner()
app = create_app(settings, stt_factory=make_stt_factory(), tts=tts, llm=llm, brain_runner=brain_runner)


@app.get("/health")
def health():
    return {"service": "telephony-agent", "status": "healthy", "build": "brain-handoff-1", "brain": brain_runner is not None}


@app.on_event("startup")
async def _warm() -> None:
    await tts.warm([settings.voicemail_text(), settings.screening_text(), "Hello?", "Representative"])
