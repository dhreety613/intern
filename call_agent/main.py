"""
Anne Voice Agent - Twilio Media Streams Integration

Your LangGraph agent running as a real-time voice agent.

Architecture:
- Twilio initiates call
- Caller connects to /media WebSocket endpoint
- Bidirectional audio streaming (caller ↔ server)
- STT: Google Cloud Speech-to-Text (audio → text)
- Agent: Your LangGraph CallSession (text → response)
- TTS: ElevenLabs API (response → audio)
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from call_agent.app.api import router

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Anne Voice Agent",
    description="Tensitron calibration reminder agent with voice",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router (has all endpoints: health, chat completions, WebSocket media)
app.include_router(router)


@app.on_event("startup")
async def startup():
    """Startup checks."""
    logger.info("=" * 60)
    logger.info("Anne Voice Agent Starting Up")
    logger.info("=" * 60)

    # Check required env vars
    required = ["GEMINI_API_KEY", "ELEVENLABS_API_KEY"]
    missing = [var for var in required if not os.getenv(var)]

    if missing:
        logger.warning(f"Missing environment variables: {', '.join(missing)}")
    else:
        logger.info("✓ All API credentials found")

    # Google Cloud credentials can be a path or application default
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.info("ℹ Using Google Cloud application default credentials")
    else:
        logger.info("✓ Google Cloud credentials configured")

    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown():
    """Shutdown cleanup."""
    logger.info("Anne Voice Agent Shutting Down")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
