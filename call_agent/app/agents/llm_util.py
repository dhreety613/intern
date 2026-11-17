"""Gemini client with a per-request deadline and one retry.

One stalled request used to freeze a whole turn (a live call saw a 24 s agent turn while the others took 1-4 s).
Every attempt now has a deadline (ANNE_LLM_TIMEOUT_SECONDS, default 12) and a failed or slow attempt is retried
(ANNE_LLM_RETRIES, default 1) before the error is raised to the caller, which already has its own fallbacks.
"""

import logging
import os

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

LLM_TIMEOUT_S = float(os.getenv("ANNE_LLM_TIMEOUT_SECONDS", "12"))
LLM_RETRIES = int(os.getenv("ANNE_LLM_RETRIES", "1"))


def make_client(api_key):
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(LLM_TIMEOUT_S * 1000)))


def generate(client, **kwargs):
    """client.models.generate_content(**kwargs), retried on error/timeout."""
    last = None
    for attempt in range(LLM_RETRIES + 1):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as exc:  # noqa: BLE001 - any SDK/network failure
            last = exc
            logger.warning("Gemini call failed or timed out (attempt %d/%d): %s", attempt + 1, LLM_RETRIES + 1, exc)
    raise last
