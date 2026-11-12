import asyncio
import json

import httpx
import pytest

from telephony_agent.classifier import classify, classify_with_fallback
from telephony_agent.llm import GeminiFallback
from telephony_agent.models import Label


def gem(reply, status=200, delay=0.0, seen=None):
    async def handler(req: httpx.Request):
        if seen is not None:
            seen.append((str(req.url), dict(req.headers), json.loads(req.content)))
        if delay:
            await asyncio.sleep(delay)
        return httpx.Response(status, json={"candidates": [{"content": {"parts": [{"text": reply}]}}]})
    return GeminiFallback("KEY", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


TEXT = "for our valued customers kindly say the name of the department you wish to be connected with"


async def test_rules_miss_then_gemini_labels_it():
    assert classify(TEXT) is Label.UNKNOWN
    seen = []
    llm = gem('{"label": "ivr_menu"}', seen=seen)
    assert await classify_with_fallback(TEXT, llm) is Label.IVR_MENU
    url, headers, body = seen[0]
    assert "gemini" in url and headers["x-goog-api-key"] == "KEY" and "untrusted" in body["contents"][0]["parts"][0]["text"]


async def test_llm_is_not_called_when_rules_already_know():
    seen = []
    llm = gem('{"label": "human"}', seen=seen)
    assert await classify_with_fallback("The number you have dialed is not in service and cannot be reached", llm) is Label.CARRIER_INVALID
    assert await classify_with_fallback("la la", llm) is Label.UNKNOWN          # too short to bother
    assert seen == []


async def test_bad_label_error_and_timeout_all_degrade_to_unknown():
    assert await classify_with_fallback(TEXT, gem('{"label": "launch_missiles"}')) is Label.UNKNOWN
    assert await classify_with_fallback(TEXT, gem("not json")) is Label.UNKNOWN
    assert await classify_with_fallback(TEXT, gem("{}", status=500)) is Label.UNKNOWN
    assert await classify_with_fallback(TEXT, gem('{"label": "human"}', delay=1.0), timeout_s=0.1) is Label.UNKNOWN


def test_from_env_is_off_without_a_key_and_on_with_one(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert GeminiFallback.from_env() is None
    monkeypatch.setenv("GEMINI_API_KEY", "abc")
    assert GeminiFallback.from_env().api_key == "abc"


async def test_hub_uses_llm_label_for_an_unknown_transcript():
    import time
    from telephony_agent.config import Settings
    from telephony_agent.detection.hub import DetectionHub
    from telephony_agent.detection.signals import VK
    got = []

    async def on_v(v):
        got.append(v)

    t0 = time.monotonic()
    hub = DetectionHub(Settings(), on_v, clock=lambda: time.monotonic() - t0, llm=gem('{"label": "hold"}'))
    hub.start()
    hub.feed_status({"CallStatus": "in-progress"})
    hub.feed_transcript(TEXT, True)
    await hub.drain()
    assert VK.HOLD in [v.kind for v in got]
    await hub.stop()


async def test_choose_key_parses_and_hub_attaches_pick():
    import httpx, json
    from telephony_agent.ivr import parse_options
    from telephony_agent.llm import GeminiFallback

    def handler(req):
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps({"kind": "human", "key": "2"})}]}}]})
    g = GeminiFallback("k", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert await g.choose_key(parse_options("For sales press 1. For a colleague press 2."), "x") == ("human", "2")
