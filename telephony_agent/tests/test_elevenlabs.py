import asyncio
import json

import httpx
import numpy as np

from telephony_agent.speech import ElevenLabsScribeSTT, ElevenLabsTTS, _wrap_mulaw_as_wav
from telephony_agent.tones import mulaw_encode


def tone(secs, amp=6000):
    t = np.arange(int(8000 * secs)) / 8000
    return mulaw_encode((amp * np.sin(2 * np.pi * 300 * t)).astype(np.int16))


def quiet(secs):
    return mulaw_encode(np.zeros(int(8000 * secs), dtype=np.int16))


async def test_tts_uses_call_agent_wiring_and_caches():
    seen = []

    def handler(req: httpx.Request):
        seen.append(req)
        return httpx.Response(200, content=b"\x7f" * 400)
    tts = ElevenLabsTTS("key", "voiceX")
    tts._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert await tts.synth("Hello?") == b"\x7f" * 400
    await tts.synth("Hello?")
    assert len(seen) == 1                                   # cached
    r = seen[0]
    assert r.url.path == "/v1/text-to-speech/voiceX/stream" and r.url.params["output_format"] == "ulaw_8000"
    body = json.loads(r.content)
    assert body["model_id"] == "eleven_flash_v2_5" and body["voice_settings"]["stability"] == 0.45
    assert r.headers["xi-api-key"] == "key"


async def test_scribe_cuts_utterances_and_strips_tags():
    posts = []

    def handler(req: httpx.Request):
        posts.append(req)
        return httpx.Response(200, json={"text": f"for sales press 1 [click] n{len(posts)}"})
    stt = ElevenLabsScribeSTT("key", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await stt.start()
    stt.feed(quiet(1.0) + tone(1.5) + quiet(1.0) + tone(0.1) + quiet(1.0) + tone(1.0) + quiet(1.0))
    await asyncio.sleep(0.2)
    await stt.close()
    out = [x async for x in stt.results()]
    assert [t for t, _ in out] == ["for sales press 1  n1", "for sales press 1  n2"]   # "[click]" tag stripped
    assert len(posts) == 2                                  # 0.1 s blip is below MIN_SPEECH_MS and never sent
    assert b"model_id" in posts[0].content and b"tag_audio_events" in posts[0].content


def test_wav_wrapper_is_mulaw_riff():
    w = _wrap_mulaw_as_wav(b"\xff" * 80)
    assert w[:4] == b"RIFF" and w[8:12] == b"WAVE" and w[20:22] == b"\x07\x00" and w.endswith(b"\xff" * 80)
