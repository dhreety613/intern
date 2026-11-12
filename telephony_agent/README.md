# Telephony agent (Anne) - pre-human layer

Dials, detects what answered, handles it, and hands off to the brain only once a human is confirmed.

    detectors (parallel tasks) -> fusion (verdicts) -> controller (actions) -> Twilio / ElevenLabs
    vad, tone, music, text, amd, progress      hub.py runs them all as asyncio tasks

## Voice layer (wired like call_agent)
    TTS  ElevenLabs REST /stream, ulaw_8000, eleven_flash_v2_5 (ELEVENLABS_VOICE_ID / _MODEL_ID / _STABILITY ...)
    STT  ElevenLabs Scribe, energy-VAD utterances (STT_ENGINE=google switches to Google streaming STT)
    Twilio  REST call with async AMD (DetectMessageEnd), dual-channel recording, Media Stream at /media/{call_id},
            DTMF in-band over the stream (DTMF_MODE=twilio uses REST <Play digits> + reconnect instead)
    Handoff to the brain is a stub: person confirmed -> log, hold HANDOFF_STUB_HOLD_S (6 s), hang up.

## Run
    pip install -r requirements.txt && pip install --no-deps silero-vad
    uvicorn telephony_agent.main:app --host 0.0.0.0 --port $PORT      # needs a public https URL for Twilio

Required env: TWILIO_ACCOUNT_SID TWILIO_AUTH_TOKEN TWILIO_FROM_NUMBER PUBLIC_BASE_URL ELEVENLABS_API_KEY AGENT_API_KEY
              (startup refuses to run without AGENT_API_KEY: POST /calls dials from our Twilio number)
Optional env: ELEVENLABS_VOICE_ID GEMINI_API_KEY (LLM fallback + IVR key pick) CALLBACK_NUMBER STT_ENGINE DTMF_MODE
              RECORD_CALLS CALL_RESULT_WEBHOOK_URL / CALL_RESULT_WEBHOOK_SECRET (outcome for the tags agent, same as call_agent)

Place a call:   POST {PUBLIC_BASE_URL}/calls   header X-API-Key: $AGENT_API_KEY   {"to": "+1...", "extension": "123", "context": {"person_id": "", "deal_id": ""}}
Read the result: GET /calls/{call_id}  (outcome, state, action log, detector signal log, transcript heard)
Logs: "CALL RESULT: {...}" is printed at call end.

## Test / score
    pytest -q                                   # 100+ tests (hub tests run in real time, ~30 s)
    python tools/make_samples.py samples/       # 10 synthetic labelled calls
    python -m telephony_agent.replay samples/manifest.json -v   # score + timeline
Replay works on real recordings too: add a wav + timed transcript events to a manifest.
