# Anne Voice Agent - Deployment Package

Your LangGraph agent is now integrated as a **real-time voice calling agent** using Twilio Media Streams.

## What Changed

✅ **Your agents are UNCHANGED** — Brain, Decision, Memory agents work exactly as they did locally

✅ **Added voice capabilities:**
- `app/speech_services.py` — Google Cloud STT + ElevenLabs TTS
- `app/media_stream.py` — Twilio Media Streams WebSocket handler
- `main.py` — FastAPI entry point for Render deployment
- `requirements.txt` — Python dependencies
- `.env.example` — Environment variable template
- `scripts/test_voice_flow.py` — Local test script

✅ **Your existing API endpoints still work:**
- `GET /health` — Health check
- `POST /v1/chat/completions` — ElevenLabs custom LLM (legacy)
- `POST /start`, `/message`, `/session`, etc. — Original REST endpoints

✅ **NEW endpoint for voice:**
- `WS /media` — Twilio Media Streams WebSocket (THIS is what you use for calling)

## Flow Overview

```
Caller dials → Twilio picks up
         ↓
Twilio connects to: wss://your-server/media?customer_name=Jane&email=john@company.com...
         ↓
Your server receives audio stream
         ↓
Google Cloud STT: audio → text (customer speech)
         ↓
Your LangGraph Agent (CallSession):
  - Memory agent
  - Brain agent  
  - Decision agent
         ↓
Agent generates: response text
         ↓
ElevenLabs TTS: text → audio (Anne's voice)
         ↓
Send audio back via WebSocket to Twilio
         ↓
Twilio plays audio to caller
         ↓
Loop until agent calls END_CALL
```

## Expected Behavior

Anne's full conversation flow (unchanged from local):

1. **Anne opens:** "Hi, this is Anne from KTC Instruments calling to confirm your details registered with us. Am I speaking with [customer_name]?"

2. **If customer confirms:** Anne confirms name → email → phone number → gives calibration reminder

3. **If customer denies:** Anne asks for correct name and continues

4. **At completion:** Anne: "Thank you for your time. Have a great day!" → call ends

## Deployment Steps

### 1. Local Test (Optional)

Test that the agent logic still works:
```bash
cd call_agent
export GEMINI_API_KEY=your-key
python scripts/test_voice_flow.py
```

Expected output shows Anne's full conversation flow.

### 2. Deploy to Render

**Prerequisites:**
- GitHub repo with this code pushed
- Google Cloud service account JSON (for STT)
- ElevenLabs API key
- Gemini API key

**In Render Dashboard:**

1. Create new **Web Service**
2. Connect to your GitHub repo
3. Set Build Command:
   ```
   pip install -r requirements.txt
   ```
4. Set Start Command:
   ```
   uvicorn call_agent.main:app --host 0.0.0.0 --port 8000
   ```
5. Add Environment Variables:
   ```
   GEMINI_API_KEY=your-key
   ELEVENLABS_API_KEY=your-key
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
   PORT=8000
   ```

6. Upload Google Cloud JSON to `/credentials/service-account-key.json` in your repo

### 3. Configure Twilio

In your Twilio workflow, direct incoming calls to stream audio to:

```
wss://your-render-app.onrender.com/media?customer_name=Jane+K&email=john@company.com&phone_number=%2B14125551234
```

Or use TwiML:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://your-render-app.onrender.com/media?customer_name={{customer_name}}&email={{email}}&phone_number={{phone_number}}" />
  </Connect>
</Response>
```

### 4. Integrate with n8n

When n8n initiates a Twilio call, pass customer data in the Media Streams URL as query parameters. Your server extracts them and initializes the CallSession automatically.

## File Structure

```
call_agent/
├── main.py                      ← Run this (uvicorn call_agent.main:app)
├── requirements.txt             ← Install these
├── .env.example                 ← Copy to .env, fill in values
├── app/
│   ├── api.py                   ← Updated with /media WebSocket endpoint
│   ├── call.py                  ← Unchanged - your session manager
│   ├── media_stream.py          ← NEW - Twilio Media Streams handler
│   ├── speech_services.py       ← NEW - STT + TTS services
│   ├── agents/                  ← UNCHANGED - your agents
│   ├── graph/                   ← UNCHANGED - your LangGraph
│   └── schemas/                 ← UNCHANGED - your data schemas
├── scripts/
│   ├── graph_manual.py          ← UNCHANGED - local test script
│   ├── test_voice_flow.py       ← NEW - voice flow test
│   └── ...
└── ...
```

## Credentials Setup

### Google Cloud (Speech-to-Text)

1. Create service account in Google Cloud Console
2. Download JSON key
3. Place in `/credentials/service-account-key.json` (git-ignored)
4. Set `GOOGLE_APPLICATION_CREDENTIALS=./credentials/service-account-key.json` in `.env`

### ElevenLabs (Text-to-Speech)

1. Get API key from https://elevenlabs.io/app/settings
2. Set `ELEVENLABS_API_KEY=your-key` in `.env`

### Gemini API

1. Get API key from Google AI Studio
2. Set `GEMINI_API_KEY=your-key` in `.env`

## Health Checks

```bash
# Local test
curl http://localhost:8000/health

# Render deployment
curl https://your-render-app.onrender.com/health
```

Expected response:
```json
{
  "service": "call-agent",
  "status": "healthy"
}
```

## Troubleshooting

**"No audio output"**
- Check ElevenLabs API key and account has credits
- Check logs for TTS errors

**"Agent not responding"**
- Check Gemini API key is valid
- Check logs for agent errors
- Verify LangGraph agents still work locally

**"Audio cutting out"**
- Check Render memory/CPU (might need upgrade)
- Check Twilio Media Streams connection logs

**"STT not working"**
- Verify Google Cloud credentials file exists and is readable
- Check Speech-to-Text API is enabled in Google Cloud

## Cost Estimate

- **Google Cloud STT:** $0-5/month (free 60 min, then ~$0.024/min)
- **ElevenLabs TTS:** $5-20/month
- **Render:** $7/month (paid tier)
- **Gemini API:** $0-10/month

**Total:** ~$12-40/month

## Questions?

Refer to the script: `scripts/test_voice_flow.py` to understand the conversation flow
Refer to the handler: `app/media_stream.py` to see how audio is processed
Refer to the services: `app/speech_services.py` to understand STT/TTS integration
