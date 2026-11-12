"""Generate 10 labelled phone-call recordings (espeak speech + synthetic beeps / tones / music) with a
manifest of timed transcript events and expected detections.  Run:  python tools/make_samples.py samples/
These are SYNTHETIC (robotic voice, clean tones) - a smoke set, not a substitute for real carrier audio."""
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import synth                                                    # noqa: E402
from telephony_agent.tones import mulaw_decode, mulaw_encode    # noqa: E402

SR = 8000
STT_DELAY = 0.5          # final transcript arrives this long after the utterance ends


class Call:
    def __init__(self):
        self.parts, self.t = [], 0.0
        self.events = [{"t": 0.0, "type": "status", "status": "in-progress"}]

    def _add(self, pcm):
        self.parts.append(pcm)
        self.t += len(pcm) / SR

    def sil(self, sec):
        self._add(synth.silence(sec))
        return self

    def say(self, text, stt=None, speed=150, voice="en-us"):
        """Speak `text`; STT final (default: same text) lands STT_DELAY after the utterance ends."""
        self._add(synth.speech(text, voice=voice, speed=speed))
        self.events.append({"t": round(self.t + STT_DELAY, 2), "type": "text", "text": stt or text, "final": True})
        return self

    def beep(self, f=1000, sec=0.45):
        self._add(synth.tone(f, sec))
        return self

    def music(self, sec):
        self._add(synth.music(sec))
        return self

    def busy(self, cycles=5):
        for _ in range(cycles):
            self._add(synth.dual_tone(480, 620, 0.5))
            self._add(synth.silence(0.5))
        return self

    def amd(self, kind):
        self.events.append({"t": round(self.t, 2), "type": "amd", "answered_by": kind})
        return self

    def save(self, path):
        pcm = np.concatenate(self.parts).astype(np.int16)
        # phone-line realism: mu-law codec round trip + faint line noise
        noisy = (pcm.astype(np.float32) + np.random.default_rng(7).normal(0, 12, len(pcm))).clip(-32768, 32767).astype(np.int16)
        out = mulaw_decode(mulaw_encode(noisy))
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(out.tobytes())


def build(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    S = []

    def add(name, desc, call: Call, expect):
        call.save(out / name)
        S.append({"file": name, "description": desc, "duration_s": round(call.t, 1), "events": call.events, "expect": expect})

    add("01_voicemail_beep.wav", "Voicemail greeting, then a beep. No Twilio AMD verdict: our own transcript + beep detection must trigger the message.",
        Call().sil(0.8).say("Hi, you have reached John Smith. I can't come to the phone right now. Please leave a message after the tone.",
                            stt="Hi, you've reached John Smith. I can't come to the phone right now. Please leave a message after the tone.")
        .sil(0.4).beep(900).sil(1.5),
        {"verdicts": ["answered", "vm_greeting", "vm_beep_ready"], "never": ["human"], "outcome": "voicemail_left", "actions": ["Speak", "Hangup"]})

    add("02_voicemail_mailbox_full.wav", "Voicemail that cannot take messages. Must hang up WITHOUT leaving a message.",
        Call().sil(0.6).say("The mailbox is full and cannot accept any messages at this time. Goodbye.").sil(1.0),
        {"verdicts": ["mailbox"], "never": ["vm_beep_ready", "human"], "outcome": "voicemail_full"})

    add("03_personal_greeting_then_voicemail.wav", "Hard case: personal greeting that STARTS like a person ('Hi, this is John') before it becomes a voicemail. Must NOT be treated as a human.",
        Call().sil(0.5).say("Hi, this is John Smith.").sil(0.5)
        .say("I'm not available right now.").sil(0.4)
        .say("Please leave a message after the beep and I will get back to you.").amd("machine_start").sil(0.4).beep(800).sil(1.5),
        {"verdicts": ["vm_greeting", "vm_beep_ready"], "never": ["human"], "outcome": "voicemail_left"})

    menu = "Thank you for calling Acme Instruments. For sales, press one. For calibration services, press two. For billing, press three."
    add("04_ivr_menu.wav", "Phone menu. Must wait for the menu to finish, then press 2 (calibration).",
        Call().sil(0.5).say(menu, stt="Thank you for calling Acme Instruments. For sales, press 1. For calibration services, press 2. For billing, press 3.").sil(3.0),
        {"verdicts": ["ivr_menu_ready"], "never": ["human"], "actions": ["Dtmf:2"]})

    m5 = "For hours and location, press one. For directions, press two. Press zero at any time to reach the operator."
    t5 = "For hours and location, press 1. For directions, press 2. Press 0 at any time to reach the operator."
    add("05_ivr_operator_fallback.wav", "Menu with no useful option, repeated. No human option: must press 0 for the operator.",
        Call().sil(0.5).say(m5, stt=t5).sil(3.5).say(m5, stt=t5).sil(3.5).say(m5, stt=t5).sil(3.0),
        {"verdicts": ["ivr_menu_ready"], "actions": ["Dtmf:0"]})

    add("06_hold_music_then_human.wav", "'Please hold', 8s of music, then a person. Must stay silent through hold, hand off only when the person speaks.",
        Call().sil(0.5).say("Please hold while we connect your call.").music(8).sil(0.3)
        .say("Hi, this is Maria. How can I help you?").sil(2.5),
        {"verdicts": ["hold", "human"], "outcome": "human_confirmed"})

    add("07_call_screening.wav", "Call-screening prompt. Must state our name after the beep, wait through 'hold on', then confirm the person.",
        Call().sil(0.5).say("Hi. The person you are calling is using a screening service. Please say your name after the tone.")
        .sil(0.3).beep(1000).sil(2.5).say("Hold on while I try to connect you.").music(3).sil(0.3).say("Hello?").sil(2.5),
        {"verdicts": ["screen_prompt", "screen_speak_ready", "hold", "human"], "outcome": "human_confirmed", "actions": ["Speak"]})

    add("08_carrier_not_in_service.wav", "Carrier recording. Must hang up and tag the number as invalid.",
        Call().sil(0.4).say("We're sorry. The number you have dialed is not in service. Please check the number and dial again.").sil(1.0),
        {"verdicts": ["carrier"], "outcome": "invalid_number", "never": ["human", "vm_beep_ready"]})

    add("09_busy_tone.wav", "Busy signal (480+620 Hz, 0.5s on/off). Must be recognised from the audio alone, with no transcript.",
        Call().sil(0.3).busy(5),
        {"verdicts": ["busy_tone"], "outcome": "busy"})

    add("10_human_hello.wav", "A person answers 'Hello?' and waits. Must confirm a human after they finish and go quiet (not while talking).",
        Call().sil(1.2).say("Hello?").sil(3.0),
        {"verdicts": ["human"], "outcome": "human_confirmed", "never": ["vm_greeting", "ivr_menu_ready"]})

    (out / "manifest.json").write_text(json.dumps({"note": "synthetic samples; see tools/make_samples.py", "samples": S}, indent=1))
    return S


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "samples")
    s = build(out)
    print(f"wrote {len(s)} recordings + manifest.json to {out}/")
