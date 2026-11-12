"""Async hub: detectors run as parallel tasks and produce the right verdicts from real audio."""
import asyncio
import time

import synth
from telephony_agent.config import Settings
from telephony_agent.detection.hub import DetectionHub
from telephony_agent.detection.signals import VK


def make(**kw):
    t0 = time.monotonic()
    got = []

    async def on_v(v):
        got.append(v)

    hub = DetectionHub(Settings(**kw), on_v, clock=lambda: time.monotonic() - t0)
    return hub, got


async def play(hub, pcm, realtime=True):
    b = synth.to_mulaw(pcm)
    for i in range(0, len(b), 800):
        hub.feed_audio(b[i:i + 800])
        if realtime:
            await asyncio.sleep(0.1)


async def test_all_detectors_run_in_parallel_tasks_for_the_whole_call():
    hub, got = make()
    hub.start()
    assert set(hub.alive()) == {"det-vad", "det-tone", "det-music", "det-progress", "det-amd", "det-text", "fusion", "ticker"}
    assert all(hub.alive().values())
    hub.feed_status({"CallStatus": "in-progress"})
    await play(hub, synth.cat(synth.silence(1), synth.tone(800, 0.4), synth.silence(0.5)), realtime=False)
    await hub.drain()
    assert all(hub.alive().values())                 # still all alive mid-call
    # each audio detector saw the same audio independently
    assert hub.vad.n > 0 and hub.tone.n > 0 and hub.music.n >= 0
    await hub.stop()
    assert not any(hub.alive().values())


async def test_voicemail_audio_plus_text_plus_amd_yields_beep_ready():
    hub, got = make()
    hub.start()
    hub.feed_status({"CallStatus": "in-progress"})
    greeting = "Hi you have reached John. Please leave a message after the tone."
    hub.feed_transcript(greeting, True)
    await play(hub, synth.cat(synth.speech(greeting), synth.silence(0.3), synth.tone(900, 0.5), synth.silence(1.0)))
    await hub.drain()
    kinds = [v.kind for v in got]
    assert VK.ANSWERED in kinds and VK.VM_GREETING in kinds and VK.VM_BEEP_READY in kinds
    beep = [v for v in got if v.kind is VK.VM_BEEP_READY]
    assert len(beep) == 1 and beep[0].data["source"] == "audio"
    await hub.stop()


async def test_hello_audio_and_text_confirm_human_exactly_once():
    hub, got = make()
    hub.start()
    hub.feed_status({"CallStatus": "in-progress"})
    await asyncio.sleep(0.05)
    hub.feed_transcript("Hello?", True)
    await play(hub, synth.cat(synth.speech("Hello?"), synth.silence(1.5)))
    await hub.drain()
    h = [v for v in got if v.kind is VK.HUMAN]
    assert len(h) == 1 and "text:human_phrase" in h[0].evidence       # rhythm path is covered in the sync rig
    await hub.stop()


async def test_ivr_menu_audio_waits_for_speaker_then_emits_menu():
    hub, got = make()
    hub.start()
    hub.feed_status({"CallStatus": "in-progress"})
    menu = "Thank you for calling Acme. For sales press one. For calibration press two."
    hub.feed_transcript("For sales press 1. For calibration press 2.", True)
    await play(hub, synth.cat(synth.speech(menu), synth.silence(2.0)))
    await hub.drain()
    ready = [v for v in got if v.kind is VK.IVR_MENU_READY]
    assert len(ready) == 1 and {o.key for o in ready[0].data["options"]} == {"1", "2"}
    # decision came after the speaker finished, not mid-menu
    assert ready[0].t >= len(synth.speech(menu)) / 8000
    await hub.stop()


async def test_hold_music_audio_gives_hold_then_person_gives_human():
    hub, got = make(music_hold_s=2.0)
    hub.start()
    hub.feed_status({"CallStatus": "in-progress"})
    await play(hub, synth.music(6))
    await hub.drain()
    assert VK.HOLD in [v.kind for v in got]
    hub.feed_transcript("Hi, this is Maria, how can I help you?", True)
    await play(hub, synth.cat(synth.speech("Hi, this is Maria, how can I help you?"), synth.silence(2)))
    await hub.drain()
    assert VK.HUMAN in [v.kind for v in got]
    await hub.stop()


async def test_signal_log_is_recorded_for_the_tags_agent():
    hub, got = make()
    hub.start()
    hub.feed_status({"CallStatus": "in-progress"})
    hub.feed_transcript("Please hold.", True)
    await hub.drain()
    assert {"src": "progress", "kind": "answered"}.items() <= {k: hub.signal_log[0][k] for k in ("src", "kind")}.items()
    assert any(e["kind"] == "text_label" and e["data"]["label"] == "hold" for e in hub.signal_log)
    await hub.stop()
