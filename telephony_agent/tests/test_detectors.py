"""Detectors on real-ish audio: espeak-ng speech + synthetic beeps/tones/music."""
import synth
from telephony_agent.detection.audio import MusicDetector, ToneDetector
from telephony_agent.detection.signals import Sig
from telephony_agent.detection.vad import VadDetector, silero_available


def run(det, pcm):
    out, b = [], synth.to_mulaw(pcm)
    for i in range(0, len(b), 800):
        out += det.feed(b[i:i + 800])
    return out


def kinds(sigs):
    return [s.kind for s in sigs]


# ---- VAD / rhythm ----
def test_vad_short_hello_is_human_rhythm():
    sigs = run(VadDetector(), synth.cat(synth.silence(1), synth.speech("Hello?"), synth.silence(2)))
    assert kinds(sigs) == [Sig.SPEECH_START, Sig.SPEECH_END, Sig.RHYTHM]
    assert sigs[-1].data["verdict"] == "human"
    assert 0.9 < sigs[0].t < 1.3                                   # onset where the speech starts


def test_vad_long_greeting_is_machine_rhythm():
    g = synth.speech("Hi, you have reached John Smith. I am not able to come to the phone right now. "
                     "Please leave your name and number after the tone and I will call you back.", speed=130)
    sigs = run(VadDetector(), synth.cat(synth.silence(0.5), g, synth.silence(1)))
    assert any(s.kind is Sig.RHYTHM and s.data["verdict"] == "machine" for s in sigs)
    assert not any(s.kind is Sig.RHYTHM and s.data["verdict"] == "human" for s in sigs)


def test_vad_ignores_noise_music_and_beeps():
    for pcm in (synth.noise(3), synth.music(4), synth.tone(1000, 1.0)):
        assert run(VadDetector(), pcm) == []


# ---- tones ----
def test_beep_fires_after_tone_ends_not_during():
    sigs = run(ToneDetector(), synth.cat(synth.silence(1), synth.tone(1000, 0.4), synth.silence(1)))
    assert kinds(sigs) == [Sig.BEEP_END]
    assert sigs[0].t >= 1.4 and abs(sigs[0].data["freq"] - 1000) < 60


def test_beep_after_spoken_greeting():
    pcm = synth.cat(synth.speech("Please leave a message after the tone"), synth.silence(0.3), synth.tone(800, 0.5), synth.silence(1))
    sigs = run(ToneDetector(), pcm)
    assert kinds(sigs) == [Sig.BEEP_END]
    assert sigs[0].t > len(synth.speech("Please leave a message after the tone")) / 8000


def test_speech_and_short_blips_are_not_beeps():
    assert run(ToneDetector(), synth.speech("Please leave a message after the tone, and I will call you back")) == []
    assert run(ToneDetector(), synth.cat(synth.tone(1000, 0.04), synth.silence(1))) == []


def test_busy_and_reorder_cadence():
    busy = synth.cat(*[synth.cat(synth.dual_tone(480, 620, 0.5), synth.silence(0.5)) for _ in range(4)])
    fast = synth.cat(*[synth.cat(synth.dual_tone(480, 620, 0.25), synth.silence(0.25)) for _ in range(6)])
    s = run(ToneDetector(), busy)
    assert kinds(s) == [Sig.TONE_BUSY] and s[0].data["cadence"] == "busy"
    s = run(ToneDetector(), fast)
    assert kinds(s) == [Sig.TONE_BUSY] and s[0].data["cadence"] == "reorder"


def test_fax_cng_repeats_gain_confidence():
    fax = synth.cat(synth.tone(1100, 0.5), synth.silence(2.5), synth.tone(1100, 0.5), synth.silence(1))
    s = run(ToneDetector(), fax)
    assert kinds(s) == [Sig.TONE_FAX, Sig.TONE_FAX] and s[0].conf < 0.8 <= s[1].conf


# ---- music ----
def test_hold_music_on_then_off_when_a_person_speaks():
    if not silero_available():
        return
    pcm = synth.cat(synth.music(8), synth.speech("Hello, this is Maria, how can I help you"), synth.silence(2))
    s = run(MusicDetector(), pcm)
    assert kinds(s) == [Sig.MUSIC_ON, Sig.MUSIC_OFF]
    assert s[0].t < 3 and 8 <= s[1].t < 10.5


def test_speech_menu_and_silence_are_not_music():
    if not silero_available():
        return
    menu = synth.speech("For sales press one. For support press two. To speak to an operator press zero.")
    assert run(MusicDetector(), synth.cat(menu, synth.silence(2))) == []
    assert run(MusicDetector(), synth.noise(6)) == []
