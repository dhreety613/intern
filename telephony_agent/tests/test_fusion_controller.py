from rig import Rig, kinds
from telephony_agent.detection.signals import Sig, VK
from telephony_agent.models import (AnsweredBy, Dtmf, HandoffToBrain, Hangup, Outcome, Speak, State)

MENU = "Thank you for calling Acme. For sales, press 1. For calibration services, press 2. For billing, press 3."
HUMAN_Q = "Hi, this is Maria, how can I help you?"


def handed_off(acts):
    return any(isinstance(a, HandoffToBrain) for a in acts)


# ---------------- pre-answer ----------------
def test_pre_answer_statuses_become_outcomes():
    for st, kw, oc in [("busy", {}, Outcome.BUSY), ("no-answer", {}, Outcome.NO_ANSWER),
                       ("failed", {"SipResponseCode": "404"}, Outcome.INVALID_NUMBER),
                       ("failed", {"SipResponseCode": "503"}, Outcome.UNREACHABLE),
                       ("failed", {}, Outcome.FAILED), ("canceled", {}, Outcome.CANCELED)]:
        r = Rig(answered=False)
        r.status("ringing")
        assert r.c.state.value == "dialing" and r.c.outcome is None
        assert r.status(st, **kw) == [] and r.c.outcome is oc and r.c.state is State.ENDED


def test_remote_hangup_after_answer():
    r = Rig()
    r.status("completed")
    assert r.c.outcome is Outcome.REMOTE_HANGUP


# ---------------- voicemail ----------------
def test_voicemail_waits_for_amd_beep_then_short_message_then_hangup():
    r = Rig()
    r.say("Hi you've reached John, leave a message after the tone.")
    assert r.c.state is State.VOICEMAIL and not r.out
    a = r.amd(AnsweredBy.MACHINE_END_BEEP)
    assert isinstance(a[0], Speak) and "Tensitron" in a[0].text and "calibration" in a[0].text
    assert len(a[0].text.split()) < 40 and r.c.state is State.LEAVING_MESSAGE
    r.say("transcript while we talk is ignored")
    assert r.speech_done() == [Hangup(Outcome.VOICEMAIL_LEFT)]


def test_voicemail_own_beep_detection_when_amd_late():
    r = Rig()
    r.amd(AnsweredBy.MACHINE_START)
    r.say("Please leave a message after the tone")
    assert r.wait(0.5) == []
    assert kinds(r.sig(Sig.BEEP_END, freq=1000, dur=0.4)) == ["Speak"]


def test_voicemail_speaks_once_even_if_amd_and_audio_both_fire():
    r = Rig()
    r.say("Leave a message after the tone")
    a = r.amd(AnsweredBy.MACHINE_END_BEEP) + r.sig(Sig.BEEP_END, freq=900, dur=0.4)
    assert kinds(a) == ["Speak"]


def test_voicemail_machine_end_silence_speaks_after_short_delay():
    r = Rig()
    r.say("The person you are calling is not available. Leave a message.")
    r.amd(AnsweredBy.MACHINE_END_SILENCE)
    assert kinds(r.wait(1.0)) == ["Speak"]


def test_voicemail_tone_announced_but_no_beep_falls_back_on_silence():
    r = Rig()
    r.say("Leave a message after the tone")
    assert kinds(r.wait(6)) == ["Speak"]


def test_voicemail_waits_while_greeting_is_still_playing():
    r = Rig()
    r.say("Leave a message after the tone")
    r.sig(Sig.SPEECH_START)                          # VAD says the greeting is still going
    assert r.wait(8) == []


def test_mailbox_full_and_disabled_hang_up_without_message():
    r = Rig()
    r.say("Leave a message after the tone")
    assert r.say("The mailbox is full and cannot accept any messages.") == [Hangup(Outcome.VOICEMAIL_FULL)]
    assert not any(isinstance(x, Speak) for x in r.out)
    assert Rig().say("The voicemail box has not been set up.") == [Hangup(Outcome.VOICEMAIL_DISABLED)]


# ---------------- carrier / tones ----------------
def test_carrier_messages():
    for text, oc in [("The number you have dialed is not in service.", Outcome.INVALID_NUMBER),
                     ("The number has been disconnected or is no longer in service", Outcome.DISCONNECTED),
                     ("The subscriber you are calling is currently switched off", Outcome.UNREACHABLE)]:
        assert Rig().say(text) == [Hangup(oc)]


def test_busy_and_fast_busy_tones_and_fax():
    assert Rig().sig(Sig.TONE_BUSY, cadence="busy") == [Hangup(Outcome.BUSY)]
    assert Rig().sig(Sig.TONE_BUSY, cadence="reorder") == [Hangup(Outcome.UNREACHABLE)]
    assert Rig().sig(Sig.TONE_FAX, 0.9, tone="CNG") == [Hangup(Outcome.INVALID_NUMBER)]
    assert Rig().sig(Sig.TONE_FAX, 0.5, tone="CNG") == []          # one CNG blip is not enough
    assert Rig().amd(AnsweredBy.FAX) == [Hangup(Outcome.INVALID_NUMBER)]


def test_busy_tone_ignored_deep_in_ivr():
    r = Rig()
    r.say(MENU)
    assert r.sig(Sig.TONE_BUSY, cadence="busy") == []


# ---------------- IVR ----------------
def test_ivr_presses_best_key_after_menu_settles():
    r = Rig()
    r.say(MENU)
    assert r.wait(0.5) == []
    assert r.wait(1.5) == [Dtmf("2")]


def test_ivr_does_not_press_while_menu_is_still_being_spoken():
    r = Rig()
    r.sig(Sig.SPEECH_START)
    r.say(MENU)
    assert r.wait(4) == []                           # VAD: speaker not done
    r.sig(Sig.SPEECH_END, dur=6)
    assert r.wait(1.5) == [Dtmf("2")]


def test_ivr_joins_split_transcripts_before_deciding():
    r = Rig()
    r.say("For hours press 1.")
    r.t += 0.4
    r.say("To speak with a representative press 4.")
    assert r.wait(2) == [Dtmf("4")]


def test_ivr_no_human_option_presses_zero_then_voicemail_option_then_ends():
    nomatch = "For hours and directions press 1. For store locations press 2. Press zero at any time for the operator."
    r = Rig()
    r.say(nomatch)
    assert r.wait(2) == [Dtmf("0")]                                   # no human route -> operator
    r.t += 3
    assert r.say("That is not a valid entry.") == []                  # let it repeat once...
    r2 = Rig()
    r2.say(nomatch)
    r2.wait(2)
    r2.t += 3
    r2.say(nomatch)
    assert any(isinstance(a, Hangup) for a in r2.wait(2))             # 0 already tried, no voicemail option -> end


def test_ivr_voicemail_option_used_after_zero():
    r = Rig()
    m = "For store hours press 1. To leave a message press 3."
    r.say(m)
    assert r.wait(2) == [Dtmf("0")]
    r.t += 3
    r.say(m)
    assert r.wait(2) == [Dtmf("3")]


def test_ivr_human_option_beats_zero():
    r = Rig()
    r.say("For hours press 1. To speak with an officer press 2.")
    assert r.wait(2) == [Dtmf("2")]


def test_ivr_invalid_entry_retries_then_zero():
    r = Rig()
    r.say(MENU)
    r.wait(2)
    r.say("That is not a valid entry.")
    r.say("That is not a valid entry.")
    assert r.say("That is not a valid entry.") == [Dtmf("0")]


def test_ivr_menu_repeating_after_a_key_press_is_not_pressed_again():
    r = Rig()
    r.say(MENU)
    assert r.wait(2) == [Dtmf("2")]
    r.t += 3
    r.say(MENU)
    assert r.wait(2) == []                            # same menu back: key didn't take; retry, don't mash
    assert r.verdicts[-1].data["repeat"] is True


def test_ivr_extension_prompt_uses_extension_else_operator():
    assert Rig(ext="214").say("Please enter the extension of the person you wish to reach") == [Dtmf("214#")]
    assert Rig(ivr_retries=0).say("Please enter the extension of the person you wish to reach") == [Dtmf("0")]


def test_speech_ivr_says_representative():
    r = Rig()
    r.say("In a few words, tell me why you're calling today.")
    assert r.wait(2) == [Speak("Representative")]


def test_ivr_to_human_after_key():
    r = Rig()
    r.say(MENU)
    r.wait(2)
    r.say("Hello, this is Dave speaking")
    assert handed_off(r.wait(1.5))


# ---------------- hold ----------------
def test_hold_stays_silent_until_a_person_comes_on():
    r = Rig()
    r.say("Please hold while we connect your call.")
    assert r.c.state is State.HOLD
    for _ in range(5):
        r.say("la la la smooth jazz lyrics")
        r.say("Your call is important to us. Please continue to hold.")
    assert r.wait(120) == [] and not r.out
    r.say(HUMAN_Q)
    assert handed_off(r.wait(1.5))


def test_hold_timeout():
    r = Rig(hold_max_s=30)
    r.say("Please hold.")
    assert r.wait(40) == [Hangup(Outcome.HOLD_TIMEOUT)]


def test_silent_hold_music_is_detected_as_hold():
    r = Rig()
    r.sig(Sig.MUSIC_ON)
    r.wait(5)
    assert r.c.state is State.HOLD and [v.kind for v in r.verdicts][-1] is VK.HOLD
    assert not r.out


def test_music_with_recent_words_is_not_hold():
    r = Rig()
    r.sig(Sig.MUSIC_ON)
    r.say("Welcome to Acme Corp")
    r.wait(5)
    assert r.c.state is not State.HOLD


def test_hold_then_ivr_then_voicemail():
    r = Rig()
    r.say("Please hold.")
    r.say(MENU)
    r.wait(2)
    r.say("Please leave a message after the tone.")
    assert r.c.state is State.VOICEMAIL


# ---------------- screening ----------------
def test_screening_speaks_name_after_beep_then_human():
    r = Rig()
    r.say("Please say your name after the tone.")
    assert r.c.state is State.SCREENING
    a = r.sig(Sig.BEEP_END, freq=1000, dur=0.4)
    assert isinstance(a[0], Speak) and "Anne" in a[0].text
    r.say("Hold on while I try to connect you.")
    assert r.c.state is State.HOLD
    r.say("Hello?")
    assert handed_off(r.wait(1.5))


def test_screening_without_beep_speaks_on_silence_and_rejection_ends():
    r = Rig()
    r.say("Record your name and I'll see if this person is available.")
    assert kinds(r.wait(3)) == ["Speak"]
    assert r.say("The person you are calling is unable to take your call.") == [Hangup(Outcome.SCREENED_OUT)]


# ---------------- human confirmation ----------------
def test_amd_human_alone_does_not_hand_off():
    r = Rig()
    r.amd(AnsweredBy.HUMAN)
    assert not r.out and r.c.state is State.LISTENING


def test_human_text_plus_short_burst_rhythm_confirms_fast():
    r = Rig()
    r.say("Hello?")
    assert r.out == []
    assert handed_off(r.sig(Sig.RHYTHM, 0.65, verdict="human"))


def test_human_text_alone_confirms_after_fallback_window():
    r = Rig()
    r.say("Hello?")
    r.wait(1.0)
    assert not handed_off(r.out) and r.c.state is not State.HUMAN
    assert handed_off(r.wait(0.5))


def test_rhythm_alone_never_confirms_a_human():
    r = Rig()
    r.amd(AnsweredBy.HUMAN)
    r.sig(Sig.RHYTHM, 0.75, verdict="human")
    r.wait(2)
    assert not handed_off(r.out) and r.c.state is not State.HUMAN


def test_long_monologue_blocks_a_human_sounding_greeting():
    r = Rig()
    r.sig(Sig.RHYTHM, 0.9, verdict="machine")
    r.say("Hi this is Dana, I can't get to my phone")
    r.wait(4)
    assert not handed_off(r.out) and r.c.state is not State.HUMAN
    assert r.f.phase != "human"


def test_amd_human_silent_gets_hello_probe_then_reply_confirms():
    r = Rig()
    r.amd(AnsweredBy.HUMAN)
    assert r.wait(3) == [Speak("Hello?")]
    assert handed_off(r.say("Yeah who's this"))


def test_ambiguous_greeting_probes_reply_confirms_but_menu_reply_does_not():
    r = Rig()
    assert kinds(r.say("Thank you for calling Acme Corp, how can I help you?")) == ["Speak"]
    assert handed_off(r.say("Hi, I'm looking for the lab manager"))

    r = Rig()
    r.say("Thank you for calling Acme Corp, how can I help you?")
    r.say("For sales press 1. For service press 2.")
    assert not handed_off(r.out) and r.f.phase == "ivr"


def test_personal_voicemail_greeting_start_is_not_a_human():
    r = Rig()
    r.amd(AnsweredBy.MACHINE_START)
    r.say("Hi this is John Smith I'm not able to get to the phone")
    r.wait(4)
    assert not handed_off(r.out) and r.c.state is not State.HUMAN and r.c.state is not State.HUMAN


def test_dead_air_machine_fails_but_silent_human_hands_off():
    r = Rig()
    r.amd(AnsweredBy.MACHINE_START)
    assert r.wait(25) == [Hangup(Outcome.FAILED)]
    r = Rig()
    r.amd(AnsweredBy.HUMAN)
    acts = r.wait(25)
    assert acts[0] == Speak("Hello?") and handed_off(acts)


def test_evidence_is_recorded_on_the_verdict():
    r = Rig()
    r.amd(AnsweredBy.HUMAN)
    r.say("Hello?")
    h = [v for v in r.verdicts if v.kind is VK.HUMAN][0]
    assert "text:human_phrase" in h.evidence and "amd:human" in h.evidence


def test_machine_rhythm_outweighs_amd_human_plus_human_sounding_text():
    r = Rig()
    r.amd(AnsweredBy.HUMAN)                          # AMD often mislabels long greetings as human
    r.sig(Sig.RHYTHM, 0.9, verdict="machine")
    r.say("Hi this is Dana")
    r.wait(4)
    assert not handed_off(r.out) and r.c.state is not State.HUMAN


def test_music_playing_outweighs_amd_human_plus_text():
    r = Rig()
    r.amd(AnsweredBy.HUMAN)
    r.sig(Sig.MUSIC_ON)
    r.say("Hello")
    r.wait(1.0)
    assert not handed_off(r.out) and r.c.state is not State.HUMAN


def test_amd_machine_outweighs_short_burst_rhythm_and_human_text():
    r = Rig()
    r.amd(AnsweredBy.MACHINE_START)
    r.sig(Sig.RHYTHM, 0.65, verdict="human")
    r.say("Hi this is John Smith I'm not able to get to the phone")
    r.wait(4)
    assert not handed_off(r.out) and r.c.state is not State.HUMAN


def test_person_like_text_inside_a_voicemail_greeting_is_not_a_human():
    r = Rig()
    r.say("Please leave a message after the tone")
    assert r.c.state is State.VOICEMAIL
    r.say("Hello this is John speaking")
    r.sig(Sig.RHYTHM, 0.65, verdict="human")
    r.wait(3)
    assert not handed_off(r.out) and r.c.state is not State.HUMAN


def test_never_confirm_a_human_while_they_are_still_talking():
    r = Rig()
    r.sig(Sig.SPEECH_START)
    r.say("Hi, this is John Smith")
    r.sig(Sig.RHYTHM, 0.65, verdict="human")
    r.wait(3)
    assert not handed_off(r.out)               # speech never ended: a greeting rolling on, not a person waiting
    r.sig(Sig.SPEECH_END, dur=3.0)
    assert handed_off(r.wait(1.5))             # ...once they go quiet and wait for us, it's a person


def test_llm_pick_chooses_human_route_rules_would_miss():
    from telephony_agent.controller import CallController
    from telephony_agent.detection.signals import VK, Verdict
    from telephony_agent.ivr import parse_options
    from telephony_agent.config import Settings
    from telephony_agent.models import CallRequest
    text = "For store hours press 1. To talk to a colleague press 2."
    opts = parse_options(text)
    c = CallController(CallRequest(to="+1", call_id="x"), Settings(callback_number="1"))
    v = Verdict(VK.IVR_MENU_READY, 5.0, 0.9, ["menu_settled"], {"text": text, "options": opts, "repeat": False, "llm_pick": ("human", "2")})
    assert c.handle(v) == [Dtmf("2")]
    c2 = CallController(CallRequest(to="+1", call_id="y"), Settings(callback_number="1"))
    v2 = Verdict(VK.IVR_MENU_READY, 5.0, 0.9, ["menu_settled"], {"text": text, "options": opts, "repeat": False, "llm_pick": ("human", "9")})
    assert c2.handle(v2) == [Dtmf("0")]      # hallucinated key rejected -> operator
