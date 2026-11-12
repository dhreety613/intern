import numpy as np
import pytest
from telephony_agent.classifier import classify
from telephony_agent.ivr import choose_key, operator_available, parse_options
from telephony_agent.models import Label
from telephony_agent.tones import dtmf_mulaw, mulaw_decode, mulaw_encode


@pytest.mark.parametrize("text,label", [
    ("The number you have dialed is not in service.", Label.CARRIER_INVALID),
    ("The number you have dialed has been disconnected or is no longer in service.", Label.CARRIER_DISCONNECTED),
    ("We're sorry, your call cannot be completed as dialed.", Label.CARRIER_INVALID),
    ("The wireless customer you are calling is not available", Label.CARRIER_UNREACHABLE),
    ("The mailbox is full and cannot accept any messages at this time.", Label.MAILBOX_FULL),
    ("The voicemail box has not been set up.", Label.MAILBOX_DISABLED),
    ("This mailbox is not accepting messages.", Label.MAILBOX_DISABLED),
    ("Please record your name and I'll see if this person is available.", Label.SCREEN_PROMPT),
    ("At the tone, state your name and reason for calling.", Label.SCREEN_PROMPT),
    ("Hi you've reached John, I can't come to the phone, leave a message after the tone.", Label.VOICEMAIL_GREETING),
    ("Thank you for calling Acme. For sales, press 1. For support, press 2.", Label.IVR_MENU),
    ("To leave a message in the voicemail box press 3. For sales press 1.", Label.IVR_MENU),
    ("In a few words, tell me why you're calling.", Label.IVR_MENU),
    ("That is not a valid entry.", Label.IVR_INVALID_ENTRY),
    ("Please enter the extension of the person you are trying to reach", Label.IVR_EXTENSION_PROMPT),
    ("Please hold, your call is important to us.", Label.HOLD),
    ("All of our representatives are currently busy.", Label.HOLD),
    ("Hello?", Label.HUMAN),
    ("Hi, this is Mike speaking", Label.HUMAN),
    ("Thank you for calling Acme, this is Sarah, how can I help you", Label.HUMAN),
    ("Thank you for calling Acme Corp, how can I help you?", Label.AMBIGUOUS_GREETING),
    ("The person you are calling is unable to take your call.", Label.SCREEN_REJECT),
    ("la la la", Label.UNKNOWN),
])
def test_classify(text, label):
    assert classify(text) is label


def test_parse_options_styles():
    a = parse_options("Thank you for calling. For sales, press 1. For support, press 2. To speak to an operator press 0.")
    assert [o.key for o in a] == ["0", "1", "2"]
    b = parse_options("Press 1 for sales, 2 for support, 3 for billing")
    assert [(o.key, o.desc) for o in b] == [("1", "sales"), ("2", "support"), ("3", "billing")]
    c = parse_options("press one for english, press two for spanish")
    assert [o.key for o in c] == ["1", "2"]


def test_choose_key_prefers_calibration_then_person():
    o = parse_options("For sales, press 1. For calibration services, press 2. For billing press 3.")
    assert choose_key(o).key == "2"
    o = parse_options("For hours and location press 1. To speak with a representative press 4.")
    assert choose_key(o).key == "4"
    assert choose_key(parse_options("For hours press 1. For directions press 2.")) is None


def test_operator_available_anytime():
    assert operator_available("You can press zero at any time to reach an operator")


def test_mulaw_roundtrip():
    t = np.arange(1600) / 8000
    pcm = (8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    back = mulaw_decode(mulaw_encode(pcm))
    assert np.max(np.abs(back.astype(int) - pcm.astype(int))) < 400   # mu-law quantization


def test_dtmf_has_both_tones():
    pcm = mulaw_decode(dtmf_mulaw("5", tone_ms=100, gap_ms=0))
    spec = np.abs(np.fft.rfft(pcm.astype(float)))
    f = np.fft.rfftfreq(len(pcm), 1 / 8000)
    top = sorted(f[np.argsort(spec)[-2:]])
    assert abs(top[0] - 770) < 25 and abs(top[1] - 1336) < 25


def test_ivr_team_priority_engineering_then_parts_service_then_qa():
    from telephony_agent.ivr import choose_key, parse_options
    menu = "For sales, press 1. For quality assurance, press 2. For parts and service, press 3. For engineering, press 4."
    assert choose_key(parse_options(menu)).key == "4"
    assert choose_key(parse_options("For sales, press 1. For quality assurance, press 2. For parts and service, press 3.")).key == "3"
    assert choose_key(parse_options("For sales, press 1. For QA, press 2. For billing, press 5.")).key == "2"
    assert choose_key(parse_options("For sales, press 1. For billing, press 5.")).key == "1"


def test_customer_service_is_not_the_parts_service_team():
    from telephony_agent.ivr import choose_key, parse_options
    o = parse_options("For customer service and billing, press 1. For quality assurance, press 2.")
    assert choose_key(o).key == "2"


def test_tail_fragment_please_press_zero_is_option_zero():
    from telephony_agent.ivr import parse_options
    assert [(o.key, o.desc) for o in parse_options("For all other inquiries, please press zero")] == [("0", "all other inquiries")]


def test_dtmf_tone_level_matches_call_agent():
    pcm = mulaw_decode(dtmf_mulaw("5", tone_ms=100, gap_ms=0))
    assert 12000 < int(abs(pcm).max()) <= 16200      # two tones of 8000 each
