"""Live-call regression (call 7ab906e55fd5): after pressing a key in an IVR a person said "Hello?" four times and was
never confirmed as human, so Anne never spoke. Two stale pieces of evidence from the MENU dragged the human score down:
the menu's long-monologue rhythm ("machine") and Twilio's late AMD machine_end_other. Both describe the menu, not whoever
answers after the key press."""
from rig import Rig, kinds
from telephony_agent.detection.signals import Sig
from telephony_agent.models import AnsweredBy, Dtmf, HandoffToBrain, State

MENU = ("Thank you for calling Tensitron. For sales, press one. For parts and service, press two. For accounts payable, "
        "press three. For the quality department, press four. For the engineering department, press five. "
        "For all other inquiries, please press zero.")


def play_menu_and_press(r):
    r.sig(Sig.SPEECH_START)
    r.t += 7.0
    r.sig(Sig.SPEECH_END, dur=7.0)
    r.sig(Sig.RHYTHM, verdict="machine")                    # a 7 s recorded menu is a long monologue
    r.say(MENU)
    acts = r.wait(2)
    assert any(isinstance(a, Dtmf) and a.digits == "5" for a in acts), acts      # engineering first, as before
    return acts


def test_person_answering_after_the_key_press_is_confirmed_despite_menu_evidence():
    r = Rig()
    play_menu_and_press(r)
    r.t += 12
    r.amd(AnsweredBy.MACHINE_END_OTHER)                     # Twilio's verdict on the ORIGINAL answer (the IVR), arrives late
    r.talk(0.3)                                             # the person: one short burst
    r.sig(Sig.RHYTHM, verdict="human", dur=0.3)
    r.say("Hello?")
    acts = r.wait(2)
    assert HandoffToBrain.__name__ in kinds(acts) and r.c.state is State.HUMAN


def test_a_later_hello_also_confirms_if_the_first_was_missed():
    r = Rig()
    play_menu_and_press(r)
    r.amd(AnsweredBy.MACHINE_END_OTHER)
    for _ in range(3):
        r.wait(10)
        r.talk(0.3)
        r.sig(Sig.RHYTHM, verdict="human", dur=0.3)
        r.say("Hello?")
        if r.c.state is State.HUMAN:
            break
    assert r.c.state is State.HUMAN


def test_a_recorded_voicemail_greeting_after_a_key_press_is_still_not_a_human():
    """The reset must not turn a mailbox greeting that starts like a person into a human."""
    r = Rig()
    play_menu_and_press(r)
    r.t += 3
    r.sig(Sig.SPEECH_START)
    r.t += 9.0
    r.sig(Sig.SPEECH_END, dur=9.0)
    r.sig(Sig.RHYTHM, verdict="machine")                    # a long greeting
    r.say("Hi, this is Mike in engineering. I can't take your call right now, please leave a message after the tone.")
    r.wait(3)
    assert r.c.state is not State.HUMAN
