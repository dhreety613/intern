"""Calibration guidance (LLM-worded, code-tracked steps) and 'new person joined -> greet + restart'.
The LLM is stubbed: these tests pin the CODE's guarantees (no skipping, no invented steps, correct yes/no handling)."""
import pytest

from call_agent.app import call as call_module
from call_agent.app.agents import calibration_guide as cg
from call_agent.app.greeting import opening_message

N = len(cg.GUIDE_STEPS)


@pytest.fixture
def llm(monkeypatch):
    """Queue canned model answers; records the prompts it was asked."""
    box = {"answers": [], "prompts": []}

    def fake(prompt):
        box["prompts"].append(prompt)
        if not box["answers"]:
            raise AssertionError("unexpected extra LLM call")
        a = box["answers"].pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(cg, "llm_json", fake)
    return box


# ------------------------------------------------------------------ the offer
def test_the_live_call_failure_not_really_and_please_use_a_different_number(llm):
    """Real transcript: 'Not really. And ... could you PLEASE use a different number' was taken as a YES because of the
    word 'please', and the walkthrough started. It must be a NO plus a side request for the normal conversation."""
    llm["answers"].append({"answer": "no", "other_request": True})
    out = cg.handle_guidance_turn("offered", 0, "Not really. And, uh, could you please use a different number to contact me from now on?")
    assert out["phase"] == "done" and out["run_graph"] is True and out["response"] == ""


def test_plain_no_is_acknowledged_and_ends_the_offer(llm):
    llm["answers"].append({"answer": "no", "other_request": False})
    out = cg.handle_guidance_turn("offered", 0, "no thanks")
    assert out["phase"] == "done" and out["response"] == cg.DECLINE_ACK and not out["run_graph"]


def test_yes_starts_the_walkthrough_at_step_one(llm):
    llm["answers"] += [{"answer": "yes", "other_request": False},
                       {"response": "Happy to help. Please open a browser and go to tensitron dot com, and tell me when you're there."}]
    out = cg.handle_guidance_turn("offered", 0, "yes please")
    assert out["phase"] == "walking" and out["step_idx"] == 0 and "tensitron dot com" in out["response"]


def test_unclear_answer_re_asks_and_silence_never_counts_as_yes(llm):
    llm["answers"].append({"answer": "unclear", "other_request": False})
    assert cg.handle_guidance_turn("offered", 0, "hmm what")["response"] == cg.OFFER_QUESTION
    out = cg.handle_guidance_turn("offered", 0, cg.TIMEOUT_SENTINEL)          # no LLM call on silence
    assert out["phase"] == "offered" and out["response"] == cg.OFFER_QUESTION


def test_model_failure_on_the_offer_re_asks_instead_of_guessing(llm):
    llm["answers"].append(RuntimeError("gemini down"))
    out = cg.handle_guidance_turn("offered", 0, "yes")
    assert out["phase"] == "offered" and out["response"] == cg.OFFER_QUESTION


# --------------------------------------------------------------- the walkthrough
def test_steps_advance_exactly_one_at_a_time_through_to_the_closing(llm):
    """Customer says 'done' every turn; the model words each next step. Code must move 0->1->...->N-1 then close."""
    idx = 0
    for i in range(N):
        nxt = i + 1
        line = cg._ALL[nxt]["fallback"]                       # a valid wording for exactly the next step
        llm["answers"].append({"intent": "done", "other_request": False, "response": line})
        out = cg.handle_guidance_turn("walking", idx, "okay done")
        if nxt < N:
            assert out["phase"] == "walking" and out["step_idx"] == nxt and out["present_task"] == "WAIT", f"after step {i}"
            idx = out["step_idx"]
        else:
            assert out["phase"] == "done" and out["present_task"] == "CONTINUE" and "one-six-one-six" in out["response"]
    assert idx == N - 1


def test_a_model_reply_that_skips_ahead_is_rejected_and_still_advances_only_one_step(llm):
    """On step 1 the model jumps to step 3 ('click Calibration and Service'). Rejected: customer gets exactly step 2."""
    llm["answers"].append({"intent": "done", "other_request": False,
                           "response": "Great, now just click Calibration and Service in the menu."})
    out = cg.handle_guidance_turn("walking", 0, "I'm there")
    assert out["step_idx"] == 1 and out["response"] == cg.GUIDE_STEPS[1]["fallback"]


def test_invented_or_missing_verbatim_details_fall_back_to_the_plain_step(llm):
    llm["answers"].append({"intent": "repeat", "other_request": False, "response": "Go to tensitron.com in your browser."})  # URL not spoken form
    out = cg.handle_guidance_turn("walking", 0, "say that again")
    assert out["step_idx"] == 0 and out["response"] == cg.GUIDE_STEPS[0]["fallback"]
    llm["answers"].append({"intent": "done", "other_request": False, "response": "Everything is done. Thanks!"})   # closing without the contact details
    out = cg.handle_guidance_turn("walking", N - 1, "done")
    assert out["phase"] == "done" and "service at tensitron dot com" in out["response"]


@pytest.mark.parametrize("intent", ["repeat", "stuck", "question"])
def test_repeat_stuck_and_question_never_advance(llm, intent):
    llm["answers"].append({"intent": intent, "other_request": False,
                           "response": "Sure. From the homepage, look toward the upper part of the page for the menu bar."})
    out = cg.handle_guidance_turn("walking", 1, "where is that?")
    assert out["step_idx"] == 1 and out["phase"] == "walking" and "menu bar" in out["response"]


def test_decline_mid_walkthrough_stops_it_instead_of_carrying_on(llm):
    """Real transcript: 'Hey, no, I do not need a guidance' made the old code advance to the next step."""
    llm["answers"].append({"intent": "decline", "other_request": False, "response": ""})
    out = cg.handle_guidance_turn("walking", 3, "Hey, no, I do not need a guidance")
    assert out["phase"] == "done" and out["response"] == cg.DECLINE_ACK and not out["run_graph"]
    llm["answers"].append({"intent": "decline", "other_request": True, "response": ""})
    assert cg.handle_guidance_turn("walking", 3, "no, and call my other number")["run_graph"] is True


def test_side_request_leaves_the_walkthrough_for_the_normal_conversation(llm):
    llm["answers"].append({"intent": "side_request", "other_request": True, "response": ""})
    out = cg.handle_guidance_turn("walking", 2, "can I speak to someone?")
    assert out["phase"] == "done" and out["run_graph"] is True


def test_any_model_failure_or_junk_repeats_the_current_step_and_never_advances(llm):
    llm["answers"].append(RuntimeError("timeout"))
    out = cg.handle_guidance_turn("walking", 4, "okay")
    assert out["step_idx"] == 4 and out["response"] == cg.GUIDE_STEPS[4]["fallback"]
    llm["answers"].append({"intent": "banana", "response": "whatever"})
    assert cg.handle_guidance_turn("walking", 4, "okay")["step_idx"] == 4


def test_silence_repeats_the_current_step(llm):
    llm["answers"].append({"response": "Please look toward the upper part of the page for the menu bar."})
    out = cg.handle_guidance_turn("walking", 1, cg.TIMEOUT_SENTINEL)
    assert out["step_idx"] == 1 and out["phase"] == "walking"


def test_the_model_is_only_ever_shown_the_current_and_next_step(llm):
    llm["answers"].append({"intent": "repeat", "other_request": False, "response": cg.GUIDE_STEPS[2]["fallback"]})
    cg.handle_guidance_turn("walking", 2, "again")
    prompt = llm["prompts"][0]
    assert cg.GUIDE_STEPS[2]["instruction"] in prompt and cg.GUIDE_STEPS[3]["instruction"] in prompt
    for k in (0, 1, 4, 5, 6, 7, 8):
        assert cg.GUIDE_STEPS[k]["instruction"] not in prompt, f"step {k + 1} leaked into the prompt"


# ------------------------------------------------------------- call.py wiring
class FakeGraph:
    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), []

    def invoke(self, state):
        self.calls.append(dict(state))
        return self.outputs.pop(0)


def make_session(monkeypatch, *graph_outputs):
    g = FakeGraph(*graph_outputs)
    monkeypatch.setattr(call_module, "call_graph", g)
    s = call_module.CallSession(customer_name="Jim", email="jim@x.com", phone_number="+15550001",
                                calibration_dates=["September 15, 2026"], products=["365-HX"])
    s.start()
    return s, g


def prog(**kw):
    base = {"name_confirmed": True, "email_confirmed": True, "phone_confirmed": True, "calibration_completed": False}
    base.update(kw)
    return base


REMINDER = ("And I would just like to remind you that your calibration for the Tensitron meter with serial number "
            "365-HX is due on September 15, 2026.")


def test_the_guidance_offer_is_part_of_the_reminder_turn(monkeypatch):
    s, _ = make_session(monkeypatch, {"progress": prog(), "current_stage": "calibration",
                                      "decision": {"response": REMINDER, "present_task": "CONTINUE", "reminder_given": True}})
    out = s.process("yes that is my number")
    assert REMINDER in out["response"] and out["response"].endswith(cg.OFFER_QUESTION)
    assert out["decision"]["present_task"] == "WAIT"
    assert s.state["guidance_phase"] == "offered" and s.state["current_stage"] == "complete"
    assert s.state["progress"]["calibration_completed"] is True


def test_offer_is_added_even_if_the_model_forgets_the_flag_and_never_doubled(monkeypatch):
    s, _ = make_session(monkeypatch, {"progress": prog(), "current_stage": "calibration",
                                      "decision": {"response": REMINDER, "present_task": "CONTINUE"}})
    assert s.process("yes").get("response", "").endswith(cg.OFFER_QUESTION)
    s2, _ = make_session(monkeypatch, {"progress": prog(), "current_stage": "calibration",
                                       "decision": {"response": REMINDER + " " + cg.OFFER_QUESTION, "present_task": "WAIT", "reminder_given": True}})
    assert s2.process("yes")["response"].count("guidance") == 1


def test_full_live_call_scenario_no_thanks_plus_new_number_goes_to_the_normal_conversation(monkeypatch, llm):
    s, g = make_session(monkeypatch,
                        {"progress": prog(), "current_stage": "calibration",
                         "decision": {"response": REMINDER, "present_task": "CONTINUE", "reminder_given": True}},
                        {"progress": prog(calibration_completed=True), "current_stage": "complete",
                         "decision": {"response": "Of course, what number should I use from now on?", "present_task": "WAIT"},
                         "memory": {}})
    s.process("yes")                                                       # reminder + offer
    llm["answers"].append({"answer": "no", "other_request": True})
    out = s.process("Not really. And could you please use a different number to contact me from now on?")
    assert out["response"] == "Of course, what number should I use from now on?"      # the number request was NOT ignored
    assert s.state["guidance_phase"] == "done" and len(g.calls) == 2                    # normal graph handled it, guidance did not start
    assert g.calls[1]["guidance_phase"] == "done"


def test_guidance_requested_anytime_starts_step_one_via_the_llm(monkeypatch, llm):
    s, _ = make_session(monkeypatch, {"progress": prog(calibration_completed=True), "current_stage": "complete",
                                      "decision": {"response": "Sure.", "present_task": "CONTINUE", "guidance_requested": True}})
    llm["answers"].append({"response": "Sure. Open a browser and go to tensitron dot com, and tell me when you're there."})
    out = s.process("how do I fill the form?")
    assert "tensitron dot com" in out["response"] and s.state["guidance_phase"] == "walking" and s.state["guidance_step_idx"] == 0


# --------------------------------------------------------------- new person
def test_new_person_is_greeted_again_and_the_workflow_restarts(monkeypatch):
    s, g = make_session(
        monkeypatch,
        {"progress": prog(phone_confirmed=False), "current_stage": "phone_confirmation", "memory": {"new_email": "old@x.com", "do_not_call": False},
         "decision": {"response": "Sure, I'll hold.", "present_task": "WAIT", "will_fetch_person": True}},
        {"progress": {}, "current_stage": "name_confirmation", "memory": {},
         "decision": {"response": "ignored", "present_task": "CONTINUE", "new_person_joined": True}},
    )
    out1 = s.process("Hold on, let me get him")
    assert s.state["awaiting_new_person"] is True and out1["response"] == "Sure, I'll hold."
    out2 = s.process("Hello? Yes this is Jim")
    assert out2["restarted"] is True and out2["response"] == opening_message("Jim")
    assert g.calls[1]["awaiting_new_person"] is True                         # the decision prompt is told to expect a new person
    st = s.state
    assert st["progress"] == {"name_confirmed": False, "email_confirmed": False, "phone_confirmed": False, "calibration_completed": False}
    assert st["current_stage"] == "name_confirmation" and st["guidance_phase"] == "none" and st["human_confirmed"] is False
    assert st["memory"] == {} and st["awaiting_new_person"] is False and st["reminder_delivered"] is False
    assert st["conversation"] == [{"role": "assistant", "content": opening_message("Jim")}]    # previous person's words are gone
    assert st["customer_name"] == "Jim" and st["email"] == "jim@x.com"                          # the customer record survives


def test_restart_keeps_a_do_not_call_request(monkeypatch):
    s, _ = make_session(monkeypatch, {"progress": {}, "current_stage": "name_confirmation", "memory": {"do_not_call": True},
                                      "decision": {"response": "x", "present_task": "CONTINUE", "new_person_joined": True}})
    s.process("this is his colleague")
    assert s.state["memory"] == {"do_not_call": True}


def test_restart_also_resets_a_walkthrough_in_progress(monkeypatch, llm):
    s, g = make_session(monkeypatch, {"progress": prog(), "current_stage": "calibration",
                                      "decision": {"response": REMINDER, "present_task": "CONTINUE", "reminder_given": True}},
                        {"progress": {}, "current_stage": "name_confirmation", "memory": {},
                         "decision": {"response": "x", "present_task": "CONTINUE", "new_person_joined": True}})
    s.process("yes")
    s.state["guidance_phase"], s.state["guidance_step_idx"] = "done", 0                    # offer already dealt with
    out = s.process("Hi, this is his wife, he handed me the phone")
    assert out["restarted"] and s.state["guidance_phase"] == "none" and s.state["guidance_step_idx"] == 0


def test_silence_never_triggers_a_restart(monkeypatch):
    s, _ = make_session(monkeypatch, {"progress": {}, "current_stage": "name_confirmation", "memory": {},
                                      "decision": {"response": "Are you still there?", "present_task": "WAIT", "new_person_joined": True}})
    out = s.process(cg.TIMEOUT_SENTINEL)
    assert not out.get("restarted") and out["response"] == "Are you still there?"


def test_acknowledgement_is_kept_when_the_customer_confirms_something_and_asks_for_the_guide(monkeypatch, llm):
    """Live call: 'Yes that's right. And now can you guide me ...' - the walkthrough replaced the whole reply, so the
    confirmation was never acknowledged."""
    s, _ = make_session(monkeypatch, {"progress": prog(calibration_completed=True), "current_stage": "complete",
                                      "decision": {"response": "Great, your email is updated. Sure, I can walk you through it.",
                                                   "present_task": "CONTINUE", "guidance_requested": True}})
    llm["answers"].append({"response": "Open a browser and go to tensitron dot com, and tell me when you're there."})
    out = s.process("Yes, that's right. And now can you please guide me on how to fill the calibration form?")
    assert out["response"].startswith("Great, your email is updated.") and "tensitron dot com" in out["response"]
    assert "walk you through" not in out["response"]
    assert s.state["guidance_phase"] == "walking" and s.state["guidance_step_idx"] == 0
