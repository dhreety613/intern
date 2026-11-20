"""The new-brain adapter speaks the interface TwilioMediaStream expects (start / process / state)."""
from call_agent.app.brain_session import BrainSession, build_meters
from call_agent.app.call_result import build_call_result
from call_agent.calibration_brain import ScriptedLLM, TextDelta, ToolCall


def T(x): return TextDelta(x)
def C(n, a): return ToolCall(n, a)


def mk(script, **kw):
    return BrainSession("Ryan Hart", "ryan@king.com", "346 487 6828", ["April 1, 2027"], ["ACX-250"],
                        llm=ScriptedLLM(script), **kw)


def test_start_and_process_shape():
    s = mk([[T("Hi, this is Anne with Tensitron. May I speak with Ryan Hart?")],
            [T("Of course, connecting you now."), C("reroute_to_rep", {"reason": "asked for person"})]])
    assert "Ryan Hart" in s.start()
    out = s.process("can I talk to a human")
    assert out["decision"]["redirect_to_rep"] is True and out["response"]
    assert s.process("anything")["call_ended"] is True          # nothing more after a transfer
    assert s.state["memory"]["wants_human_call"] is True


def test_speakable_applied_and_result_for_busy():
    s = mk([[T("Hi")],
            [T("No worries, reach us at service@tensitron.com. Take care."),
             C("end_call", {"outcome": "busy_call_back_later", "summary": "busy"})]])
    s.start()
    out = s.process("I'm busy, call later")
    assert "service at tensitron dot com" in out["response"] and out["call_ended"] is True
    res = build_call_result(s.state, "CA1", "p1", "d1")
    assert res["outcome"] == "busy_call_back_later" and res["call_successful"] is False and res["last_call_date"] is None


def test_meters_pairing_tolerates_mismatch():
    ms = build_meters(["A", "B"], ["2027-01-01"])
    assert [m.name for m in ms] == ["A", "B"]
    assert build_meters([], [])  == []
