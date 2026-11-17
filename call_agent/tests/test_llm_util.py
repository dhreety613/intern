"""One stalled Gemini request must not freeze a turn: retry once, then raise to the caller's own fallback."""
import pytest

from call_agent.app.agents import llm_util


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0
        self.models = self

    def generate_content(self, **kwargs):
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_first_success_is_not_retried():
    c = FakeClient(["ok"])
    assert llm_util.generate(c, model="m", contents="x") == "ok" and c.calls == 1


def test_a_stalled_or_failed_first_attempt_is_retried_once(monkeypatch):
    monkeypatch.setattr(llm_util, "LLM_RETRIES", 1)
    c = FakeClient([TimeoutError("deadline exceeded"), "ok"])
    assert llm_util.generate(c, model="m", contents="x") == "ok" and c.calls == 2


def test_persistent_failure_is_raised_after_the_retries(monkeypatch):
    monkeypatch.setattr(llm_util, "LLM_RETRIES", 1)
    c = FakeClient([RuntimeError("a"), RuntimeError("b")])
    with pytest.raises(RuntimeError):
        llm_util.generate(c, model="m", contents="x")
    assert c.calls == 2


def test_client_carries_the_request_deadline():
    c = llm_util.make_client("fake-key")
    assert c is not None
