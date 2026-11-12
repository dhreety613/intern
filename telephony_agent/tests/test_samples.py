"""Regenerates the 10 labelled sample calls and replays each through detectors+fusion+controller."""
import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from telephony_agent.replay import check, load_wav, replay  # noqa: E402

pytestmark = pytest.mark.skipif(importlib.util.find_spec("espeakng_loader") is None, reason="needs espeak for synthetic speech")


@pytest.fixture(scope="module")
def samples(tmp_path_factory):
    import make_samples
    out = tmp_path_factory.mktemp("samples")
    make_samples.build(out)
    return out, json.loads((out / "manifest.json").read_text())["samples"]


def test_all_ten_samples_detect_what_they_should(samples):
    out, items = samples
    assert len(items) == 10
    failures = {}
    for it in items:
        res = replay(load_wav(out / it["file"]), it["events"])
        probs = check(res, it["expect"])
        if probs:
            failures[it["file"]] = probs
    assert not failures, failures


def test_the_hard_case_personal_greeting_is_never_a_human(samples):
    out, items = samples
    it = next(i for i in items if i["file"].startswith("03"))
    res = replay(load_wav(out / it["file"]), it["events"])
    assert "human" not in [v.kind.value for v in res.verdicts]
    # and it does NOT start talking until the beep has ended
    speak_t = [t for t, a in res.actions if type(a).__name__ == "Speak"][0]
    assert speak_t > 9.0


def test_message_waits_for_the_beep_not_the_greeting_text(samples):
    out, items = samples
    it = next(i for i in items if i["file"].startswith("01"))
    res = replay(load_wav(out / it["file"]), it["events"])
    beep = [v for v in res.verdicts if v.kind.value == "vm_beep_ready"][0]
    assert beep.data["source"] == "audio"
