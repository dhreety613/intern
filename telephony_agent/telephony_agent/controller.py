"""CallController: the action layer. Consumes fused Verdicts (detection/fusion.py) and decides what
to do: stay silent, speak, press keys, hang up, or hand off. It does no detection itself."""
from __future__ import annotations

from .config import Settings
from .ivr import IvrOption, choose_key, operator_available, validate_llm_pick, voicemail_option
from .models import (Action, CallRequest, Dtmf, HandoffToBrain, Hangup, Outcome, Speak, SpeechDone, State, Tick)
from .detection.signals import VK, Verdict

_SPEECH_IVR_HINT = ("say ", "tell me", "in a few words", "briefly")


class CallController:
    def __init__(self, req: CallRequest, settings: Settings | None = None, targets: list[str] | None = None):
        self.req = req
        self.s = settings or Settings()
        self.targets = targets
        self.state = State.DIALING
        self.outcome: Outcome | None = None
        self.log: list[dict] = []          # per-call action/verdict trail (stage 2 persists this)
        self.heard: list[str] = []
        self.state_t = 0.0
        self.last_action_t = 0.0
        self.hold_t = 0.0
        # ivr
        self.menus = 0
        self.retries = 0
        self.tried_zero = False
        self.tried_speech = False
        self.tried_vm = False
        self.last_opts: list[IvrOption] = []
        self.last_was_press = False
        # misc
        self.screen_spoken = False

    # ------------------------------------------------------------------ api
    def handle(self, ev) -> list[Action]:
        if self.state in (State.ENDED, State.HUMAN):
            return []
        if isinstance(ev, Verdict):
            return self._on_verdict(ev)
        if isinstance(ev, SpeechDone):
            if self.state is State.LEAVING_MESSAGE:
                return self._end(Outcome.VOICEMAIL_LEFT, ev.t)
            return []
        if isinstance(ev, Tick):
            return self._on_tick(ev.t)
        return []

    def finish_pre_answer(self, outcome: Outcome) -> None:
        """Dial never connected (Lookup invalid / Twilio rejected the call)."""
        self._note(0, "pre_answer_outcome", outcome=outcome.value)
        self.state, self.outcome = State.ENDED, outcome

    # ------------------------------------------------------------ internals
    def _note(self, t: float, kind: str, **kw) -> None:
        self.log.append({"t": round(t, 2), "kind": kind, "state": self.state.value, **kw})

    def _go(self, st: State, t: float) -> None:
        if st is not self.state:
            self._note(t, "state", to=st.value)
            self.state, self.state_t = st, t

    def _end(self, outcome: Outcome, t: float, hangup: bool = True) -> list[Action]:
        self._note(t, "end", outcome=outcome.value)
        self.state, self.outcome = State.ENDED, outcome
        return [Hangup(outcome)] if hangup else []

    def _human(self, t: float, heard: list[str]) -> list[Action]:
        self._note(t, "human_confirmed")
        self.heard = heard
        self.state, self.outcome = State.HUMAN, Outcome.HUMAN_CONFIRMED
        return [HandoffToBrain(transcript_so_far=list(heard))]

    def _speak(self, text: str, t: float) -> list[Action]:
        self._note(t, "speak", text=text)
        self.last_action_t = t
        return [Speak(text)]

    def _press(self, keys: str, t: float, why: str) -> list[Action]:
        self._note(t, "dtmf", keys=keys, why=why)
        self.last_action_t = t
        return [Dtmf(keys)]

    # ------------------------------------------------------------- verdicts
    def _on_verdict(self, v: Verdict) -> list[Action]:
        self._note(v.t, "verdict", verdict=v.kind.value, conf=round(v.conf, 2), evidence=v.evidence)
        k, t, d = v.kind, v.t, v.data
        if k is VK.ANSWERED:
            self._go(State.LISTENING, t)
            return []
        if k is VK.PRE_ANSWER:
            return self._end(d["outcome"], t, hangup=False)
        if k is VK.REMOTE_HANGUP:
            self.state, self.outcome = State.ENDED, self.outcome or Outcome.REMOTE_HANGUP
            return []
        if k is VK.FAX:
            return self._end(Outcome.INVALID_NUMBER, t)
        if k is VK.BUSY_TONE:
            return self._end(Outcome.BUSY if d.get("cadence") == "busy" else Outcome.UNREACHABLE, t)
        if k in (VK.CARRIER, VK.MAILBOX):
            return self._end(d["outcome"], t)
        if k is VK.HUMAN:
            return self._human(t, d.get("heard", []))
        if k is VK.PROBE_DUE:
            return self._speak("Hello?", t)
        if k is VK.AMBIGUOUS_GREETING:
            return self._speak(self.s.screening_text(), t)
        if k is VK.VM_GREETING:
            self._go(State.VOICEMAIL, t)
            return []
        if k is VK.VM_BEEP_READY:
            if self.state in (State.LISTENING, State.VOICEMAIL):
                self._go(State.LEAVING_MESSAGE, t)
                return self._speak(self.s.voicemail_text(), t)
            return []
        if k is VK.SCREEN_PROMPT:
            self._go(State.SCREENING, t)
            return []
        if k is VK.SCREEN_SPEAK_READY:
            if self.state is State.SCREENING and not self.screen_spoken:
                self.screen_spoken = True
                return self._speak(self.s.screening_text(), t)
            return []
        if k is VK.SCREEN_REJECT:
            return self._end(Outcome.SCREENED_OUT, t)
        if k is VK.HOLD:
            self._go(State.HOLD, t)
            self.hold_t = t
            return []
        if k is VK.IVR_MENU_READY:
            self._go(State.IVR, t)
            return self._decide_menu(v)
        if k is VK.IVR_INVALID:
            self._go(State.IVR, t)
            self.last_was_press = False
            return self._ivr_fallback(t, "invalid entry")
        if k is VK.IVR_EXTENSION:
            self._go(State.IVR, t)
            if self.req.extension:
                return self._press(self.req.extension + "#", t, "extension")
            return self._ivr_fallback(t, "extension prompt, no extension")
        if k is VK.DEAD_AIR:
            if d.get("amd_human"):
                return self._human(t, [])                  # answered, said nothing: still a person
            return self._end(Outcome.FAILED, t)
        return []

    # ------------------------------------------------------------------ IVR
    def _ivr_fallback(self, t: float, why: str, direct: bool = False) -> list[Action]:
        """No usable human route. direct=True (a fresh menu with no human option): go straight down the ladder.
        Otherwise (invalid entry / stuck / key didn't take): let the menu repeat first. Ladder: 0 -> voicemail -> end."""
        if not direct and self.retries < self.s.ivr_retries:
            self.retries += 1
            self._note(t, "ivr_retry", n=self.retries, why=why)
            self.last_action_t = t
            return []
        if not self.tried_zero:
            self.tried_zero = True
            return self._press("0", t, f"operator ({why})")
        vm = voicemail_option(self.last_opts)
        if vm and not self.tried_vm:
            self.tried_vm = True
            return self._press(vm.key, t, f"voicemail option ({why})")
        return self._end(Outcome.IVR_FAILED, t)

    def _decide_menu(self, v: Verdict) -> list[Action]:
        t, text, opts = v.t, v.data["text"], v.data["options"]
        self.menus += 1
        if self.menus > self.s.ivr_max_menus:
            return self._end(Outcome.IVR_FAILED, t)
        if v.data.get("repeat") and self.last_was_press:
            self.last_was_press = False                    # same menu again after a key: it didn't take
            return self._ivr_fallback(t, "menu repeated after key press")
        self.last_opts = opts
        # priority ladder: 1) option reaching a human (LLM pick, else rules)  2) press 0  3) voicemail option  4) end
        pick, how = v.data.get("llm_pick"), "rules"
        human = None
        if pick and pick[0] == "human":
            human, how = validate_llm_pick("human", pick[1], opts), "llm"
        if human is None:
            human = choose_key(opts, self.targets)           # rules: also catches a route the LLM missed
        self._note(t, "ivr_menu", text=text[-160:], stt_busy=v.data.get("stt_busy"), options=[(o.key, o.desc) for o in opts], pick=pick, via=how,
                   chose=human.key if human else None)
        if human:
            self.retries = 0
            self.last_was_press = True
            return self._press(human.key, t, f"menu->human ({how}): {human.desc}")
        self.last_was_press = False
        if not opts and any(h in text.lower() for h in _SPEECH_IVR_HINT) and not self.tried_speech:
            self.tried_speech = True
            return self._speak("Representative", t)
        return self._ivr_fallback(t, "no human option", direct=True)

    # ---------------------------------------------------------------- timers
    def _on_tick(self, t: float) -> list[Action]:
        st = self.state
        if st is State.HOLD and t - self.hold_t >= self.s.hold_max_s:
            return self._end(Outcome.HOLD_TIMEOUT, t)
        if st is State.IVR and t - self.last_action_t >= self.s.ivr_stuck_timeout_s:
            self.last_action_t = t
            return self._ivr_fallback(t, "stuck")
        if st is State.VOICEMAIL and t - self.state_t >= self.s.voicemail_max_wait_s:
            return self._end(Outcome.FAILED, t)
        if st is State.SCREENING and t - self.state_t >= 40:
            return self._end(Outcome.SCREENED_OUT, t)
        return []
