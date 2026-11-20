"""The brain: one object per call. Feed it what the person said, it streams back sentences to speak + actions to execute."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Union

from .config import Config, load_config
from .guard import FactBook, SentenceSplitter, clean_for_speech
from .llm import LLM, Msg, TextDelta, ToolCall
from .prompts import build_system
from .state import CallInput, CallState
from .tools import TOOL_SPECS, ToolResult, apply_tool

log = logging.getLogger("anne.brain")

CALL_CONNECTED = "[system: the call just connected; someone has picked up but has not spoken yet]"
GAP_FILL = "Sorry, I missed that. Could you say that one more time?"


@dataclass
class Say:
    text: str


@dataclass
class Action:
    type: str  # "end_call" | "transfer"
    data: dict = field(default_factory=dict)


BrainEvent = Union[Say, Action]


class Brain:
    MAX_ROUNDS = 3  # 1 normal + up to 2 follow-ups (tool-only reply, or rejected tool call)

    def __init__(self, call: CallInput, llm: LLM, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.llm = llm
        self.state = CallState(call=call)
        self.facts = FactBook(call, self.cfg.support_phone, self.cfg.support_email)
        self.history: list[Msg] = []
        self.guard_log: list[dict] = []
        self.tool_log: list[dict] = []

    # ------------------------------------------------------------------ public API
    async def _safe_stream(self, system, tools=None):
        """LLM errors must never kill the call: log and yield nothing (the turn then falls back to GAP_FILL)."""
        try:
            async for ev in self.llm.stream(system, self.history, TOOL_SPECS if tools is None else tools):
                yield ev
        except Exception:  # noqa: BLE001
            log.exception("LLM stream failed")

    async def start(self, heard: str | None = None) -> AsyncIterator[BrainEvent]:
        """Call when a human is confirmed. `heard` = first thing they said ('Hello?'), if anything."""
        async for ev in self._turn(heard.strip() if heard and heard.strip() else CALL_CONNECTED, internal=not heard):
            yield ev

    async def handle_user_turn(self, text: str) -> AsyncIterator[BrainEvent]:
        self.facts.allow_caller(text)
        async for ev in self._turn(text.strip()):
            yield ev

    async def reenter(self, kind: str, heard: str | None = None) -> AsyncIterator[BrainEvent]:
        """Telephony agent calls this after hold-resume / transfer / new pickup. Wipes per-person progress, greets fresh."""
        notes = {
            "transferred": "The call was just transferred; a different person may be on the line now.",
            "hold_resumed": "The call was on hold and someone has just come back on the line.",
            "new_pickup": "Someone new has picked up.",
        }
        self.state.reset_for_new_person(notes.get(kind, "The line changed hands."))
        if heard:
            self.facts.allow_caller(heard)
        async for ev in self._turn(heard.strip() if heard else CALL_CONNECTED, internal=not heard):
            yield ev

    def truncate_last_assistant(self, actually_spoken: str):
        """Barge-in: the caller cut Anne off. Record only what was really heard."""
        for m in reversed(self.history):
            if m.role == "model":
                m.text = (actually_spoken.strip() + " [interrupted]").strip()
                return

    def export(self) -> dict:
        """For the post-call tags/CRM job."""
        return {
            **self.state.result(),
            "transcript": [{"role": "anne" if m.role == "model" else "caller", "text": m.text}
                           for m in self.history if not m.text.startswith("[system")],
            "guardrail_log": self.guard_log,
            "tool_log": self.tool_log,
        }

    # ------------------------------------------------------------------ internals
    def _refusal(self) -> str:
        c = self.cfg
        contact = " or ".join(x for x in [c.support_phone, c.support_email] if x)
        if self.state.guardrail_hits <= 1:
            reach = f" You can reach our team at {contact}, or" if contact else ""
            tail = f"{reach} I can connect you to a representative right now if you'd like." if reach else " I can connect you to a representative right now if you'd like."
            return "I'm sorry, I don't have that information. I can only help with your calibration reminder and contact details." + tail
        return "That's outside what I can help with here. Would you like me to connect you to a representative?"

    async def _turn(self, user_text: str, internal: bool = False) -> AsyncIterator[BrainEvent]:
        st = self.state
        self.history.append(Msg("user", user_text))
        spoken_this_turn: list[str] = []
        deferred: list[Action] = []
        blocked = False

        speak_only = False
        for rnd in range(self.MAX_ROUNDS):
            for v in (st.email,):
                if v:
                    self.facts.allow_email(v)    # whatever is saved in state may be read back
            if st.phone:
                self.facts.allow_phone(st.phone)
            system = build_system(self.cfg, st)
            splitter, calls = SentenceSplitter(), []
            soft_block = None
            round_spoken: list[str] = []

            async def sentences(stream):
                nonlocal blocked
                async for ev in stream:
                    if isinstance(ev, ToolCall):
                        calls.append(ev)
                    elif isinstance(ev, TextDelta) and not blocked:
                        for s in splitter.feed(ev.text):
                            yield s
                tail = splitter.flush()
                if tail and not blocked:
                    yield tail

            async for raw in sentences(self._safe_stream(system, [] if speak_only else None)):
                s = clean_for_speech(raw)
                if not s or blocked:
                    continue  # keep draining the stream so late tool calls (save_contact etc.) aren't lost
                g = self.facts.check(s)
                if not g.ok and rnd < self.MAX_ROUNDS - 1 and g.reason.startswith(("unknown number", "unknown figure", "unknown email")):
                    # the model invented a figure (e.g. an example phone number): don't speak it and don't refuse,
                    # just have it re-say that bit properly in the next round
                    soft_block = g.reason
                    self.guard_log.append({"blocked": s, "reason": g.reason, "action": "regenerate"})
                    continue
                if not g.ok:
                    blocked = True
                    st.guardrail_hits += 1
                    st.guardrail_topics.append(f"auto:{g.reason}"[:60])
                    self.guard_log.append({"blocked": s, "reason": g.reason})
                    log.warning("guard blocked %r (%s)", s, g.reason)
                    refusal = self._refusal()
                    round_spoken.append(refusal)
                    yield Say(refusal)
                    continue
                round_spoken.append(s)
                yield Say(s)

            spoken_this_turn += round_spoken
            if round_spoken:
                self.history.append(Msg("model", " ".join(round_spoken)))

            rejected: list[ToolResult] = []
            for c in calls:
                res = apply_tool(st, c.name, c.args)
                self.tool_log.append({"tool": c.name, "args": c.args, "ok": res.ok, "msg": res.message})
                if not res.ok:
                    rejected.append(res)
                elif res.action:
                    if blocked and res.action["type"] == "end_call":
                        # never hang up straight after an auto-refusal
                        st.ended, st.outcome = False, None
                        continue
                    data = dict(res.action)
                    if data["type"] == "transfer":
                        data["number"] = self.cfg.rep_transfer_number
                    deferred.append(Action(res.action["type"], data))
                if c.name == "log_guardrail_hit" and blocked:
                    st.guardrail_hits -= 1  # already counted by the guard; avoid double count

            if st.reentry_note and st.speaker != "unknown":
                st.reentry_note = None
            if soft_block and not blocked and rnd < self.MAX_ROUNDS - 1:
                self.history.append(Msg("user", f"[system: your last sentence contained a figure that is not in the call data or what the caller said ({soft_block}). Do not invent examples or numbers. Say that part again using only real details, naturally, without repeating anything you already said; never mention this note.]"))
                speak_only = True
                continue
            if rnd < self.MAX_ROUNDS - 1 and not blocked and not rejected and not spoken_this_turn and not deferred and calls:
                # tool-only reply (model recorded something but said nothing): ask for the spoken part, never go silent
                self.history.append(Msg("user", "[system: your update was saved. Now say your spoken reply to the person's last words, naturally and briefly; never mention this note.]"))
                speak_only = True  # no tools this round: the model can only talk
                continue
            if not rejected or rnd == self.MAX_ROUNDS - 1 or blocked:
                break
            note = " ".join(r.message for r in rejected)
            self.history.append(Msg("user", f"[system: your last action was rejected: {note} Continue the same reply naturally; do not repeat what you already said and never mention this note.]"))

        st.turns += 1
        if not spoken_this_turn and not deferred:
            self.history.append(Msg("model", GAP_FILL))
            yield Say(GAP_FILL)
        if any(a.type == "end_call" for a in deferred) and not spoken_this_turn:
            bye = "Thanks so much for your time. Take care!"   # never hang up without a goodbye
            self.history.append(Msg("model", bye))
            yield Say(bye)
        for a in deferred:
            yield a
