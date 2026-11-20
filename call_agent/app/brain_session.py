"""CallSession-compatible adapter around the new conversation brain (call_agent/calibration_brain).

TwilioMediaStream only needs three things from a session: start() -> opening line, process(text) -> dict, and a
`state` dict for the end-of-call result. This adapter provides exactly that, so audio / VAD / barge-in / watchdog /
rep-transfer code is unchanged. Select with env ANNE_BRAIN=new (default) or ANNE_BRAIN=old (instant rollback).

The brain is async; process() is called from a worker thread (asyncio.to_thread), so the brain lives on its own
long-lived event loop thread (its Gemini client must stay on one loop).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

from call_agent.calibration_brain import Action, Brain, CallInput, FormStep, GeminiLLM, Meter, Say, load_config
from call_agent.calibration_brain.voice import speakable

logger = logging.getLogger(__name__)

_STEPS = Path(__file__).resolve().parent.parent / "calibration_brain" / "data" / "form_steps.json"
TURN_TIMEOUT_S = float(os.getenv("ANNE_BRAIN_TURN_TIMEOUT_SECONDS", "30"))


def build_meters(products: List[str], dates: List[str]) -> List[Meter]:
    """Pair CRM products with due dates by position; tolerate length mismatches."""
    n = max(len(products), len(dates))
    meters = []
    for i in range(n):
        name = products[i] if i < len(products) else (products[0] if products else "your tension meter")
        due = dates[i] if i < len(dates) else (dates[-1] if dates else "")
        meters.append(Meter(name, due))
    return meters


class BrainSession:
    def __init__(self, customer_name: str, email: str, phone_number: str, calibration_dates: List[str],
                 products: List[str], heard: str = "", llm: Any = None, config: Any = None):
        self.heard = (heard or "").strip()
        self.cfg = config or load_config()
        if os.getenv("ANNE_BRAIN_MODEL"):
            self.cfg = replace(self.cfg, gemini_model=os.environ["ANNE_BRAIN_MODEL"])
        steps = [FormStep(**s) for s in json.loads(_STEPS.read_text(encoding="utf8"))]
        self.call = CallInput(customer_name or "the contact", phone_number or "", email or "",
                              build_meters(products, calibration_dates), steps)
        self._llm = llm
        self.brain: Brain | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="anne-brain-loop", daemon=True)
        self._thread.start()
        self._ended = False
        self._rep_reason = ""

    def _run(self, coro, timeout: float = TURN_TIMEOUT_S):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)

    async def _collect(self, gen) -> Dict[str, Any]:
        says: List[str] = []
        end_call, transfer = False, False
        async for ev in gen:
            if isinstance(ev, Say):
                says.append(ev.text)
            elif isinstance(ev, Action):
                if ev.type == "end_call":
                    end_call = True
                elif ev.type == "transfer":
                    transfer = True
                    self._rep_reason = (ev.data or {}).get("reason", "") or "caller asked for a representative"
        return {
            "response": speakable(" ".join(says)),
            "call_ended": end_call,
            "decision": {"present_task": "CONTINUE", "redirect_to_rep": transfer,
                         "redirect_reason": self._rep_reason if transfer else ""},
        }

    async def _ensure_brain(self):
        if self.brain is None:
            llm = self._llm or GeminiLLM(self.cfg.gemini_model, self.cfg.temperature)
            self.brain = Brain(self.call, llm, self.cfg)

    async def _start(self) -> Dict[str, Any]:
        await self._ensure_brain()
        return await self._collect(self.brain.start(self.heard or None))

    async def _turn(self, text: str) -> Dict[str, Any]:
        await self._ensure_brain()
        return await self._collect(self.brain.handle_user_turn(text))

    # ---- CallSession interface -------------------------------------------------------------------------------
    def start(self) -> str:
        try:
            return self._run(self._start())["response"] or "Hi, this is Anne with Tensitron."
        except Exception:  # noqa: BLE001
            logger.exception("brain start failed")
            return f"Hi, this is Anne with Tensitron. May I speak with {self.call.registered_name}?"

    def process(self, text: str) -> Dict[str, Any]:
        if self._ended:
            return {"response": "", "call_ended": True, "decision": {"present_task": "CONTINUE"}}
        try:
            out = self._run(self._turn(text))
        except Exception:  # noqa: BLE001
            logger.exception("brain turn failed")
            out = {"response": "Sorry, could you say that one more time?", "call_ended": False,
                   "decision": {"present_task": "CONTINUE"}}
        if out.get("call_ended") or out["decision"].get("redirect_to_rep"):
            self._ended = True
        return out

    @property
    def state(self) -> Dict[str, Any]:
        """Dict shaped for call_result.build_call_result and the media stream's tracking."""
        base = {"customer_name": self.call.registered_name, "phone_number": self.call.registered_phone,
                "human_confirmed": True}
        if self.brain is None:
            return {**base, "memory": {}}
        r = self.brain.export()
        c = r["contact"]
        outcome = r.get("outcome")
        new_contact = bool(c.get("is_new_contact"))
        memory = {
            "wrong_number_so_avoid": outcome == "wrong_number",
            "do_not_call": outcome == "do_not_call",
            "new_name": c.get("name") if new_contact else None,
            "new_email": c.get("email") if (new_contact or c.get("email_changed")) else None,
            "new_phone_number": c.get("phone") if (new_contact or c.get("phone_changed")) else None,
            "wants_human_call": bool(r.get("reroute_requested")),
            "important_facts": [x for x in [r.get("summary")] if x],
            "next_call_date": None,
            "call_again_when": "later" if outcome == "busy_call_back_later" else None,
        }
        return {
            **base,
            "brain_outcome": outcome,
            "current_stage": self.brain.state.phase(),
            "guidance_phase": "done" if r.get("form_finished") else ("walking" if r.get("form_help_wanted") else "none"),
            "memory": memory,
        }
