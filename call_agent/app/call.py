import re
from typing import Any, Dict, List

from call_agent.app.agents.brain_agent import get_next_stage
from call_agent.app.agents.calibration_guide import (
    OFFER_QUESTION,
    TIMEOUT_SENTINEL,
    handle_guidance_turn,
    start_walkthrough,
)
from call_agent.app.graph.graph import call_graph
from call_agent.app.greeting import opening_message


class CallSession:

    def __init__(
        self,
        customer_name: str,
        email: str,
        phone_number: str,
        calibration_dates: List[str],
        products: List[str] = None,
    ):

        self.state: Dict[str, Any] = {

            "conversation": [],
            "context": [],

            "customer_name": customer_name,
            "email": email,
            "phone_number": phone_number,
            "calibration_dates": calibration_dates,
            "products": products or [],

            "memory": {},

            "progress": {
                "name_confirmed": False,
                "email_confirmed": False,
                "phone_confirmed": False,
                "calibration_completed": False,
            },

            "current_stage": "name_confirmation",

            # Calibration form guidance (offer + step-by-step
            # walkthrough) is deterministic, not Gemini-tracked - see
            # calibration_guide.py. "none" -> not reached yet / the
            # offer question is about to be asked by the normal
            # workflow; "offered" -> waiting on yes/no to the offer;
            # "walking" -> mid-walkthrough; "done" -> resume normal
            # workflow ("anything else?" + end call).
            "guidance_phase": "none",
            "guidance_step_idx": 0,

            "brain_output": {},
            "decision": {},

            "call_progress": {
                "topic_diversion_count": 0,
                "random_conversation_count": 0,
            },

            "call_ended": False,

            # True once the person has engaged (name confirmed / past stage 1).
            "human_confirmed": False,
        }

    # ========================================================
    # NEW PERSON JOINED
    # ========================================================

    def _restart_conversation(self):
        """Someone new is on the line: forget everything the previous person
        said and start the workflow again from the top. Only the customer
        record and a do-not-call request survive (that one is always honoured)."""
        keep_dnc = bool((self.state.get("memory") or {}).get("do_not_call"))
        self.state.update({
            "conversation": [],
            "context": [],
            "memory": {"do_not_call": True} if keep_dnc else {},
            "progress": {
                "name_confirmed": False,
                "email_confirmed": False,
                "phone_confirmed": False,
                "calibration_completed": False,
            },
            "current_stage": "name_confirmation",
            "guidance_phase": "none",
            "guidance_step_idx": 0,
            "reminder_delivered": False,
            "awaiting_new_person": False,
            "brain_output": {},
            "decision": {},
            "call_progress": {
                "topic_diversion_count": 0,
                "random_conversation_count": 0,
            },
            "call_ended": False,
            "human_confirmed": False,
        })

    def _new_person_turn(self) -> Dict[str, Any]:
        """Greet again (scripted opening) with the workflow restarted."""
        self._restart_conversation()
        greeting = opening_message(self.state["customer_name"])
        self.add_message("assistant", greeting)
        decision = {
            "response": greeting,
            "present_task": "CONTINUE",
            "needs_context": False,
            "topic_diversion": False,
            "random_conversation": False,
            "redirect_to_rep": False,
            "redirect_reason": "",
        }
        self.state["decision"] = decision
        return {
            "response": greeting,
            "decision": decision,
            "brain": {},
            "progress": self.state["progress"],
            "memory": self.state["memory"],
            "current_stage": self.state["current_stage"],
            "call_ended": False,
            "restarted": True,
        }

    # ========================================================
    # ADD MESSAGE
    # ========================================================

    def add_message(
        self,
        role: str,
        content: str,
    ):

        self.state["conversation"].append({
            "role": role,
            "content": content,
        })

    # ========================================================
    # START CALL
    # ========================================================

    def start(self) -> str:
        """Anne's scripted opening line. Only confirmed humans reach the
        brain, so there is nothing to screen here."""

        opening = opening_message(self.state["customer_name"])

        self.add_message(
            "assistant",
            opening
        )

        return opening

    # ========================================================
    # PROCESS CUSTOMER MESSAGE
    # ========================================================

    def process(
        self,
        user_message: str,
    ) -> Dict[str, Any]:

        self.state["user_message"] = user_message
        is_timeout = (user_message or "").strip() == TIMEOUT_SENTINEL

        # IMPORTANT:
        # Customer message enters conversation BEFORE
        # Brain / Memory / Decision run.
        self.add_message(
            "user",
            user_message
        )

        guidance_phase = self.state.get("guidance_phase", "none")

        # Once the calibration-guidance offer has been made, the offer's
        # answer and the walkthrough are run by calibration_guide.py: the
        # LLM words each step, code owns the step counter (no skipping, no
        # invented steps). The normal graph is only re-entered when this
        # sub-flow reports "done" - or when the customer asked for
        # something else (run_graph), e.g. "no thanks, and please use a
        # different number".
        if guidance_phase in ("offered", "walking"):
            guidance_result = handle_guidance_turn(
                guidance_phase,
                self.state.get("guidance_step_idx", 0),
                user_message,
                self.state["conversation"][:-1],
            )

            self.state["guidance_phase"] = guidance_result["phase"]
            self.state["guidance_step_idx"] = guidance_result["step_idx"]

            if not guidance_result.get("run_graph"):
                response = guidance_result["response"]
                if response:
                    self.add_message("assistant", response)

                return {
                    "response": response,
                    "decision": {
                        "response": response,
                        "present_task": guidance_result["present_task"],
                        "needs_context": False,
                        "topic_diversion": False,
                        "random_conversation": False,
                        "redirect_to_rep": False,
                        "redirect_reason": "",
                    },
                    "brain": self.state.get("brain_output", {}),
                    "progress": self.state.get("progress", {}),
                    "memory": self.state.get("memory", {}),
                    "current_stage": self.state.get("current_stage", ""),
                    "call_ended": guidance_result["call_ended"],
                }

            guidance_phase = guidance_result["phase"]      # "done": the side request goes through the normal graph

        result = call_graph.invoke(
            self.state
        )

        # Update persistent session state
        self.state.update(result)

        decision = result.get(
            "decision",
            {}
        )

        # Someone new is speaking (a person was fetched / the phone was
        # handed over): greet again and restart the whole workflow.
        if decision.get("new_person_joined") and not is_timeout:
            return self._new_person_turn()
        # The speaker said they will get someone: the next speaker is
        # probably that someone (the Decision prompt is told so).
        self.state["awaiting_new_person"] = bool(decision.get("will_fetch_person"))

        # A person has answered the name question (or the flow has moved
        # past it): the person has engaged with the conversation.
        if (
            self.state.get("progress", {}).get("name_confirmed")
            or self.state.get("current_stage", "name_confirmation")
            != "name_confirmation"
        ):
            self.state["human_confirmed"] = True

        # The Decision agent flagged a request for form guidance (like it
        # flags END_CALL): the orchestrator replaces Gemini's reply with the
        # start of the walkthrough (step 1, worded by the LLM from the step
        # data, validated).
        guidance_switched = False
        if decision.get("guidance_requested") and guidance_phase != "walking":
            walk = start_walkthrough(self.state["conversation"][:-1])
            # Keep anything the model acknowledged in the same reply ("Great, your email is updated.") and
            # replace only the walkthrough sentence itself.
            ack = " ".join(
                part.strip()
                for part in re.split(r"(?<=[.!?])\s+", str(decision.get("response", "")))
                if part.strip() and not any(k in part.lower() for k in ("walk you through", "guide you", "guidance"))
            )
            decision = {
                **decision,
                "response": (ack + " " + walk["response"]).strip(),
                "present_task": walk["present_task"],
                "redirect_to_rep": False,
                "redirect_reason": "",
            }
            result = {**result, "decision": decision, "call_ended": False}
            self.state["decision"] = decision
            self.state["call_ended"] = False
            self.state["guidance_phase"] = walk["phase"]
            self.state["guidance_step_idx"] = walk["step_idx"]
            guidance_switched = True

        # The calibration reminder was just spoken: the guidance offer
        # belongs in the SAME turn, not a separate one afterwards. The model
        # is asked to include it; this guarantees it either way. The
        # reminder counts as delivered now, and the customer's next reply is
        # the answer to the offer (handled by calibration_guide.py).
        if (
            not guidance_switched
            and guidance_phase == "none"
            and not self.state.get("reminder_delivered")
            and decision.get("present_task") != "END_CALL"
            and not decision.get("redirect_to_rep")
        ):
            spoken = str(decision.get("response", ""))
            low = spoken.lower()
            gave_reminder = bool(decision.get("reminder_given")) or (
                self.state.get("current_stage") == "calibration"
                and "calibration" in low
                and "due" in low
            )
            if gave_reminder:
                if "guidance" not in low:
                    spoken = f"{spoken.rstrip()} {OFFER_QUESTION}"
                decision = {**decision, "response": spoken, "present_task": "WAIT"}
                result = {**result, "decision": decision}
                self.state["decision"] = decision
                self.state["reminder_delivered"] = True
                self.state["guidance_phase"] = "offered"
                self.state["guidance_step_idx"] = 0
                progress = dict(self.state.get("progress") or {})
                progress["calibration_completed"] = True
                self.state["progress"] = progress
                self.state["current_stage"] = get_next_stage(progress)

        response = decision.get(
            "response",
            ""
        )

        # Add Anne's response to conversation
        if response:

            self.add_message(
                "assistant",
                response
            )

        return {

            "response":
                response,

            "decision":
                decision,

            "brain":
                result.get(
                    "brain_output",
                    {}
                ),

            "progress":
                self.state.get(
                    "progress",
                    {}
                ),

            "memory":
                result.get(
                    "memory",
                    self.state.get(
                        "memory",
                        {}
                    )
                ),

            "current_stage":
                self.state.get(
                    "current_stage",
                    ""
                ),

            "call_ended":
                result.get(
                    "call_ended",
                    False
                ),
        }
