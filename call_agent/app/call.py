from typing import Any, Dict, List

from call_agent.app.agents.calibration_guide import handle_guidance_turn
from call_agent.app.agents.ivr_agent import looks_like_ivr
from call_agent.app.agents.voicemail_agent import (
    VOICEMAIL_MESSAGE,
    looks_like_voicemail,
)
from call_agent.app.graph.graph import call_graph, start_graph


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

            # Voicemail phrase check runs only until this turns True.
            "human_confirmed": False,
        }

    def _restart_conversation(self):
        """Reset everything except customer data and IVR bookkeeping."""
        self.state.update({
            "conversation": [],
            "context": [],
            "memory": {},
            "progress": {
                "name_confirmed": False,
                "email_confirmed": False,
                "phone_confirmed": False,
                "calibration_completed": False,
            },
            "current_stage": "name_confirmation",
            "guidance_phase": "none",
            "guidance_step_idx": 0,
            "brain_output": {},
            "call_progress": {
                "topic_diversion_count": 0,
                "random_conversation_count": 0,
            },
            "call_ended": False,
            "human_confirmed": True,  # a human joined after the IVR
        })

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

    def start(self, answered_by: str = "") -> str:
        """Run the pre-greeting graph: voicemail -> fixed message + end
        call, otherwise Anne's normal opening."""

        result = start_graph.invoke({
            "customer_name": self.state["customer_name"],
            "answered_by": answered_by,
        })

        opening = result["decision"]["response"]

        if result.get("is_voicemail"):
            self.state["call_ended"] = True
            self.state["is_voicemail"] = True

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

        # Voicemail phrase check - only until a human is confirmed. IVR menus
        # say "to leave a message, press 2": that is a menu, not a greeting.
        if (
            not self.state.get("human_confirmed")
            and looks_like_voicemail(user_message)
            and not looks_like_ivr(user_message)
        ):
            self.add_message("user", user_message)
            self.add_message("assistant", VOICEMAIL_MESSAGE)
            self.state["call_ended"] = True
            self.state["is_voicemail"] = True
            return {
                "response": VOICEMAIL_MESSAGE,
                "decision": {
                    "response": VOICEMAIL_MESSAGE,
                    "present_task": "END_CALL",
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
                "call_ended": True,
            }

        # IMPORTANT:
        # Customer message enters conversation BEFORE
        # Brain / Memory / Decision run.
        self.add_message(
            "user",
            user_message
        )

        guidance_phase = self.state.get("guidance_phase", "none")

        # Once the calibration-guidance offer has actually been made,
        # everything through the end of the walkthrough is handled
        # deterministically (no Gemini) - see calibration_guide.py's
        # module docstring for why. The graph is only re-entered once
        # this sub-flow reports "done".
        if guidance_phase in ("offered", "walking"):
            guidance_result = handle_guidance_turn(
                guidance_phase,
                self.state.get("guidance_step_idx", 0),
                user_message,
            )

            self.state["guidance_phase"] = guidance_result["phase"]
            self.state["guidance_step_idx"] = guidance_result["step_idx"]

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

        result = call_graph.invoke(
            self.state
        )

        # Update persistent session state
        self.state.update(result)

        # A human joined after the IVR menu: forget the menu chatter and
        # restart the whole workflow from the beginning. The response
        # below is then the scripted opening.
        if result.get("ivr_result", {}).get("action") == "restart":
            self._restart_conversation()

        # A person has answered the name question (or the flow has moved
        # past it): stop looking for voicemail phrases.
        if (
            self.state.get("progress", {}).get("name_confirmed")
            or self.state.get("current_stage", "name_confirmation")
            != "name_confirmation"
        ):
            self.state["human_confirmed"] = True

        decision = result.get(
            "decision",
            {}
        )

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

        # The Gemini graph just asked the calibration-guidance offer
        # question this turn (CURRENT STAGE reached "complete" with
        # GUIDANCE PHASE still "none" going in) - from the next customer
        # reply onward, the deterministic branch above takes over.
        if (
            guidance_phase == "none"
            and self.state.get("progress", {}).get("calibration_completed")
        ):
            self.state["guidance_phase"] = "offered"

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
                result.get(
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
                result.get(
                    "current_stage",
                    self.state.get(
                        "current_stage",
                        ""
                    )
                ),

            "call_ended":
                result.get(
                    "call_ended",
                    False
                ),
        }
