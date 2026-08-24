from typing import Any, Dict, List

from call_agent.app.graph.graph import call_graph


class CallSession:

    def __init__(
        self,
        customer_name: str,
        email: str,
        phone_number: str,
        calibration_dates: List[str],
    ):

        self.state: Dict[str, Any] = {

            "conversation": [],
            "context": [],

            "customer_name": customer_name,
            "email": email,
            "phone_number": phone_number,
            "calibration_dates": calibration_dates,

            "memory": {},

            "progress": {
                "name_confirmed": False,
                "email_confirmed": False,
                "phone_confirmed": False,
                "calibration_completed": False,
            },

            "current_stage": "name_confirmation",

            "brain_output": {},
            "decision": {},

            "call_progress": {
                "topic_diversion_count": 0,
                "random_conversation_count": 0,
            },

            "call_ended": False,
        }

    def add_message(
        self,
        role: str,
        content: str,
    ):
        self.state["conversation"].append({
            "role": role,
            "content": content,
        })

    def start(self) -> str:
        """
        Start one complete call session.

        This is the same opening previously hardcoded
        inside graph_manual.py.
        """

        opening = (
            "Hi, this is Anne from KTC Instruments. "
            f"Am I speaking with {self.state['customer_name']}?"
        )

        self.add_message(
            "assistant",
            opening
        )

        return opening

    def process(
        self,
        user_message: str,
    ) -> Dict[str, Any]:

        self.state["user_message"] = user_message

        self.add_message(
            "user",
            user_message
        )

        result = call_graph.invoke(
            self.state
        )

        self.state.update(result)

        decision = result.get(
            "decision",
            {}
        )

        response = decision.get(
            "response",
            ""
        )

        if response:
            self.add_message(
                "assistant",
                response
            )

        return {
            "response": response,

            "decision": decision,

            "brain": result.get(
                "brain_output",
                {}
            ),

            "progress": result.get(
                "progress",
                {}
            ),

            "memory": result.get(
                "memory",
                self.state.get(
                    "memory",
                    {}
                )
            ),

            "current_stage": result.get(
                "current_stage",
                self.state.get(
                    "current_stage",
                    ""
                )
            ),

            "call_ended": result.get(
                "call_ended",
                False
            ),
        }