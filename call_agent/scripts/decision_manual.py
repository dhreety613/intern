import json

from call_agent.app.agents.decision_agent import decision_agent


def main():

    # ---------------------------------------------------------
    # Simulated data coming from the queue / Pipedrive card
    # ---------------------------------------------------------

    state = {
        "call_id": "test-call-001",

        # Customer information from queue
        "customer_name": "John Smith",
        "email": "john@company.com",
        "phone_number": "+1 412 555 1234",

        "calibration_dates": [
            "September 15, 2026"
        ],

        # Current customer message
        "user_message": "",

        # Empty initially.
        # Brain will populate this only if Decision needs context.
        "context": [],

        # Memory maintained separately by Memory Agent
        "memory": {},

        # Deterministic workflow state
        "call_progress": {
            "introduced": True,

            "call_redirected": False,

            "contact_confirmation_started": True,
            "contact_confirmation_complete": False,

            "calibration_reminder_started": False,
            "calibration_reminder_complete": False,

            "closing_started": False,
            "call_objective_complete": False,

            "topic_diversion_count": 0,
            "random_conversation_count": 0
        }
    }

    print("\n==========================================")
    print("       DECISION AGENT MANUAL TEST")
    print("==========================================\n")

    print("Queue/Card information:")
    print(
        f"Name: {state['customer_name']}"
    )
    print(
        f"Email: {state['email']}"
    )
    print(
        f"Phone: {state['phone_number']}"
    )
    print(
        f"Calibration: {state['calibration_dates']}"
    )

    print(
        "\nType ONLY what the customer says."
    )
    print(
        "Type 'exit' to stop.\n"
    )

    while True:

        user_message = input(
            "Customer: "
        ).strip()

        if user_message.lower() in {
            "exit",
            "quit"
        }:
            break

        if not user_message:
            continue

        state["user_message"] = user_message

        try:

            result = decision_agent(
                state
            )

            decision = result[
                "decision"
            ]

        except Exception as exc:

            print(
                "\nERROR:"
            )
            print(exc)
            print()

            continue

        print(
            "\n========== DECISION ==========\n"
        )

        print(
            json.dumps(
                decision,
                indent=2
            )
        )

        print(
            "\nAnne:"
        )

        print(
            decision["response"]
        )

        print()

        # -----------------------------------------------------
        # Simulate the orchestrator updating diversion state
        # -----------------------------------------------------

        if decision["topic_diversion"]:

            state[
                "call_progress"
            ][
                "topic_diversion_count"
            ] += 1

        else:

            state[
                "call_progress"
            ][
                "topic_diversion_count"
            ] = 0

        if decision["random_conversation"]:

            state[
                "call_progress"
            ][
                "random_conversation_count"
            ] += 1

        else:

            state[
                "call_progress"
            ][
                "random_conversation_count"
            ] = 0

        # -----------------------------------------------------
        # Show whether the brain would need context
        # -----------------------------------------------------

        if decision["needs_context"]:

            print(
                ">> Brain would now fetch the last 5 "
                "conversation lines and call Decision again."
            )

        # -----------------------------------------------------
        # End call
        # -----------------------------------------------------

        if decision[
            "present_task"
        ] == "END_CALL":

            print(
                "\n========== CALL ENDED ==========\n"
            )

            break


if __name__ == "__main__":
    main()