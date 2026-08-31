import json

from call_agent.app.agents.memory_agent import memory_agent


def main():

    state = {
        "call_id": "test-call-001",

        "user_message": "",

        # Recent context.
        # For this standalone test we can leave it empty.
        # The graph will populate it later when needed.
        "context": [],

        # Initial memory from the call/queue.
        "memory": {
            "customer_name": "John Smith",
            "email": "john@company.com",
            "phone_number": "+1 412 555 1234",

            "new_name": None,
            "new_email": None,
            "new_phone_number": None,

            "calibration_dates": [
                "September 15, 2026"
            ],

            "calibration_information": [],

            "email_calibration_confirmation": False,
            "email_crm_update_confirmation": False,

            "do_not_call": False,
            "wrong_number_so_avoid": False,

            "call_again_when": None,

            "wants_human_call": False,
            "wants_who": None,
            "wants_human_call_when": None,

            "all_tasks_ended": False,

            "important_facts": []
        }
    }

    print("\n==========================================")
    print("          MEMORY AGENT MANUAL TEST")
    print("==========================================\n")

    print("Initial customer information:")
    print(
        f"Name: {state['memory']['customer_name']}"
    )
    print(
        f"Email: {state['memory']['email']}"
    )
    print(
        f"Phone: {state['memory']['phone_number']}"
    )
    print(
        f"Calibration: {state['memory']['calibration_dates']}"
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

            result = memory_agent(
                state
            )

            # Persist memory for the next turn.
            state["memory"] = result[
                "memory"
            ]

        except Exception as exc:

            print(
                "\nERROR:"
            )
            print(exc)
            print()

            continue

        print(
            "\n========== UPDATED MEMORY ==========\n"
        )

        print(
            json.dumps(
                state["memory"],
                indent=2
            )
        )

        print()


if __name__ == "__main__":
    main()