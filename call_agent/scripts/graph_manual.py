import json

from call_agent.app.call import CallSession


def main():

    print()
    print("=" * 50)
    print("        FULL CALL ORCHESTRATOR TEST")
    print("=" * 50)
    print()

    session = CallSession(

        customer_name="John Smith",

        email="john@company.com",

        phone_number="+1 412 555 1234",

        calibration_dates=[
            "September 15, 2026"
        ],
    )

    # --------------------------------------------------------
    # Initial Anne introduction
    # --------------------------------------------------------

    opening = (
        "Hi, this is Anne from Tensitron. "
        "Am I speaking with John Smith?"
    )

    print("Anne:")
    print(opening)
    print()

    session.add_message(
        "assistant",
        opening
    )

    print(
        "Type ONLY what the customer says."
    )

    print(
        "Type 'exit' to stop."
    )

    print()

    # --------------------------------------------------------
    # Conversation loop
    # --------------------------------------------------------

    while True:

        user_message = input(
            "Customer: "
        ).strip()

        if user_message.lower() == "exit":

            break

        if not user_message:

            continue

        print()

        print(
            "[Running Memory + Brain → Decision...]"
        )

        try:

            result = session.process(
                user_message
            )

        except Exception as exc:

            print()
            print("ERROR:")
            print(exc)
            print()

            continue

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        print()

        print("Anne:")

        print(
            result["response"]
        )

        print()

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        print(
            "[Task:",
            result["decision"].get(
                "present_task"
            ),
            "]"
        )

        print(
            "[Current Stage:",
            result["current_stage"],
            "]"
        )

        # ----------------------------------------------------
        # Brain
        # ----------------------------------------------------

        print()
        print(
            "========== BRAIN =========="
        )

        print(
            json.dumps(
                result["brain"],
                indent=2
            )
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        print()
        print(
            "========== PROGRESS =========="
        )

        print(
            json.dumps(
                result["progress"],
                indent=2
            )
        )

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        print()
        print(
            "========== MEMORY =========="
        )

        print(
            json.dumps(
                result["memory"],
                indent=2
            )
        )

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        print()
        print(
            "========== DECISION =========="
        )

        print(
            json.dumps(
                result["decision"],
                indent=2
            )
        )

        print()
        print("=" * 50)
        print()

        # ----------------------------------------------------
        # ONLY Decision can end the call
        # ----------------------------------------------------

        if result["call_ended"]:

            print(
                "CALL ENDED"
            )

            break

    # --------------------------------------------------------
    # Final snapshot
    # --------------------------------------------------------

    print()
    print(
        "========== FINAL CALL SNAPSHOT =========="
    )

    print(
        json.dumps(
            session.state,
            indent=2
        )
    )


if __name__ == "__main__":
    main()