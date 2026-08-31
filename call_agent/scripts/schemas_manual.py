from call_agent.app.schemas.call import (
    CallRequest,
    CallEndRequest,
    CallResponse,
)

from call_agent.app.schemas.decision import (
    Task,
    Decision,
)

from call_agent.app.schemas.report import (
    ReportInput,
    CallReport,
)


def test_call_schemas():

    print("\n========== CALL SCHEMAS ==========\n")

    call_id = input("Call ID: ")
    message = input("User message: ")

    request = CallRequest(
        call_id=call_id,
        user_message=message
    )

    print("\nCallRequest:")
    print(request.model_dump())

    response = CallResponse(
        call_id=call_id,
        response="Example response",
        end_call=False
    )

    print("\nCallResponse:")
    print(response.model_dump())


def test_decision_schema():

    print("\n========== DECISION SCHEMA ==========\n")

    task_type = input(
        "Task type (leave empty for no task): "
    )

    tasks = []

    if task_type:

        task = Task(
            type=task_type,
            payload={
                "example": "test"
            }
        )

        tasks.append(task)

    decision = Decision(
        response="This is an example response.",
        tasks=tasks,
        end_call=False
    )

    print("\nDecision:")
    print(decision.model_dump())


def test_report_schema():

    print("\n========== REPORT SCHEMA ==========\n")

    report = CallReport(
        summary=[
            "Customer discussed product requirements.",
            "Customer requested follow-up."
        ],
        call_state="FOLLOW_UP",
        next_action="Contact customer."
    )

    print("\nCallReport:")
    print(report.model_dump())


if __name__ == "__main__":

    test_call_schemas()

    test_decision_schema()

    test_report_schema()

    print("\n========== STEP 1 COMPLETE ==========\n")