from typing import Literal

from pydantic import BaseModel, Field


class DecisionOutput(BaseModel):
    response: str = Field(
        ...,
        description="Short natural spoken response."
    )

    present_task: Literal[
        "CONTINUE",
        "WAIT",
        "CLARIFY",
        "END_CALL"
    ]

    needs_context: bool = Field(
        default=False,
        description=(
            "True only when the current message cannot be "
            "understood without previous conversation."
        )
    )

    topic_diversion: bool = Field(
        default=False,
        description=(
            "True when the caller has gone off the purpose "
            "of the call."
        )
    )

    random_conversation: bool = Field(
        default=False,
        description=(
            "True when the caller is repeatedly engaging in "
            "random or unproductive conversation."
        )
    )

    redirect_to_rep: bool = Field(
        default=False,
        description=(
            "True only when the caller has explicitly confirmed (yes) "
            "that the call should be redirected to a representative "
            "now - whether that offer came from a guardrail miss or "
            "from the caller directly asking for a person/human/rep."
        )
    )

    guidance_requested: bool = Field(
        default=False,
        description=(
            "True when the caller asks for the calibration form "
            "walkthrough/guidance (any wording, any point in the call). "
            "The orchestrator then replaces the response with the "
            "scripted walkthrough."
        )
    )

    redirect_reason: str = Field(
        default="",
        description=(
            "Only set when redirect_to_rep is true: a short natural-"
            "language phrase for what the caller actually needs help "
            "with (e.g. 'a pricing question', 'an order status update'). "
            "Not limited to calibration - whatever the caller's topic "
            "actually was."
        )
    )

    reminder_given: bool = Field(
        default=False,
        description=(
            "True when this reply delivers the calibration reminder "
            "(serial number(s) + date). The orchestrator then makes sure "
            "the form-guidance offer is part of the SAME spoken turn."
        )
    )

    will_fetch_person: bool = Field(
        default=False,
        description=(
            "True when the speaker says they will get / transfer / hand the "
            "phone to someone else (e.g. 'hold on, let me get him')."
        )
    )

    new_person_joined: bool = Field(
        default=False,
        description=(
            "True when the person speaking NOW is a different person from "
            "whoever was speaking before (someone was fetched, the phone was "
            "handed over, a colleague/family member took over). The "
            "orchestrator then greets again and restarts the workflow."
        )
    )
