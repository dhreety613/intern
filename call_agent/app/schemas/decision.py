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