from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ReportInput(BaseModel):
    """
    Final snapshot of the temporary call state.

    This is sent to the Report Agent after the call ends.
    """

    call_id: str

    conversation: List[Dict[str, Any]]

    memory: Dict[str, Any]

    intent: Dict[str, Any]

    guardrail: Dict[str, Any]

    decisions: List[Dict[str, Any]]

    tasks: List[Dict[str, Any]]


class CallReport(BaseModel):
    """
    Final report produced by the Report Agent.
    """

    summary: List[str] = Field(min_length=1, max_length=3)

    call_state: str = Field(min_length=1)

    next_action: Optional[str] = None