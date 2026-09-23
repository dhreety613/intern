from typing import Any, Dict, List, TypedDict


class CallState(TypedDict, total=False):

    # --------------------------------------------------
    # Current customer input
    # --------------------------------------------------

    user_message: str

    # --------------------------------------------------
    # Conversation
    # --------------------------------------------------

    conversation: List[Dict[str, str]]

    context: List[Dict[str, str]]

    # --------------------------------------------------
    # Queue / CRM information
    # --------------------------------------------------

    customer_name: str
    email: str
    phone_number: str
    calibration_dates: List[str]
    products: List[str]

    # --------------------------------------------------
    # Persistent call memory
    # --------------------------------------------------

    memory: Dict[str, Any]

    # --------------------------------------------------
    # Brain workflow state
    # --------------------------------------------------

    progress: Dict[str, bool]

    current_stage: str

    brain_output: Dict[str, Any]

    # --------------------------------------------------
    # Decision output
    # --------------------------------------------------

    decision: Dict[str, Any]

    # --------------------------------------------------
    # Other call-level progress
    # --------------------------------------------------

    call_progress: Dict[str, Any]

    # --------------------------------------------------
    # Call status
    # --------------------------------------------------

    call_ended: bool