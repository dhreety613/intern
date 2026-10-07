from typing import List, Optional

from pydantic import BaseModel, Field


class CallMemory(BaseModel):

    # ---------------------------------------------------------
    # Current known customer details
    # ---------------------------------------------------------

    customer_name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None

    # ---------------------------------------------------------
    # Latest customer corrections
    # These overwrite previous values when explicitly corrected.
    # ---------------------------------------------------------

    new_name: Optional[str] = None
    new_email: Optional[str] = None
    new_phone_number: Optional[str] = None

    # ---------------------------------------------------------
    # Calibration information
    # ---------------------------------------------------------

    calibration_dates: List[str] = Field(
        default_factory=list
    )

    calibration_information: List[str] = Field(
        default_factory=list
    )

    # ---------------------------------------------------------
    # Email requests
    # ---------------------------------------------------------

    email_calibration_confirmation: bool = False
    email_crm_update_confirmation: bool = False

    # ---------------------------------------------------------
    # Call outcome / follow-up information
    # ---------------------------------------------------------

    do_not_call: bool = False

    wrong_number_so_avoid: bool = False

    # Customer wants another call (free text, as said)
    call_again_when: Optional[str] = None

    # The earliest date (YYYY-MM-DD) we may call again, ONLY when the
    # customer gave a proper date. Used to build the n8n calling queue.
    next_call_date: Optional[str] = None

    # Customer wants a human representative
    wants_human_call: bool = False

    # Specific person requested, if mentioned
    wants_who: Optional[str] = None

    # When they want the human call
    wants_human_call_when: Optional[str] = None

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------

    all_tasks_ended: bool = False

    # ---------------------------------------------------------
    # Useful facts for report / queue
    # ---------------------------------------------------------

    important_facts: List[str] = Field(
        default_factory=list
    )