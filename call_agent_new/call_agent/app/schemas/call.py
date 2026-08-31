from pydantic import BaseModel, Field


class CallRequest(BaseModel):
    """
    Input received for a conversational turn.
    """

    call_id: str = Field(min_length=1)
    user_message: str = Field(min_length=1)


class CallEndRequest(BaseModel):
    """
    Request received when a call ends.
    """

    call_id: str = Field(min_length=1)


class CallResponse(BaseModel):
    """
    Response returned after the call layer processes
    a conversational turn.
    """

    call_id: str
    response: str
    end_call: bool = False