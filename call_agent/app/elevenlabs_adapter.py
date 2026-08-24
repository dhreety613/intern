import json
from typing import Any, Dict

from call_agent.app.call import CallSession


# ------------------------------------------------------------
# One CallSession per ElevenLabs call_id
# ------------------------------------------------------------

_sessions: Dict[str, CallSession] = {}


def create_session(
    call_id: str,
    customer_name: str,
    email: str,
    phone_number: str,
    calibration_dates: list,
) -> CallSession:

    session = CallSession(
        customer_name=customer_name,
        email=email,
        phone_number=phone_number,
        calibration_dates=calibration_dates,
    )

    _sessions[call_id] = session

    return session


def get_session(
    call_id: str,
) -> CallSession | None:

    return _sessions.get(call_id)


def delete_session(
    call_id: str,
):
    _sessions.pop(
        call_id,
        None
    )


def start_call(
    call_id: str,
    customer_name: str,
    email: str,
    phone_number: str,
    calibration_dates: list,
) -> str:

    session = create_session(
        call_id=call_id,
        customer_name=customer_name,
        email=email,
        phone_number=phone_number,
        calibration_dates=calibration_dates,
    )

    return session.start()


def process_message(
    call_id: str,
    user_message: str,
) -> Dict[str, Any]:

    session = get_session(
        call_id
    )

    if session is None:
        raise ValueError(
            f"No active call session for call_id={call_id}"
        )

    return session.process(
        user_message
    )