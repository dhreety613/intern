"""End-of-call result record, sent to n8n so it can build the calling queue.

Delivered to CALL_RESULT_WEBHOOK_URL (optional header X-Webhook-Secret from
CALL_RESULT_WEBHOOK_SECRET). Always logged as "CALL RESULT: {...}" too.
"""

import json
import logging
import os
from datetime import date
from typing import Any, Dict

import httpx

logger = logging.getLogger(__name__)


def build_call_result(
    state: Dict[str, Any],
    call_sid: str,
    person_id: str = "",
    deal_id: str = "",
) -> Dict[str, Any]:
    memory = state.get("memory") or {}
    ivr = state.get("ivr") or {}

    human = bool(state.get("human_confirmed"))
    voicemail = bool(state.get("is_voicemail"))
    wrong_number = bool(memory.get("wrong_number_so_avoid"))

    # A call counts as successful only if a real person was reached and it
    # was the right person. Voicemail, IVR dead-ends, no answer and wrong
    # numbers are not successful.
    if voicemail:
        outcome = "voicemail"
    elif wrong_number:
        outcome = "wrong_number"
    elif human:
        outcome = "completed"
    elif ivr.get("seen"):
        outcome = "ivr_no_human"
    else:
        outcome = "no_conversation"
    successful = outcome == "completed"

    return {
        "call_sid": call_sid,
        "person_id": person_id,
        "deal_id": deal_id,
        "phone_number": state.get("phone_number", ""),
        "outcome": outcome,
        "call_successful": successful,
        # null when unsuccessful: the caller must keep the older date.
        "last_call_date": date.today().isoformat() if successful else None,
        "do_not_call": bool(memory.get("do_not_call")),
        "next_call_date": memory.get("next_call_date"),
        "call_again_when": memory.get("call_again_when"),
        "wrong_number_so_avoid": wrong_number,
        "new_name": memory.get("new_name"),
        "new_email": memory.get("new_email"),
        "new_phone_number": memory.get("new_phone_number"),
        "wants_human_call": bool(memory.get("wants_human_call")),
        "important_facts": memory.get("important_facts") or [],
    }


async def deliver_call_result(result: Dict[str, Any]) -> None:
    logger.info("CALL RESULT: %s", json.dumps(result, default=str))

    url = os.getenv("CALL_RESULT_WEBHOOK_URL")
    if not url:
        return

    headers = {}
    secret = os.getenv("CALL_RESULT_WEBHOOK_SECRET")
    if secret:
        headers["X-Webhook-Secret"] = secret

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=result, headers=headers)
            logger.info("Call result webhook -> HTTP %s", resp.status_code)
    except Exception as e:
        logger.error("Call result webhook failed: %s", e)
