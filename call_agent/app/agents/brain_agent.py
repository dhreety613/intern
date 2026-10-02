import json
import re
from typing import Any, Dict
import os

from google import genai

from call_agent.app.agents.decision_agent import CALIBRATION_FORM_GUIDE_STEPS
from call_agent.app.graph.state import CallState
from call_agent.app.schemas.brain import BrainOutput


#GEMINI_API_KEY = "AQ.Ab8RN6I9yLXUZOZMRIng4P0x8BNevM0_EERqN-u7pIw-2rFPJw"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")
MODEL = "gemini-3.5-flash-lite"

MODEL = "gemini-3.5-flash-lite"


def get_client():
    return genai.Client(
        api_key=GEMINI_API_KEY
    )


def extract_json(
    text: str,
) -> Dict[str, Any]:

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"```$",
        "",
        text,
    )

    return json.loads(
        text.strip()
    )


def get_next_stage(
    progress: Dict[str, bool],
) -> str:

    if not progress.get(
        "name_confirmed",
        False,
    ):
        return "name_confirmation"

    if not progress.get(
        "email_confirmed",
        False,
    ):
        return "email_confirmation"

    if not progress.get(
        "phone_confirmed",
        False,
    ):
        return "phone_confirmation"

    if not progress.get(
        "calibration_completed",
        False,
    ):
        return "calibration"

    if not progress.get(
        "calibration_guidance_offered",
        False,
    ):
        return "calibration_guidance"

    if (
        progress.get("calibration_guidance_wanted", False)
        and progress.get("calibration_guidance_step", 0) < len(CALIBRATION_FORM_GUIDE_STEPS)
    ):
        return "calibration_guidance"

    return "complete"


def brain_agent(
    state: CallState,
) -> Dict[str, Any]:

    client = get_client()

    old_progress = state.get(
        "progress",
        {
            "name_confirmed": False,
            "email_confirmed": False,
            "phone_confirmed": False,
            "calibration_completed": False,
            "calibration_guidance_offered": False,
            "calibration_guidance_wanted": False,
            "calibration_guidance_step": 0,
        },
    )

    current_stage = get_next_stage(
        old_progress
    )

    user_message = state.get(
        "user_message",
        "",
    )

    decision = state.get(
        "decision",
        {},
    )

    memory = state.get(
        "memory",
        {},
    )

    conversation = state.get(
        "conversation",
        [],
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Give Brain the previous conversation turns.
    # This is essential for short answers like "yes".
    # --------------------------------------------------------

    recent_context = conversation[-5:]

    prompt = f"""
You are the workflow Brain for Anne from KTC Instruments.

You do NOT speak to the customer.
Your ONLY job is to maintain permanent workflow progress.

WORKFLOW:
name → email → phone → calibration → calibration_guidance → complete

CURRENT PROGRESS:
{json.dumps(old_progress)}

CURRENT STAGE:
{current_stage}

RECENT CONVERSATION:
{json.dumps(recent_context)}

CURRENT CUSTOMER MESSAGE:
{user_message}

DECISION OUTPUT:
{json.dumps(decision)}

MEMORY:
{json.dumps(memory)}

RULES:

- Progress is permanent. TRUE can NEVER become FALSE.
- Only update the CURRENT unfinished stage.
- Never skip an unfinished stage.
- Customer interruptions, diversions, or corrections do not reset progress.
- Reconfirming an already-completed item does not make it FALSE.
- Use the recent conversation to interpret short replies such as
  "yes", "yeah", "correct", "sure", "no", or "that's right".
- A stage becomes TRUE only when Anne explicitly asked for that
  confirmation/completion AND the customer clearly confirmed it.
- Never mark a stage TRUE from assumption or from Anne's response alone.
- For contact confirmation, corrections do not undo previous confirmation.
- For calibration, mark calibration_completed TRUE only after Anne
  has given the calibration information and the customer acknowledges
  or responds to it.
- "complete" means the required workflow is finished.
  It does NOT mean the phone call should end.
- The Decision Agent alone determines whether the call ends.
-for calibration_completed - it is justa reminder, as soon as customer RESPONDS to the date reminder it turns TRUE.
-for calibration_guidance_offered - Anne asks whether the customer wants
  guidance on filling out the calibration form. As soon as the customer
  RESPONDS to that question (yes or no), calibration_guidance_offered
  turns TRUE. It only applies once calibration_completed is already TRUE.
-for calibration_guidance_wanted - TRUE only if the customer's answer to
  that same guidance question was affirmative (yes). Stays FALSE if they
  declined. Once TRUE it stays TRUE.
-for calibration_guidance_step - only relevant when calibration_guidance_wanted
  is TRUE. Anne reads the guidance steps one at a time. Each time the
  customer's CURRENT message confirms or acknowledges the step Anne had
  just read out (e.g. "ok", "got it", "done", "yes", "sure"), increase
  calibration_guidance_step by exactly 1 from its previous value. Never
  increase it by more than 1 per customer message, never decrease it,
  and do not increase it at all if the customer declined guidance
  (calibration_guidance_wanted is FALSE) or if calibration_guidance_offered
  is not yet TRUE.

IMPORTANT:

If the immediately preceding assistant message was:

"Is your email john@company.com?"

and the customer says:

"yes"

then email_confirmed = TRUE.

If the immediately preceding assistant message was:

"Is your phone number +1 412 555 1234?"

and the customer says:

"yes"

then phone_confirmed = TRUE.

Return ONLY JSON:

{{
  "progress": {{
    "name_confirmed": false,
    "email_confirmed": false,
    "phone_confirmed": false,
    "calibration_completed": false,
    "calibration_guidance_offered": false,
    "calibration_guidance_wanted": false,
    "calibration_guidance_step": 0
  }}
}}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        },
    )

    try:

        raw = extract_json(
            response.text
        )

        brain = BrainOutput(
            **raw
        )

    except Exception as exc:

        raise ValueError(
            "Brain Agent returned invalid output: "
            f"{response.text}"
        ) from exc

    proposed = brain.progress.model_dump()

    # --------------------------------------------------------
    # PERMANENT PROGRESS
    # --------------------------------------------------------

    final_progress = {

        "name_confirmed":
            old_progress.get(
                "name_confirmed",
                False,
            )
            or proposed.get(
                "name_confirmed",
                False,
            ),

        "email_confirmed":
            old_progress.get(
                "email_confirmed",
                False,
            )
            or proposed.get(
                "email_confirmed",
                False,
            ),

        "phone_confirmed":
            old_progress.get(
                "phone_confirmed",
                False,
            )
            or proposed.get(
                "phone_confirmed",
                False,
            ),

        "calibration_completed":
            old_progress.get(
                "calibration_completed",
                False,
            )
            or proposed.get(
                "calibration_completed",
                False,
            ),

        "calibration_guidance_offered":
            old_progress.get(
                "calibration_guidance_offered",
                False,
            )
            or proposed.get(
                "calibration_guidance_offered",
                False,
            ),

        "calibration_guidance_wanted":
            old_progress.get(
                "calibration_guidance_wanted",
                False,
            )
            or proposed.get(
                "calibration_guidance_wanted",
                False,
            ),

        "calibration_guidance_step":
            max(
                old_progress.get(
                    "calibration_guidance_step",
                    0,
                ),
                proposed.get(
                    "calibration_guidance_step",
                    0,
                ),
            ),
    }

    return {
        "brain_output": {
            "progress": final_progress,
            "current_stage": get_next_stage(
                final_progress
            ),
        }
    }