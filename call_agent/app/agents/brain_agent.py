import json
import re
from typing import Any, Dict
import os

from google import genai

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

    # ---- ORIGINAL BRAIN PROMPT (revert: delete the new prompt below, uncomment this) ----
    # prompt = f"""
    # You are the workflow Brain for Anne from Tensitron.
    #
    # You do NOT speak to the customer.
    # Your ONLY job is to maintain permanent workflow progress.
    #
    # WORKFLOW:
    # name → email → phone → calibration → complete
    #
    # (After "complete", a separate non-LLM system offers calibration-form
    # guidance and, if wanted, walks through it step by step - that part is
    # NOT tracked here; this Brain's job ends at calibration_completed.)
    #
    # CURRENT PROGRESS:
    # {json.dumps(old_progress)}
    #
    # CURRENT STAGE:
    # {current_stage}
    #
    # RECENT CONVERSATION:
    # {json.dumps(recent_context)}
    #
    # CURRENT CUSTOMER MESSAGE:
    # {user_message}
    #
    # DECISION OUTPUT:
    # {json.dumps(decision)}
    #
    # MEMORY:
    # {json.dumps(memory)}
    #
    # RULES:
    #
    # - Progress is permanent. TRUE can NEVER become FALSE.
    # - Only update the CURRENT unfinished stage.
    # - Never skip an unfinished stage.
    # - Customer interruptions, diversions, or corrections do not reset progress.
    # - Reconfirming an already-completed item does not make it FALSE.
    # - Use the recent conversation to interpret short replies such as
    #   "yes", "yeah", "correct", "sure", "no", or "that's right".
    # - A stage becomes TRUE only when Anne explicitly asked for that
    #   confirmation/completion AND the customer clearly confirmed it.
    # - Never mark a stage TRUE from assumption or from Anne's response alone.
    # - For contact confirmation, corrections do not undo previous confirmation.
    # - For calibration, mark calibration_completed TRUE only after Anne
    #   has given the calibration information and the customer acknowledges
    #   or responds to it.
    # - "complete" means the required workflow is finished.
    #   It does NOT mean the phone call should end.
    # - The Decision Agent alone determines whether the call ends.
    # -for calibration_completed - it is justa reminder, as soon as customer RESPONDS to the date reminder it turns TRUE.
    #
    # IMPORTANT:
    #
    # If the immediately preceding assistant message was:
    #
    # "Is your email john@company.com?"
    #
    # and the customer says:
    #
    # "yes"
    #
    # then email_confirmed = TRUE.
    #
    # If the immediately preceding assistant message was:
    #
    # "Is your phone number +1 412 555 1234?"
    #
    # and the customer says:
    #
    # "yes"
    #
    # then phone_confirmed = TRUE.
    #
    # Return ONLY JSON:
    #
    # {{
    #   "progress": {{
    # "name_confirmed": false,
    # "email_confirmed": false,
    # "phone_confirmed": false,
    # "calibration_completed": false
    #   }}
    # }}
    # """
    # ---- END ORIGINAL ----

    prompt = f"""
You are the workflow Brain for Anne from Tensitron. You never speak to the customer; you only maintain permanent workflow progress.

WORKFLOW: name → email → phone → calibration → complete
(After "complete" a separate non-LLM system handles calibration-form guidance; it is not tracked here.)

CURRENT PROGRESS: {json.dumps(old_progress)}
CURRENT STAGE: {current_stage}
RECENT CONVERSATION: {json.dumps(recent_context)}
CURRENT CUSTOMER MESSAGE: {user_message}
DECISION OUTPUT: {json.dumps(decision)}
MEMORY: {json.dumps(memory)}

RULES:
- Progress is permanent: TRUE never becomes FALSE. Interruptions, diversions, corrections and reconfirmations never reset it.
- Only update the CURRENT unfinished stage; never skip a stage.
- Use the recent conversation to read short replies ("yes", "yeah", "correct", "sure", "no", "that's right").
- A contact stage becomes TRUE only when Anne explicitly asked for that confirmation AND the customer clearly confirmed it (e.g. Anne: "Is your email john@company.com?" + customer "yes" → email_confirmed TRUE; same for phone). Never from assumption or Anne's reply alone.
- calibration_completed is just a reminder: it turns TRUE as soon as the customer RESPONDS to the date reminder.
- "complete" does not mean the call ends; only the Decision Agent ends calls.

Return ONLY JSON:
{{
  "progress": {{
    "name_confirmed": false,
    "email_confirmed": false,
    "phone_confirmed": false,
    "calibration_completed": false
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
    }

    return {
        "brain_output": {
            "progress": final_progress,
            "current_stage": get_next_stage(
                final_progress
            ),
        }
    }
