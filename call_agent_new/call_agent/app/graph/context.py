from typing import Any, Dict, List


def get_recent_context(
    conversation: List[Dict[str, str]],
    limit: int = 5,
) -> List[Dict[str, str]]:

    if not conversation:
        return []

    return conversation[-limit:]