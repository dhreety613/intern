from typing import Any, Dict, List


def get_recent_context(
    conversation: List[Dict[str, str]],
    limit: int = 12,
) -> List[Dict[str, str]]:

    if not conversation:
        return []

    return conversation[-limit:]