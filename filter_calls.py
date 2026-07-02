from collections import Counter

VALID_AGENTS = {
    "Tyler Schinosi",
    "Travis Jeke",
    "Ian Semasko",
    "Alex De Jesus",
    "Matthew Pelton"
}

VALID_TYPES = {
    "incoming",
    "outgoing",
    "inbound",
    "outbound"
}

MIN_DURATION = 180


def filter_the_calls(calls):

    filtered = []

    for call in calls:

        cdr = call.get("Cdr", {})
        agent = call.get("Agent", {})

        call_type = cdr.get("type", "").lower()

        if call_type not in VALID_TYPES:
            continue

        first = (agent.get("firstname") or "").strip()
        last = (agent.get("lastname") or "").strip()

        agent_name = f"{first} {last}".strip()

        if agent_name not in VALID_AGENTS:
            continue

        try:
            duration = int(cdr.get("talking_time", 0))
        except:
            continue

        if duration < MIN_DURATION:
            continue

        started = cdr.get("started_at", "")

        filtered.append({
            "call_id": cdr.get("id"),
            "agent": agent_name,
            "started_at": started,
            "date": started[:10],
            "time": started[11:19].replace(":", "-"),
            "duration_seconds": duration
        })

    summary = dict(Counter(c["agent"] for c in filtered))

    return {
        "success": True,
        "original_calls": len(calls),
        "filtered_calls": len(filtered),
        "agent_summary": summary,
        "filtered_calls": filtered
    }