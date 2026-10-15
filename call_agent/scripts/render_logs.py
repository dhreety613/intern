"""Pull the call agent's TRACK lines from Render.

    RENDER_API_KEY=... python -m call_agent.scripts.render_logs [--call CAxxxx] [--deal 43776] [--hours 2]
"""
import argparse
import os
from datetime import datetime, timedelta, timezone

import httpx

OWNER_ID = "tea-d3co33ali9vc73dol5u0"
SERVICE_ID = "srv-d8nvgvog4nts73bmb8i0"  # fetch_history (hosts /call-agent)


def fetch(hours: float, text: str):
    key = os.environ["RENDER_API_KEY"]
    params = {
        "ownerId": OWNER_ID,
        "resource": SERVICE_ID,
        "text": text,
        "direction": "forward",
        "limit": 100,
        "startTime": (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(),
    }
    while True:
        r = httpx.get("https://api.render.com/v1/logs", params=params,
                      headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        d = r.json()
        for line in d.get("logs", []):
            yield line
        if not d.get("hasMore"):
            return
        params["startTime"], params["endTime"] = d["nextStartTime"], d["nextEndTime"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--call")
    ap.add_argument("--deal")
    ap.add_argument("--hours", type=float, default=2)
    a = ap.parse_args()
    for line in fetch(a.hours, a.call or (f"deal={a.deal}" if a.deal else "TRACK")):
        if "TRACK |" in line["message"]:
            print(line["timestamp"][11:19], line["message"])
