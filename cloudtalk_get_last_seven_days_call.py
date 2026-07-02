import json
import os
from datetime import datetime, timedelta

import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()

ACCESS_KEY_ID = os.getenv("CLOUDTALK_KEY_ID")
ACCESS_KEY_SECRET = os.getenv("CLOUDTALK_KEY_SECRET")

BASE_URL = "https://my.cloudtalk.io/api/calls/index.json"

STORAGE_DIR = "storage"
OUTPUT_FILE = os.path.join(STORAGE_DIR, "calls_last7days.json")


def fetch_calls():

    os.makedirs(STORAGE_DIR, exist_ok=True)

    today = datetime.utcnow()
    yesterday = today - timedelta(days=1)
    week_ago = yesterday - timedelta(days=5)

    FROM_DATE = week_ago.strftime("%Y-%m-%d")
    TO_DATE = yesterday.strftime("%Y-%m-%d")

    LIMIT = 500
    page = 1

    all_calls = []

    while True:

        params = {
            "page": page,
            "limit": LIMIT,
            "date_from": FROM_DATE,
            "date_to": TO_DATE
        }

        r = requests.get(
            BASE_URL,
            params=params,
            auth=HTTPBasicAuth(
                ACCESS_KEY_ID,
                ACCESS_KEY_SECRET
            ),
            timeout=60
        )

        if r.status_code != 200:
            return {
                "success": False,
                "status": r.status_code,
                "message": r.text
            }

        data = r.json()

        if "responseData" not in data:
            return {
                "success": False,
                "message": "Unexpected API response",
                "response": data
            }

        response = data["responseData"]

        calls = response.get("data", [])

        all_calls.extend(calls)

        if len(calls) < LIMIT:
            break

        page += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_calls, f, indent=4)

    return {
        "success": True,
        "total_calls": len(all_calls),
        "from": FROM_DATE,
        "to": TO_DATE,
        "file": OUTPUT_FILE
    }