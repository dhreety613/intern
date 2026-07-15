import os
import time
import requests


API_TOKEN = os.getenv("PIPEDRIVE_TOKEN")
BASE_URL = "https://tensitron.pipedrive.com/api/v1"

SALES_STAGES = {
    98: "Installed Base",
    86: "Emails (6 Month)",
    99: "Email (3 Month)",
    100: "(Email) 1 Month",
    89: "Due",
    90: "Missed",
    91: "Submitted Form",
    112: "In House",
    92: "Quoted/Estimate Made",
    93: "Payment Confirmed",
    94: "Device Shipped / New Purchase",
    95: "DNR/Others",
    113: "Unknown Status",
}

def get_sales_deal_ids_for_actsummary():


    deal_ids = []
    seen_ids = set()

    for stage_id, stage_name in SALES_STAGES.items():

        print(f"\n========== {stage_name} ({stage_id}) ==========")

        start = 0

        while True:

            params = {
                "api_token": API_TOKEN,
                "stage_id": stage_id,
                "status": "open",
                "start": start,
                "limit": 500,
            }

            response = requests.get(
                f"{BASE_URL}/deals",
                params=params,
                timeout=30,
            )

            response.raise_for_status()

            result = response.json()

            deals = result.get("data") or []

            if not deals:
                break

            for deal in deals:
                deal_id = deal["id"]

                if deal_id not in seen_ids:
                    seen_ids.add(deal_id)
                    deal_ids.append(deal_id)

            pagination = (
                result.get("additional_data", {})
                .get("pagination", {})
            )

            if not pagination.get("more_items_in_collection", False):
                break

            start = pagination["next_start"]

            time.sleep(0.8)

    return {
        "count": len(deal_ids),
        "deal_ids": deal_ids
    }