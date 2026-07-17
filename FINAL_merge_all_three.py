import os
import time
import requests

# =====================================================
# CONFIG
# =====================================================

DOMAINS = {
    "tensitron": {
        "domain": "tensitron",
        "token": os.getenv("PIPEDRIVE_TOKEN"),
    },
    "cianflone": {
        "domain": "cianflone",
        "token": os.getenv("CIANFLONE_TOKEN"),
    },
    "kingtester": {
        "domain": "kingtester",
        "token": os.getenv("KING_TOKEN"),
    },
}


# =====================================================
# HELPERS
# =====================================================

def get_session(company):

    token = DOMAINS[company]["token"]

    session = requests.Session()
    session.params = {
        "api_token": token
    }

    return session


def get_base_url(company):

    return f"https://{DOMAINS[company]['domain']}.pipedrive.com/api/v1"


def fetch_duplicate_groups(company, entity):

    session = get_session(company)

    r = session.get(
        f"{get_base_url(company)}/duplicate-spotter/{entity}",
        timeout=120,
    )

    r.raise_for_status()

    return r.json().get("data", [])


def merge_group(company, entity, group):

    session = get_session(company)

    group_id = group["_id"]
    duplicate_ids = group["duplicate_ids"]

    if len(duplicate_ids) < 2:
        return {
            "success": False,
            "group_id": group_id,
            "reason": "Only one duplicate",
        }

    payload = {
        "primary_id": duplicate_ids[0],
        "checked_ids": ",".join(map(str, duplicate_ids)),
    }

    url = (
        f"{get_base_url(company)}"
        f"/duplicate-spotter/{entity}/{group_id}/merge"
    )

    try:

        r = session.put(
            url,
            json=payload,
            timeout=120,
        )

        if r.ok:

            return {
                "success": True,
                "group_id": group_id,
                "primary_id": duplicate_ids[0],
            }

        return {
            "success": False,
            "group_id": group_id,
            "primary_id": duplicate_ids[0],
            "status": r.status_code,
            "response": r.text,
        }

    except Exception as e:

        return {
            "success": False,
            "group_id": group_id,
            "primary_id": duplicate_ids[0],
            "error": str(e),
        }


# =====================================================
# MAIN FUNCTION
# =====================================================

def merge_duplicates(company, entity):

    groups = fetch_duplicate_groups(company, entity)

    passed = 0
    failed = 0

    failed_groups = []

    for i, group in enumerate(groups, start=1):

        result = merge_group(company, entity, group)

        if result["success"]:
            passed += 1
        else:
            failed += 1
            failed_groups.append(result)

        print(f"[{company}] [{entity}] {i}/{len(groups)}")

        time.sleep(0.5)

    print("\n" + "=" * 70)
    print(f"{company.upper()} | {entity.upper()}")
    print("=" * 70)
    print(f"Total Groups : {len(groups)}")
    print(f"Passed       : {passed}")
    print(f"Failed       : {failed}")

    if failed_groups:

        print("\nFailed Groups\n")

        for fg in failed_groups:

            print("-----------------------------------")
            print("Group ID   :", fg.get("group_id"))
            print("Primary ID :", fg.get("primary_id"))
            print("Reason     :", fg.get("reason"))
            print("Status     :", fg.get("status"))
            print("Error      :", fg.get("error"))
            print("Response   :", fg.get("response"))

    print("=" * 70)

    return {
        "company": company,
        "entity": entity,
        "total_groups": len(groups),
        "passed": passed,
        "failed": failed,
        "failed_groups": failed_groups,
    }