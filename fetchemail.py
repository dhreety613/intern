import os
import json
import requests
from dotenv import load_dotenv

# =====================================
# CONFIG
# =====================================

load_dotenv()

API_TOKEN = os.getenv("PIPEDRIVE_API_TOKEN")

if not API_TOKEN:
    raise ValueError("PIPEDRIVE_API_TOKEN not found in .env")

USER_ID = 11275953
QUICK_FILTER_ID = 391495

BASE_URL = "https://tensitron.pipedrive.com/api/v1/persons"
LIMIT = 500
START = 0

OUTPUT_FILE = "people_export.json"

# =====================================
# HELPERS
# =====================================

def split_name(full_name):
    full_name = (full_name or "").strip()

    if not full_name:
        return "", ""

    parts = full_name.split()

    first_name = parts[0]
    last_name = " ".join(parts[1:]) if len(parts) > 1 else ""

    return first_name, last_name


def extract_email(person):
    emails = person.get("email", [])

    if not emails:
        return None

    if isinstance(emails, list):
        for e in emails:
            if isinstance(e, dict):
                value = e.get("value")
                if value:
                    return value

    return None


# =====================================
# START
# =====================================

print("=" * 70)
print("PIPEDRIVE CONTACT EXPORT")
print("=" * 70)
print(f"User ID        : {USER_ID}")
print(f"Quick Filter   : {QUICK_FILTER_ID}")
print(f"Batch Size     : {LIMIT}")
print("=" * 70)

all_people = []

start = START
batch_number = 1

while True:

    print(
        f"\n[Batch {batch_number}] "
        f"Fetching contacts {start} -> {start + LIMIT}"
    )

    params = {
    "start": start,
    "limit": LIMIT,
    "api_token": API_TOKEN
}
    response = requests.get(BASE_URL, params=params)

    if response.status_code != 200:
        print(f"\nERROR {response.status_code}")
        print(response.text)
        break

    data = response.json()

    if not data.get("success"):
        print("\nAPI returned success=False")
        print(data)
        break

    people = data.get("data", [])

    if not people:
        print("\nNo more contacts found.")
        break

    for person in people:

        full_name = person.get("name", "").strip()

        first_name, last_name = split_name(full_name)

        email = extract_email(person)

        row = {
            "name": full_name,
            "first_name": first_name,
            "last_name": last_name,
        }

        if email:
            row["email"] = email
        else:
            row["has_email"] = False

        all_people.append(row)

    total_with_email = sum(
        1 for p in all_people if "email" in p
    )

    print(
        f"[Batch {batch_number}] "
        f"Fetched {len(people)} contacts | "
        f"Running Total: {len(all_people)} | "
        f"Emails Found: {total_with_email}"
    )

    # checkpoint every 500
    if len(all_people) % 500 == 0:

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                all_people,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"Checkpoint Saved -> "
            f"{len(all_people)} contacts"
        )

    if len(people) < LIMIT:
        print("\nLast page reached.")
        break

    start += LIMIT
    batch_number += 1

# =====================================
# FINAL SAVE
# =====================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        all_people,
        f,
        indent=2,
        ensure_ascii=False
    )

with_email = sum(
    1 for p in all_people if "email" in p
)

without_email = len(all_people) - with_email

print("\n" + "=" * 70)
print("EXPORT COMPLETE")
print("=" * 70)
print(f"Total Contacts   : {len(all_people)}")
print(f"With Email       : {with_email}")
print(f"Without Email    : {without_email}")
print(f"JSON File        : {OUTPUT_FILE}")
print("=" * 70)

print("\nFirst Record Preview:\n")

if all_people:
    print(json.dumps(all_people[0], indent=2))