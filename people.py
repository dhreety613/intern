import os
import requests

API_TOKEN = os.getenv("PIPEDRIVE_TOKEN")

USER_ID = 11275953
QUICK_FILTER_ID = 391495

BASE_URL = "https://tensitron.pipedrive.com/api/v1/persons"

LIMIT = 500


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

    for e in emails:
        if isinstance(e, dict):
            value = e.get("value")
            if value:
                return value

    return None


def fetch_people():

    all_people = []
    start = 0

    while True:

        params = {
            "start": start,
            "limit": LIMIT,
            "api_token": API_TOKEN
        }

        response = requests.get(
            BASE_URL,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        people = data.get("data", [])

        if not people:
            break

        for person in people:

            full_name = person.get(
                "name",
                ""
            ).strip()

            first_name, last_name = split_name(
                full_name
            )

            email = extract_email(
                person
            )

            row = {
                "name": full_name,
                "first_name": first_name,
                "last_name": last_name
            }

            if email:
                row["email"] = email

            all_people.append(row)

        if len(people) < LIMIT:
            break

        start += LIMIT

    return {
        "count": len(all_people),
        "people": all_people
    }