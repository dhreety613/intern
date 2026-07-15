
import os
import requests

CLIENT_ID = os.getenv("CONSTANT_CONTACT_CLIENT_ID")
CLIENT_SECRET = os.getenv("CONSTANT_CONTACT_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("CONSTANT_CONTACT_REFRESH_TOKEN")

API_BASE = "https://api.cc.email/v3"


def get_access_token():

    response = requests.post(
        "https://authz.constantcontact.com/oauth2/default/v1/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
        },
        auth=(CLIENT_ID, CLIENT_SECRET),
        timeout=30,
    )

    response.raise_for_status()

    return response.json()["access_token"]


def get_kingtester_main_list_id(headers):

    response = requests.get(
        f"{API_BASE}/contact_lists",
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    lists = response.json()["lists"]

    for lst in lists:

        if lst["name"].strip().lower() == "king - main":
            return lst["list_id"]

    raise Exception("Tensitron - Main list not found.")


def fetch_existing_cc_emails_king():

    access_token = get_access_token()

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }

    list_id = get_kingtester_main_list_id(headers)

    emails = set()

    next_url = (
        f"{API_BASE}/contacts"
        f"?lists={list_id}&limit=500"
    )

    while next_url:

        response = requests.get(
            next_url,
            headers=headers,
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        contacts = data.get("contacts", [])

        for contact in contacts:

            email = (
                contact.get("email_address", {})
                       .get("address", "")
                       .strip()
                       .lower()
            )

            if email:
                emails.add(email)

        next_link = data.get("_links", {}).get("next")

        if next_link:

            href = next_link["href"]

            if href.startswith("/"):

                next_url = (
                    "https://api.cc.email"
                    + href
                    + f"&lists={list_id}"
                )

            else:

                next_url = (
                    href
                    + f"&lists={list_id}"
                )

        else:

            next_url = None

    return {
        "count": len(emails),
        "emails": sorted(list(emails))
    }