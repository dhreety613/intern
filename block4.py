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


def upload_contacts(contacts):

    access_token = get_access_token()

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    uploaded = 0
    failed = []

    for i, contact in enumerate(contacts, start=1):

        response = requests.post(
            f"{API_BASE}/contacts",
            headers=headers,
            json=contact,
            timeout=60,
        )

        if response.ok:

            uploaded += 1

        else:

            failed.append({
                "index": i,
                "email": contact.get("email_address", {}).get("address"),
                "status": response.status_code,
                "response": response.text,
            })

    return {
        "received": len(contacts),
        "uploaded": uploaded,
        "failed": len(failed),
        "errors": failed,
    }