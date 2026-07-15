import requests
import json
from bs4 import BeautifulSoup
from html import unescape
from datetime import datetime
import os

class PipedriveDealHistoryFetcher:
    

    def __init__(self, domain, api_token):

        self.domain = domain
        self.api_token = api_token
        self.base_url = f"https://{domain}.pipedrive.com/api/v1"

    # =====================================================
    # CORE
    # =====================================================

    def _get(self, endpoint, params=None):

        params = params or {}
        params["api_token"] = self.api_token

        r = requests.get(
            f"{self.base_url}{endpoint}",
            params=params,
            timeout=60
        )

        r.raise_for_status()

        return r.json()

    # =====================================================
    # HTML CLEANER
    # =====================================================

    @staticmethod
    def clean_html(text):

        if not text:
            return ""

        text = unescape(text)

        return BeautifulSoup(
            text,
            "html.parser"
        ).get_text(
            separator="\n",
            strip=True
        )

    # =====================================================
    # DATE PARSER
    # =====================================================

    @staticmethod
    def parse_date(value):

        if not value:
            return None

        try:
            return datetime.fromisoformat(
                value.replace("Z", "")
            )
        except:
            pass

        try:
            return datetime.strptime(
                value,
                "%Y-%m-%d %H:%M:%S"
            )
        except:
            pass

        return None
    
    @staticmethod
    def filter_after(items, field_name, cutoff):

        if not cutoff:
            return items

        cutoff_dt = PipedriveDealHistoryFetcher.parse_date(
            cutoff
    )

        if not cutoff_dt:
            return items

        filtered = []

        for item in items:

            item_dt = PipedriveDealHistoryFetcher.parse_date(
                item.get(field_name)
        )

            if (
                item_dt
                and item_dt > cutoff_dt
            ):
                filtered.append(item)

        return filtered

    # =====================================================
    # ACTIVITIES
    # =====================================================

    def get_activities(self, deal_id):

        print("\nSTEP 1 - ACTIVITIES")

        data = self._get(
            f"/deals/{deal_id}/activities"
        )

        results = []

        for a in (data.get("data") or []):

            results.append({

                "id":
                    a.get("id"),

                "type":
                    a.get("type"),

                "subject":
                    a.get("subject"),

                "text":
                    self.clean_html(
                        a.get("note")
                    ),

                "created_at":
                    a.get("add_time"),

                "updated_at":
                    a.get("update_time")
            })

        print(
            f"Activities fetched: {len(results)}"
        )

        return results

    # =====================================================
    # NOTES
    # =====================================================

    def get_notes(self, deal_id):

        print("\nSTEP 2 - NOTES")

        data = self._get(
            f"/deals/{deal_id}/notes"
        )

        notes = []

        for n in (data.get("data") or []):

            notes.append({

                "id":
                    n.get("id"),

                "text":
                    self.clean_html(
                        n.get("content")
                    ),

                "created_at":
                    n.get("add_time"),

                "updated_at":
                    n.get("update_time")
            })

        print(
            f"Notes fetched: {len(notes)}"
        )

        return notes

    # =====================================================
    # EMAIL LIST
    # =====================================================

    def get_email_messages(self, deal_id):

        print("\nSTEP 3 - EMAIL HEADERS")

        data = self._get(
            f"/deals/{deal_id}/mailMessages"
        )

        emails = []

        for item in (data.get("data") or []):

            if item.get("object") != "mailMessage":
                continue

            emails.append(
                item.get("data", {})
            )

        print(
            f"Email headers fetched: {len(emails)}"
        )

        return emails

    # =====================================================
    # EMAIL DETAILS
    # =====================================================

    def get_email_details(self, message_id):

        try:

            data = self._get(
                f"/mailbox/mailMessages/{message_id}"
            )

            return data.get("data", {})

        except Exception as e:

            print(
                f"FAILED EMAIL DETAIL {message_id}: {e}"
            )

            return {}

    # =====================================================
    # EMAIL BODY
    # =====================================================

    def fetch_email_body(self, body_url):

        if not body_url:
            return ""

        try:

            r = requests.get(
                body_url,
                timeout=60
            )

            if r.status_code != 200:
                return ""

            return self.clean_html(
                r.text
            )

        except Exception as e:

            print(
                f"BODY DOWNLOAD FAILED: {e}"
            )

            return ""

    # =====================================================
    # EMAILS
    # =====================================================

    def get_emails(
            self,
        deal_id,
        latest_email_date=None
):

        headers = self.get_email_messages(
            deal_id
    )

        cutoff_dt = self.parse_date(
            latest_email_date
    )

        filtered_headers = []

        for email in headers:

            msg_time = email.get(
                "message_time"
        )

            msg_dt = self.parse_date(
                msg_time
        )

            if (
                cutoff_dt
                and msg_dt
                and msg_dt <= cutoff_dt
        ):
                continue

            filtered_headers.append(
                email
        )

        emails = []

        total = len(
            filtered_headers
    )

        print(
            f"\nSTEP 4 - FETCHING {total} NEW EMAILS"
    )

        for i, email in enumerate(
            filtered_headers,
            start=1
    ):

            msg_id = email.get("id")

            print(
                f"[{i}/{total}] Email ID {msg_id}"
        )

            details = self.get_email_details(
                msg_id
        )

            body = ""

            if details.get(
                "has_body_flag"
        ):

                body = self.fetch_email_body(
                    details.get(
                        "body_url"
                )
            )

            emails.append({

                "id":
                    details.get("id"),

                "thread_id":
                    details.get(
                        "mail_thread_id"
                ),

                "subject":
                    details.get(
                        "subject"
                ),

                "snippet":
                    details.get(
                        "snippet"
                ),

                "body":
                    body,

                "from":
                    details.get(
                        "from",
                        []
                ),

                "to":
                    details.get(
                        "to",
                        []
                ),

                "created_at":
                    details.get(
                        "message_time"
                )
        })

        print(
            f"\nEmails fetched: {len(emails)}"
    )

        return emails
    # =====================================================
    # LATEST CALL
    # =====================================================

    def latest_call(self, activities):

        calls = []

        for a in activities:

            activity_type = str(
                a.get(
                    "type",
                    ""
                )
            ).lower()

            subject = str(
                a.get(
                    "subject",
                    ""
                )
            ).lower()

            if (
                activity_type != "call"
                and subject != "call"
            ):
                continue

            dt = self.parse_date(
                a.get("updated_at")
            )

            if not dt:
                continue

            calls.append({

                "date":
                    dt,

                "activity_id":
                    a.get("id"),

                "subject":
                    a.get("subject"),

                "text":
                    a.get("text")
            })

        if not calls:
            return None

        latest = max(
            calls,
            key=lambda x: x["date"]
        )

        return {

            "date":
                latest["date"].isoformat(),

            "activity_id":
                latest["activity_id"],

            "subject":
                latest["subject"],

            "text":
                latest["text"]
        }

    # =====================================================
    # LATEST EMAIL
    # =====================================================

    def latest_email(self, emails):

        events = []

        for e in emails:

            dt = self.parse_date(
                e.get("created_at")
            )

            if dt:

                events.append({

                    "date":
                        dt,

                    "email":
                        e
                })

        if not events:
            return None

        latest = max(
            events,
            key=lambda x: x["date"]
        )

        return latest["email"]

    # =====================================================
    # LATEST NOTE
    # =====================================================

    def latest_note(self, notes):

        events = []

        for n in notes:

            dt = self.parse_date(
                n.get("updated_at")
            )

            if dt:

                events.append({

                    "date":
                        dt,

                    "note":
                        n
                })

        if not events:
            return None

        latest = max(
            events,
            key=lambda x: x["date"]
        )

        return latest["note"]

    # =====================================================
    # LATEST ACTIVITY
    # =====================================================

    def latest_activity(self, activities):

        events = []

        for a in activities:

            dt = self.parse_date(
                a.get("updated_at")
            )

            if dt:

                events.append({

                    "date":
                        dt,

                    "activity":
                        a
                })

        if not events:
            return None

        latest = max(
            events,
            key=lambda x: x["date"]
        )

        return latest["activity"]

    # =====================================================
    # MASTER
    # =====================================================
def fetch_history(
    self,
    deal_id,
    latest_call_date=None,
    latest_email_date=None,
    latest_note_date=None,
    latest_activity_date=None
):

    # Fetch full history
    activities = self.get_activities(deal_id)

    notes = self.get_notes(deal_id)

    emails = self.get_emails(
        deal_id,
        latest_email_date
    )

    # Compute latest values BEFORE filtering
    latest_call = self.latest_call(activities)

    latest_note = self.latest_note(notes)

    latest_activity = self.latest_activity(activities)

    latest_email = self.latest_email(emails)

    # Filter only what will be returned
    filtered_activities = self.filter_after(
        activities,
        "updated_at",
        latest_activity_date
    )

    filtered_notes = self.filter_after(
        notes,
        "updated_at",
        latest_note_date
    )

    return {

        "deal_id": deal_id,

        "latest_call": latest_call,

        "latest_email": latest_email,

        "latest_note": latest_note,

        "latest_activity": latest_activity,

        "activities": filtered_activities,

        "notes": filtered_notes,

        "emails": emails
    }
# =====================================================
# TEST
# =====================================================

DOMAIN = "tensitron"
API_TOKEN = os.getenv("PIPEDRIVE_TOKEN")

def run_delta(
    deal_id,
    latest_call_date,
    latest_email_date,
    latest_note_date,
    latest_activity_date
):

    fetcher = PipedriveDealHistoryFetcher(
        DOMAIN,
        API_TOKEN
    )

    history = fetcher.fetch_history(
        deal_id,
        latest_call_date,
        latest_email_date,
        latest_note_date,
        latest_activity_date
    )

    latest_dates = {

        "latest_call_date":
            history["latest_call"]["date"]
            if history["latest_call"] else None,

        "latest_call_id":
            history["latest_call"]["activity_id"]
            if history["latest_call"] else None,

        "latest_email_date":
            history["latest_email"]["created_at"]
            if history["latest_email"] else None,

        "latest_email_id":
            history["latest_email"]["id"]
            if history["latest_email"] else None,

        "latest_note_date":
            history["latest_note"]["updated_at"]
            if history["latest_note"] else None,

        "latest_note_id":
            history["latest_note"]["id"]
            if history["latest_note"] else None,

        "latest_activity_date":
            history["latest_activity"]["updated_at"]
            if history["latest_activity"] else None,

        "latest_activity_id":
            history["latest_activity"]["id"]
            if history["latest_activity"] else None
    }

    return {

        "deal_id":
            deal_id,

        "latest_dates":
            latest_dates,

        "history": {

            "activities":
                history["activities"],

            "notes":
                history["notes"],

            "emails":
                history["emails"]
        }
    }