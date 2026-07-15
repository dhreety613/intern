import requests
import json
from bs4 import BeautifulSoup
from html import unescape
from datetime import datetime, timezone, timedelta
import os

from fastapi import FastAPI
from pydantic import BaseModel

# Pipedrive account timezone. add_time/update_time/message_time from the API
# are naive strings in the ACCOUNT's local timezone, not UTC. Philippines is
# UTC+8 — change this if your Pipedrive account's timezone differs
# (Company Settings -> General -> Timezone).
PIPEDRIVE_TZ = timezone(timedelta(hours=8))


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
    # Always returns a timezone-AWARE datetime in UTC, or None.
    # - Strings with an explicit offset/Z (anything this service itself
    #   emits, or that gets round-tripped back through n8n/JS) are trusted
    #   as-is and converted to UTC.
    # - Naive strings (Pipedrive's raw add_time/update_time/message_time)
    #   are assumed to be in PIPEDRIVE_TZ before converting to UTC.
    # This is what stops the silent multi-hour drift that was making old
    # items look "new" again on every run.
    # =====================================================

    @staticmethod
    def parse_date(value):

        if not value:
            return None

        value = value.strip()
        dt = None

        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            pass

        if dt is None:
            try:
                dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

        if dt is None:
            print(f"WARNING: could not parse date value: {value!r}")
            return None

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=PIPEDRIVE_TZ)

        return dt.astimezone(timezone.utc)

    @staticmethod
    def to_utc_iso(dt):
        if dt is None:
            return None
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def filter_after(items, field_name, cutoff):

        cutoff_dt = PipedriveDealHistoryFetcher.parse_date(cutoff)

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
    # ACTIVITIES  (raw fetch, unfiltered — kept as its own step)
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
    # CALLS  (extracted as its own step, not sifted out later —
    # this is what was missing before: calls need their own cutoff,
    # not the generic latest_activity_date one)
    # =====================================================

    def get_calls(self, activities):

        print("\nSTEP 1B - CALLS (from activities)")

        calls = [
            a for a in activities
            if str(a.get("type", "")).lower() == "call"
            or str(a.get("subject", "")).lower() == "call"
        ]

        print(
            f"Calls fetched: {len(calls)}"
        )

        return calls

    @staticmethod
    def non_call_activities(activities):
        return [
            a for a in activities
            if not (
                str(a.get("type", "")).lower() == "call"
                or str(a.get("subject", "")).lower() == "call"
            )
        ]

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

        new_headers = self.filter_after(
            headers,
            "message_time",
            latest_email_date
    )

        emails = []

        total = len(
            new_headers
    )

        print(
            f"\nSTEP 4 - FETCHING {total} NEW EMAILS"
    )

        for i, email in enumerate(
            new_headers,
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
    # LATEST-DATE HELPER  (shared by all four types)
    # Returns a normalized UTC ISO string + matching id. If items is
    # empty, re-normalizes and returns the fallback cutoff unchanged
    # (no id), so persisting this doesn't move the cutoff backwards
    # and doesn't drift when it's fed back in on the next run.
    # =====================================================

    @classmethod
    def latest_of(cls, items, field_name, fallback, id_field="id"):

        dated = []

        for item in items:
            dt = cls.parse_date(item.get(field_name))
            if dt:
                dated.append((dt, item))

        if not dated:
            return cls.to_utc_iso(cls.parse_date(fallback)) or fallback, None

        dt, item = max(dated, key=lambda x: x[0])
        return cls.to_utc_iso(dt), item.get(id_field)

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

        # Individual extraction steps, kept separate on purpose.
        all_activities = self.get_activities(deal_id)
        calls = self.get_calls(all_activities)
        other_activities = self.non_call_activities(all_activities)

        notes = self.get_notes(deal_id)
        emails = self.get_emails(deal_id, latest_email_date)

        # Each type filtered by its OWN cutoff — this is the actual fix.
        # Before, calls were never split out, so latest_call_date was
        # silently ignored and calls kept riding along inside "activities".
        new_calls = self.filter_after(calls, "updated_at", latest_call_date)
        new_activities = self.filter_after(other_activities, "updated_at", latest_activity_date)
        new_notes = self.filter_after(notes, "updated_at", latest_note_date)
        # emails are already filtered inside get_emails()

        call_date, call_id = self.latest_of(new_calls, "updated_at", latest_call_date)
        activity_date, activity_id = self.latest_of(new_activities, "updated_at", latest_activity_date)
        note_date, note_id = self.latest_of(new_notes, "updated_at", latest_note_date)
        email_date, email_id = self.latest_of(emails, "created_at", latest_email_date)

        return {

            "deal_id": deal_id,

            "latest_dates": {
                "latest_call_date": call_date,
                "latest_call_id": call_id,
                "latest_activity_date": activity_date,
                "latest_activity_id": activity_id,
                "latest_note_date": note_date,
                "latest_note_id": note_id,
                "latest_email_date": email_date,
                "latest_email_id": email_id,
            },

            "calls": new_calls,

            "activities": new_activities,

            "notes": new_notes,

            "emails": emails
        }


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI()

DOMAIN = "tensitron"
API_TOKEN = os.getenv("PIPEDRIVE_TOKEN")


class DealRequest(BaseModel):
    deal_id: int

    latest_call_date: str | None = None
    latest_email_date: str | None = None
    latest_note_date: str | None = None
    latest_activity_date: str | None = None


@app.post("/deal-history")
def deal_history(request: DealRequest):

    fetcher = PipedriveDealHistoryFetcher(
        DOMAIN,
        API_TOKEN
    )

    history = fetcher.fetch_history(
        request.deal_id,
        latest_call_date=request.latest_call_date,
        latest_email_date=request.latest_email_date,
        latest_note_date=request.latest_note_date,
        latest_activity_date=request.latest_activity_date
    )

    return {

        "deal_id":
            request.deal_id,

        "latest_dates":
            history["latest_dates"],

        "history": {

            "calls":
                history["calls"],

            "activities":
                history["activities"],

            "notes":
                history["notes"],

            "emails":
                history["emails"]
        }
    }