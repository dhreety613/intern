import requests
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

        r = requests.get(f"{self.base_url}{endpoint}", params=params, timeout=60)
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
        return BeautifulSoup(text, "html.parser").get_text(separator="\n", strip=True)

    # =====================================================
    # DATE PARSER
    # =====================================================

    @staticmethod
    def parse_date(value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", ""))
        except Exception:
            pass
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
        print(f"WARNING: could not parse date value: {value!r}")
        return None

    @classmethod
    def filter_after(cls, items, field_name, cutoff):
        """Return only items whose [field_name] date is strictly after cutoff.
        No cutoff -> nothing is filtered out."""
        cutoff_dt = cls.parse_date(cutoff)
        if not cutoff_dt:
            return items

        result = []
        for item in items:
            item_dt = cls.parse_date(item.get(field_name))
            if item_dt and item_dt > cutoff_dt:
                result.append(item)
        return result

    # =====================================================
    # ACTIVITIES (split into calls vs. everything else)
    # =====================================================

    def get_activities(self, deal_id):
        data = self._get(f"/deals/{deal_id}/activities")

        calls, other = [], []
        for a in (data.get("data") or []):
            record = {
                "id": a.get("id"),
                "type": a.get("type"),
                "subject": a.get("subject"),
                "text": self.clean_html(a.get("note")),
                "created_at": a.get("add_time"),
                "updated_at": a.get("update_time"),
            }
            is_call = str(a.get("type", "")).lower() == "call" or str(a.get("subject", "")).lower() == "call"
            (calls if is_call else other).append(record)

        return calls, other

    # =====================================================
    # NOTES
    # =====================================================

    def get_notes(self, deal_id):
        data = self._get(f"/deals/{deal_id}/notes")

        return [
            {
                "id": n.get("id"),
                "text": self.clean_html(n.get("content")),
                "created_at": n.get("add_time"),
                "updated_at": n.get("update_time"),
            }
            for n in (data.get("data") or [])
        ]

    # =====================================================
    # EMAILS
    # =====================================================

    def get_email_messages(self, deal_id):
        data = self._get(f"/deals/{deal_id}/mailMessages")
        return [item.get("data", {}) for item in (data.get("data") or []) if item.get("object") == "mailMessage"]

    def get_email_details(self, message_id):
        try:
            data = self._get(f"/mailbox/mailMessages/{message_id}")
            return data.get("data", {})
        except Exception as e:
            print(f"FAILED EMAIL DETAIL {message_id}: {e}")
            return {}

    def fetch_email_body(self, body_url):
        if not body_url:
            return ""
        try:
            r = requests.get(body_url, timeout=60)
            return self.clean_html(r.text) if r.status_code == 200 else ""
        except Exception as e:
            print(f"BODY DOWNLOAD FAILED: {e}")
            return ""

    def get_emails(self, deal_id, latest_email_date=None):
        headers = self.get_email_messages(deal_id)

        # only pull full details/bodies for headers already past the cutoff,
        # so we don't waste calls on emails we're going to throw away anyway
        new_headers = self.filter_after(headers, "message_time", latest_email_date)

        emails = []
        for email in new_headers:
            details = self.get_email_details(email.get("id"))

            body = ""
            if details.get("has_body_flag"):
                body = self.fetch_email_body(details.get("body_url"))

            emails.append({
                "id": details.get("id"),
                "thread_id": details.get("mail_thread_id"),
                "subject": details.get("subject"),
                "snippet": details.get("snippet"),
                "body": body,
                "from": details.get("from", []),
                "to": details.get("to", []),
                "created_at": details.get("message_time"),
            })

        return emails

    # =====================================================
    # LATEST DATE HELPER
    # =====================================================

    @classmethod
    def latest_date(cls, items, field_name, fallback, id_field="id"):
        """Max [field_name] date among items, as (date_str, id). If items is
        empty, returns the fallback cutoff unchanged (and no id) so the next
        run's cutoff doesn't move when nothing new came in."""
        dated = []
        for item in items:
            dt = cls.parse_date(item.get(field_name))
            if dt:
                dated.append((dt, item))

        if not dated:
            return fallback, None

        dt, item = max(dated, key=lambda x: x[0])
        return item.get(field_name), item.get(id_field)

    # =====================================================
    # MASTER — returns ONLY items after the given cutoff dates,
    # plus updated cutoffs to persist for the next run
    # =====================================================

    def fetch_new_history(
        self,
        deal_id,
        latest_call_date=None,
        latest_email_date=None,
        latest_note_date=None,
        latest_activity_date=None,
    ):
        calls, other_activities = self.get_activities(deal_id)
        notes = self.get_notes(deal_id)

        new_calls = self.filter_after(calls, "updated_at", latest_call_date)
        new_activities = self.filter_after(other_activities, "updated_at", latest_activity_date)
        new_notes = self.filter_after(notes, "updated_at", latest_note_date)
        new_emails = self.get_emails(deal_id, latest_email_date)

        call_date, call_id = self.latest_date(new_calls, "updated_at", latest_call_date)
        activity_date, activity_id = self.latest_date(new_activities, "updated_at", latest_activity_date)
        note_date, note_id = self.latest_date(new_notes, "updated_at", latest_note_date)
        email_date, email_id = self.latest_date(new_emails, "created_at", latest_email_date)

        return {
            "deal_id": deal_id,
            "calls": new_calls,
            "activities": new_activities,
            "notes": new_notes,
            "emails": new_emails,
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
        }


# =====================================================
# ENTRY POINT — import this into main.py
# =====================================================

DOMAIN = "tensitron"
API_TOKEN = os.getenv("PIPEDRIVE_TOKEN")


def run_delta(
    deal_id,
    latest_call_date=None,
    latest_email_date=None,
    latest_note_date=None,
    latest_activity_date=None,
):
    """Fetch only what's new since the given cutoff dates for a deal.
    Returns the new items plus updated cutoffs to persist for next time."""

    fetcher = PipedriveDealHistoryFetcher(DOMAIN, API_TOKEN)

    result = fetcher.fetch_new_history(
        deal_id,
        latest_call_date=latest_call_date,
        latest_email_date=latest_email_date,
        latest_note_date=latest_note_date,
        latest_activity_date=latest_activity_date,
    )

    return {
        "deal_id": deal_id,
        "latest_dates": result["latest_dates"],
        "history": {
            "calls": result["calls"],
            "activities": result["activities"],
            "notes": result["notes"],
            "emails": result["emails"],
        },
    }


if __name__ == "__main__":
    # quick local test
    output = run_delta(
        deal_id=123,  # replace with your deal id
        latest_call_date="2025-06-01 00:00:00",
        latest_email_date="2025-06-01 00:00:00",
        latest_note_date="2025-06-01 00:00:00",
        latest_activity_date="2025-06-01 00:00:00",
    )
    print(output)