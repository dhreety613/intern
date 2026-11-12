"""Twilio Lookup v2 pre-dial check. Fails open on API errors (never block a call on Lookup downtime)."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from .config import Settings


@dataclass
class LookupResult:
    valid: bool
    e164: str | None = None
    line_type: str | None = None     # mobile / landline / fixedVoip / nonFixedVoip / tollFree ...
    reason: str | None = None
    checked: bool = True             # False => Lookup unavailable, we failed open


async def lookup_number(number: str, s: Settings, client: httpx.AsyncClient | None = None) -> LookupResult:
    if not s.lookup_enabled or not s.account_sid:
        return LookupResult(valid=True, e164=number, checked=False)
    url = f"https://lookups.twilio.com/v2/PhoneNumbers/{number}"
    own = client is None
    client = client or httpx.AsyncClient(timeout=5)
    try:
        r = await client.get(url, params={"Fields": "line_type_intelligence"},
                             auth=(s.account_sid, s.auth_token))
        if r.status_code == 404:
            return LookupResult(valid=False, reason="not_found")
        if r.status_code >= 400:
            return LookupResult(valid=True, e164=number, checked=False, reason=f"http_{r.status_code}")
        d = r.json()
        lti = d.get("line_type_intelligence") or {}
        if not d.get("valid", False):
            return LookupResult(valid=False, reason=",".join(d.get("validation_errors") or []) or "invalid")
        return LookupResult(valid=True, e164=d.get("phone_number", number), line_type=lti.get("type"))
    except httpx.HTTPError as e:
        return LookupResult(valid=True, e164=number, checked=False, reason=f"lookup_error:{type(e).__name__}")
    finally:
        if own:
            await client.aclose()
