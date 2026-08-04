"""
Company Research Agent
=======================
Input : ExtractedInfo (from Extract Agent — needs company_name, claimed_company_domain)
Output: CompanyProfile

Two-step approach:
  1. Deterministic pre-fetch: pull the company's own homepage text and scan it
     for a phone number / contact email. Cheap, fast, grounds the LLM in real
     page content instead of relying purely on search.
  2. Gemini + Google Search grounding: fills in everything the homepage alone
     can't answer confidently — LinkedIn URL, other office locations, employee
     count, industry classification, products/services, founded year,
     estimated revenue bucket.

This agent assumes check_company_legitimacy() has already run and the domain
is not FAKE — it doesn't re-verify legitimacy, it assumes the company is real
and just goes deeper on who they are.
"""
import json
import logging
import re
from typing import Optional, List

import requests

from dataclasses import asdict
import json
from pathlib import Path
import re

from pipeline_types import ExtractedInfo, CompanyProfile
from llm_utils import call_gemini_with_retry, robust_json_parse, DEFAULT_MODEL

logger = logging.getLogger("company_research_agent")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

MODEL = DEFAULT_MODEL
RESEARCH_MAX_OUTPUT_TOKENS = 2000  # generous — grounded calls can involve several searches before the final answer
HTTP_TIMEOUT = 6.0
HOMEPAGE_TEXT_LIMIT = 1200  # chars of homepage text to feed the LLM as grounding context


# =============================================================================
# Deterministic pre-fetch
# =============================================================================

TAG_STRIP_PATTERN = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")

PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

GENERIC_EMAIL_PREFIXES = {"noreply", "no-reply", "donotreply", "webmaster", "postmaster"}


def fetch_homepage_text(domain: str) -> Optional[str]:
    """Best-effort fetch + crude HTML-to-text. Returns None if unreachable."""
    for scheme in ("https://", "http://"):
        try:
            resp = requests.get(f"{scheme}{domain}", timeout=HTTP_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code >= 400:
                continue
            text = TAG_STRIP_PATTERN.sub(" ", resp.text)
            text = HTML_TAG_PATTERN.sub(" ", text)
            text = WHITESPACE_PATTERN.sub(" ", text).strip()
            return text[:HOMEPAGE_TEXT_LIMIT] if text else None
        except requests.RequestException as e:
            logger.info(f"Could not fetch {scheme}{domain}: {e}")
            continue
    return None


def extract_phone(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    match = PHONE_PATTERN.search(text)
    return match.group(0).strip() if match else None


def extract_contact_email(text: Optional[str], domain: Optional[str] = None) -> Optional[str]:
    """Prefers an email on the company's own domain, skips generic noreply-style addresses."""
    if not text:
        return None
    candidates = EMAIL_PATTERN.findall(text)
    if not candidates:
        return None

    def is_generic(addr: str) -> bool:
        return addr.split("@")[0].lower() in GENERIC_EMAIL_PREFIXES

    non_generic = [c for c in candidates if not is_generic(c)]
    pool = non_generic or candidates

    if domain:
        on_domain = [c for c in pool if c.lower().endswith(f"@{domain.lower()}")]
        if on_domain:
            return on_domain[0]

    return pool[0]


# =============================================================================
# LLM research step
# =============================================================================

RESEARCH_SYSTEM_PROMPT = """You are a company research analyst. Given a company name, domain,
and (optionally) some raw text scraped from their homepage, research the company using Google
Search and produce a concise, factual profile. Prioritize the company's own website and
LinkedIn page as sources; avoid low-quality aggregator sites.

Return ONLY a JSON object, no markdown fences, no preamble, with this exact shape:
{
  "official_name": string or null,
  "website": string or null,
  "linkedin": string or null,
  "phone": string or null,
  "email": string or null,
  "headquarters": string or null,
  "locations": [string, ...],
  "industry": string or null,
  "field_of_work": string or null,
  "description": string or null,
  "products": [string, ...],
  "services": [string, ...],
  "employee_count": integer or null,
  "estimated_revenue": string or null,
  "founded_year": integer or null
}

Rules:
- If you cannot find a field with reasonable confidence, use null (or an empty list for
  locations/products/services) rather than guessing.
- Do not invent a LinkedIn URL, phone number, or revenue figure — only report what search
  results actually support.
- description should read like a factual summary, not marketing copy.
- estimated_revenue should be a bucket range (e.g. "$10M-$50M"), not a fabricated precise number.
"""


def build_research_prompt(extracted: ExtractedInfo, homepage_text: Optional[str]) -> str:
    parts = [
        f"COMPANY NAME (as given): {extracted.company_name}",
        f"DOMAIN: {extracted.claimed_company_domain}",
    ]
    if homepage_text:
        parts.append(f"\nHOMEPAGE TEXT (scraped, for grounding — verify/expand via search):\n{homepage_text}")
    else:
        parts.append("\nHOMEPAGE TEXT: unavailable (site could not be fetched) — rely on web search.")
    return "\n".join(parts)


def research_company(extracted: ExtractedInfo) -> CompanyProfile:
    if not extracted.claimed_company_domain and not extracted.company_name:
        logger.warning("No domain or company name to research — returning empty profile.")
        return CompanyProfile()

    domain = extracted.claimed_company_domain
    homepage_text = fetch_homepage_text(domain) if domain else None

    fallback_phone = extract_phone(homepage_text)
    fallback_email = extract_contact_email(homepage_text, domain)

    user_content = build_research_prompt(extracted, homepage_text)

    logger.info(f"Researching company: {extracted.company_name!r} ({domain})")
    raw_text = call_gemini_with_retry(
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        user_content=user_content,
        model=MODEL,
        max_output_tokens=RESEARCH_MAX_OUTPUT_TOKENS,
        use_google_search=False,
    )

    try:
        parsed = robust_json_parse(raw_text)
    except ValueError as e:
        logger.error(f"Research Agent failed to parse LLM output, returning partial profile: {e}")
        return CompanyProfile(
            official_name=extracted.company_name,
            website=f"https://{domain}" if domain else None,
            phone=fallback_phone,
            email=fallback_email,
        )

    profile = CompanyProfile(
        official_name=parsed.get("official_name") or extracted.company_name,
        website=parsed.get("website") or (f"https://{domain}" if domain else None),
        linkedin=parsed.get("linkedin"),
        phone=parsed.get("phone") or fallback_phone,
        email=parsed.get("email") or fallback_email,
        headquarters=parsed.get("headquarters"),
        locations=_coerce_str_list(parsed.get("locations")),
        industry=parsed.get("industry"),
        field_of_work=parsed.get("field_of_work"),
        description=parsed.get("description"),
        products=_coerce_str_list(parsed.get("products")),
        services=_coerce_str_list(parsed.get("services")),
        employee_count=_coerce_int(parsed.get("employee_count")),
        estimated_revenue=parsed.get("estimated_revenue"),
        founded_year=_coerce_int(parsed.get("founded_year")),
    )
    logger.info(
        f"Research complete: industry={profile.industry!r}, "
        f"employees={profile.employee_count}, locations={len(profile.locations)}"
    )
    return profile


# =============================================================================
# Small coercion helpers
# =============================================================================

def _coerce_str_list(value) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def _coerce_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        digits = re.sub(r"[^\d]", "", value)
        return int(digits) if digits else None
    return None


# =============================================================================
# Demo / manual test
# =============================================================================

if __name__ == "__main__":
    from pipeline_types import ExtractedInfo

    sample = ExtractedInfo(
        agent_name="Sarah Klein",
        agent_email="sarah.klein@acme-industrial.com",
        agent_email_domain="acme-industrial.com",
        company_name="Acme Industrial Inc",
        claimed_company_domain="acme-industrial.com",
        website_from_signature="www.acme-industrial.com",
        ktc_branch="tensitron",
        is_active_branch=True,
        extraction_confidence=1.0,
    )
    profile = research_company(sample)

output_dir = Path("outputs/research")
output_dir.mkdir(parents=True, exist_ok=True)

safe_name = re.sub(r"[^A-Za-z0-9]+", "_", profile.official_name or sample.company_name).strip("_")

output_file = output_dir / f"{safe_name}.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(asdict(profile), f, indent=4, ensure_ascii=False)

print(f"Saved to {output_file}")