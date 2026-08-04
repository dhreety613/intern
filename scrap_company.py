import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    )
}

PAGE_TYPES = {
    "about": [
        "about",
        "about us",
        "company",
        "who we are",
        "our story"
    ],

    "products": [
        "products",
        "product",
        "catalog",
        "solutions"
    ],

    "services": [
        "services",
        "service",
        "capabilities"
    ],

    "contact": [
        "contact",
        "contact us",
        "locations",
        "location",
        "find us"
    ]
}


EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
PHONE_REGEX = r"\+?\d[\d\s().-]{7,}\d"


def clean_html(html):

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "iframe",
        "header",
        "footer",
        "nav",
        "aside"
    ]):
        tag.decompose()

    return " ".join(
        soup.get_text(" ", strip=True).split()
    )


def fetch(session, url):

    try:

        r = session.get(
            url,
            timeout=10,
            allow_redirects=True
        )

        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "html.parser")

        title = ""
        if soup.title:
            title = soup.title.get_text(strip=True)

        meta = ""

        tag = soup.find(
            "meta",
            attrs={"name": "description"}
        )

        if tag:
            meta = tag.get("content", "")

        return {
            "url": r.url,
            "html": r.text,
            "title": title,
            "meta": meta,
            "text": clean_html(r.text)
        }

    except Exception:
        return None


def discover_links(base_url, html):

    soup = BeautifulSoup(html, "html.parser")

    discovered = {}

    for a in soup.find_all("a", href=True):

        text = a.get_text(" ", strip=True).lower()
        href = a["href"]

        url = urljoin(base_url, href)

        for page_type, keywords in PAGE_TYPES.items():

            if page_type in discovered:
                continue

            if any(keyword in text for keyword in keywords):
                discovered[page_type] = url

    return discovered


def scrape_website(website):

    if not website.startswith("http"):
        website = "https://" + website

    session = requests.Session()
    session.headers.update(HEADERS)

    homepage = fetch(session, website)

    if homepage is None:

        return {
            "website": website,
            "error": "Unable to access website."
        }

    links = discover_links(
        homepage["url"],
        homepage["html"]
    )

    result = {
        "website": homepage["url"],

        "homepage": {
            "title": homepage["title"],
            "meta_description": homepage["meta"],
            "text": homepage["text"]
        }
    }

    for section in [
        "about",
        "products",
        "services",
        "contact"
    ]:

        if section not in links:
            continue

        page = fetch(session, links[section])

        if page is None:
            continue

        result[section] = {
            "url": page["url"],
            "text": page["text"]
        }

        if section == "contact":

            result[section]["emails"] = list(
                set(
                    re.findall(
                        EMAIL_REGEX,
                        page["text"]
                    )
                )
            )

            result[section]["phones"] = list(
                set(
                    re.findall(
                        PHONE_REGEX,
                        page["text"]
                    )
                )
            )

    return result