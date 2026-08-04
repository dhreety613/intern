#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
    )
}

PAGES = [
    "",
    "about",
    "about-us",
    "company",
    "products",
    "services",
    "contact"
]


def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove unnecessary tags
    for tag in soup(["script", "style", "noscript", "svg", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


def scrape_page(url):

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=10,
            allow_redirects=True,
        )

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        title = ""
        if soup.title:
            title = soup.title.get_text(strip=True)

        meta = ""
        tag = soup.find("meta", attrs={"name": "description"})
        if tag:
            meta = tag.get("content", "")

        content = clean_text(response.text)

        return {
            "title": title,
            "meta": meta,
            "content": content
        }

    except Exception:
        return None


def scrape_website(base_url):

    pages_content = []

    title = ""
    meta = ""

    for page in PAGES:

        url = urljoin(base_url.rstrip("/") + "/", page)

        result = scrape_page(url)

        if not result:
            continue

        if not title:
            title = result["title"]

        if not meta:
            meta = result["meta"]

        pages_content.append(result["content"])

    return {
        "website": base_url,
        "title": title,
        "meta_description": meta,
        "content": "\n\n".join(pages_content)[:30000]
    }


if __name__ == "__main__":

    website = input("Website: ").strip()

    if not website.startswith("http"):
        website = "https://" + website

    result = scrape_website(website)

    import json
    print(json.dumps(result, indent=2))