from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests

app = FastAPI(title="Company Website Scraper")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    )
}

PAGES = [
    "",
    "about",
    "about-us",
    "company",
    "products",
    "services",
    "contact",
]


class ScrapeRequest(BaseModel):
    website: str


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "iframe"
    ]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    return " ".join(text.split())


def scrape_page(url: str):

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

        meta_description = ""

        meta = soup.find(
            "meta",
            attrs={"name": "description"}
        )

        if meta:
            meta_description = meta.get("content", "")

        content = clean_html(response.text)

        return {
            "title": title,
            "meta_description": meta_description,
            "content": content
        }

    except Exception:
        return None


@app.post("/scrape")
def scrape(request: ScrapeRequest):

    website = request.website.strip()

    if not website.startswith("http"):
        website = "https://" + website

    scraped_pages = []

    title = ""
    meta_description = ""

    collected_text = []

    for page in PAGES:

        url = urljoin(website.rstrip("/") + "/", page)

        result = scrape_page(url)

        if result is None:
            continue

        scraped_pages.append(url)

        if not title:
            title = result["title"]

        if not meta_description:
            meta_description = result["meta_description"]

        collected_text.append(result["content"])

    if not collected_text:
        raise HTTPException(
            status_code=404,
            detail="Unable to scrape website."
        )

    return {
        "website": website,
        "title": title,
        "meta_description": meta_description,
        "pages_scraped": scraped_pages,
        "content": "\n\n".join(collected_text)[:30000]
    }