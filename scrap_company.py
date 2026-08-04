from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests

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
    "contact"
]


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


def scrape_page(session: requests.Session, url: str):

    try:
        response = session.get(
            url,
            timeout=10,
            allow_redirects=True
        )

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.get_text(strip=True) if soup.title else ""

        meta_description = ""
        meta = soup.find("meta", attrs={"name": "description"})
        if meta:
            meta_description = meta.get("content", "")

        return {
            "title": title,
            "meta_description": meta_description,
            "content": clean_html(response.text)
        }

    except Exception:
        return None


def scrape_website(website: str):

    website = website.strip()

    if not website.startswith("http"):
        website = "https://" + website

    session = requests.Session()
    session.headers.update(HEADERS)

    title = ""
    meta_description = ""

    pages_scraped = []
    content = []

    for page in PAGES:

        url = urljoin(website.rstrip("/") + "/", page)

        result = scrape_page(session, url)

        if result is None:
            continue

        pages_scraped.append(url)

        if not title:
            title = result["title"]

        if not meta_description:
            meta_description = result["meta_description"]

        content.append(result["content"])

    return {
        "website": website,
        "title": title,
        "meta_description": meta_description,
        "pages_scraped": pages_scraped,
        "content": "\n\n".join(content)[:30000]
    }