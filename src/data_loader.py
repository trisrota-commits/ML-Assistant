import requests
from bs4 import BeautifulSoup


def fetch_clean_text(url: str) -> str:
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = "\n".join([p.get_text() for p in paragraphs])
    return text


def clean_text(text: str) -> str:
    lines = text.split("\n")
    cleaned = []

    for l in lines:
        l = l.strip()

        if len(l) < 40:
            continue

        if any(keyword in l.lower() for keyword in [
            "thank", "thanks", "acknowledgment",
            "grateful", "colleagues", "correspondence"
        ]):
            continue

        cleaned.append(l)

    return "\n".join(cleaned)