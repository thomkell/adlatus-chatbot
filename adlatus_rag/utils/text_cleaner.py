import re
from bs4 import BeautifulSoup

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return normalize_ws(text)

_WS_RE = re.compile(r"\s+")
def normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", (text or "")).strip()

def make_title(s: str) -> str:
    s = normalize_ws(s)
    return s[:140]
