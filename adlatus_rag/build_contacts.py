# build_contacts.py
import json, re, pathlib
from bs4 import BeautifulSoup

IN  = "data/raw/adlatus_pages.jsonl"
OUT = "data/processed/contacts.json"

def clean_text(s: str) -> str:
    if not s:
        return None
    return re.sub(r"\s+", " ", s.strip())

def parse_contacts_from_html(html: str, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    entries = []
    for card in soup.select(".wpbdp-listing"):
        # Name
        name_el = card.select_one(".listing-title h3")
        name = clean_text(name_el.get_text()) if name_el else None

        # Helpers
        def field_value(cls):
            el = card.select_one(cls)
            if not el:
                return None
            return clean_text(el.get_text(" ", strip=True))

        competencies_raw = field_value(".wpbdp-field-kernkompetenzen")
        email_raw        = field_value(".wpbdp-field-kontakt_email")
        phone_raw        = field_value(".wpbdp-field-telefon")
        location_raw     = field_value(".wpbdp-field-ort") or field_value(".wpbdp-field-standort")

        # Normalize
        def strip_label(val, labels):
            if not val: return None
            v = val
            for lab in labels:
                v = re.sub(rf"^{lab}\s*[:\-]?\s*", "", v, flags=re.I)
            return v.strip()

        email    = strip_label(email_raw, ["Kontakt eMail", "Kontakt Email", "Email", "E-Mail"])
        phone    = strip_label(phone_raw, ["Telefon", "Tel.", "Mobile", "Mobil"])
        location = strip_label(location_raw, ["Ort", "Basis", "Standort"])

        comps = []
        if competencies_raw:
            parts = re.split(r"[;,•]\s*", competencies_raw)
            for c in parts:
                c = re.sub(r"^(Kernkompetenzen)\s*", "", c, flags=re.I).strip()
                if c and c.lower() != "kernkompetenzen":
                    comps.append(c)

        # Profile link
        link_el = card.select_one(".listing-thumbnail a[href]")
        url = link_el["href"] if link_el and link_el.has_attr("href") else base_url

        entries.append({
            "name": name,
            "email": email,
            "phone": phone,
            "location": location,
            "competencies": comps,
            "profile_url": url,
        })
    return entries

def main():
    with open(IN, "r", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]

    all_contacts = []
    for rec in recs:
        url = rec["url"]
        html = rec["html"]
        contacts = parse_contacts_from_html(html, url)
        all_contacts.extend(contacts)

    pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(all_contacts, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(all_contacts)} contacts -> {OUT}")

if __name__ == "__main__":
    main()
