import json
import os
from playwright.sync_api import sync_playwright

# Add all relevant Adlatus pages you want to scrape
URLS = [
    "https://adlatus-zh.ch/",
    "https://adlatus-zh.ch/mentoring/",
    "https://adlatus-zh.ch/angebot/",
    "https://adlatus-zh.ch/organisation/",
    "https://adlatus-zh.ch/kontakt/",
    # Add more pages as needed...
]

os.makedirs("../data", exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    result = []

    for url in URLS:
        print(f"Scraping {url}")
        page.goto(url)
        page.wait_for_timeout(1000)  # Wait for 1 second

        text = page.content()
        result.append({
            "url": url,
            "html": text
        })

    with open("../data/adlatus_pages.jsonl", "w", encoding="utf-8") as f:
        for entry in result:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("✅ Scraping complete.")
    browser.close()
