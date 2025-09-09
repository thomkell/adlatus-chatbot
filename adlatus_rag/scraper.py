# scraper.py
import os, json, time
from playwright.sync_api import sync_playwright

URL = "https://www.adlatus-zh.ch/kompetenzen/unser-team/"
OUT = "data/raw/adlatus_pages.jsonl"

os.makedirs("data/raw", exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    print(f"Scraping {URL}")
    page.goto(URL)
    page.wait_for_load_state("domcontentloaded")

    # try to click all "Mehr anzeigen" buttons if present
    buttons = page.locator("text=Mehr anzeigen")
    count = buttons.count()
    if count > 0:
        for i in range(count):
            try:
                btn = buttons.nth(i)
                btn.click()
                time.sleep(0.5)
            except Exception as e:
                print("Could not click expand button:", e)

    # capture final HTML and title
    html = page.content()
    title = page.title()

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(json.dumps({"url": URL, "title": title, "html": html}) + "\n")

    browser.close()

print(f"Saved Unser Team page -> {OUT}")
    