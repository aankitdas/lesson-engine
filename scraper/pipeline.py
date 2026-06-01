import httpx
import json
import asyncio
import aiofiles
from pathlib import Path
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import trafilatura
from langdetect import detect, LangDetectException

from scraper.sources import SOURCES, CURATED_SEEDS, CURATED_ONLY
from scraper.cleaner import clean_text, word_count, is_usable

RAW_DIR = Path("scraper/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


async def fetch(client: httpx.AsyncClient, url: str) -> str | None:
    try:
        r = await client.get(url, headers=HEADERS, timeout=20, follow_redirects=True)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  [SKIP] {url} — {type(e).__name__}: {repr(e)}")
        return None


# ── Gutenberg handler ──────────────────────────────────────────────────────

async def fetch_gutenberg(
    client: httpx.AsyncClient,
    api_url: str,
    lesson_type: str,
    age_group_hint: str,
    max_books: int = 3,
    chunks_per_book: int = 15,
) -> list[dict]:
    print(f"  Querying Gutenberg: {api_url}")
    try:
        r = await client.get(api_url, timeout=90, follow_redirects=True)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [SKIP] Gutenberg API — {repr(e)}")
        return []

    books = data.get("results", [])[:max_books]
    items = []

    for book in books:
        title = book.get("title", "unknown")
        formats = book.get("formats", {})
        txt_url = (
            formats.get("text/plain; charset=utf-8")
            or formats.get("text/plain; charset=us-ascii")
            or formats.get("text/plain")
        )
        if not txt_url:
            continue

        print(f"  Fetching '{title}'")
        try:
            r = await client.get(txt_url, timeout=60, follow_redirects=True)
            r.raise_for_status()
            raw_text = r.text
        except Exception as e:
            print(f"  [SKIP] book text {txt_url} — {repr(e)}")
            continue
        if not raw_text:
            continue

        # strip Gutenberg header/footer boilerplate
        start = raw_text.find("*** START OF")
        end = raw_text.find("*** END OF")
        if start != -1:
            raw_text = raw_text[start + 50:]
        if end != -1:
            raw_text = raw_text[:end]

        raw_text = raw_text.replace("\r\n", "\n")
        paragraphs = [p.strip() for p in raw_text.split("\n\n")]
        count = 0
        for para in paragraphs:
            if count >= chunks_per_book:
                break
            c = clean_text(para)
            if is_usable(c) and is_english(c):
                items.append({
                    "lesson_type":      lesson_type,
                    "age_group_hint":   age_group_hint,
                    "source_url":       txt_url,
                    "source_notes":     f"gutenberg: {title}",
                    "raw_text":         c,
                    "word_count":       word_count(c),
                    "scraped_at":       datetime.now(timezone.utc).isoformat(),
                    "labeled":          False,
                    "source_type":      "scraped",
                })
                count += 1

        await asyncio.sleep(1)

    print(f"  → {len(items)} chunks from Gutenberg")
    return items


# ── Generic HTML scraper (fallback) ───────────────────────────────────────

def extract_text(html: str) -> list[str]:
    results = []
    extracted = trafilatura.extract(html, include_links=False, include_images=False)
    if extracted:
        for chunk in extracted.split("\n\n"):
            c = clean_text(chunk)
            if is_usable(c) and is_english(c):
                results.append(c)
    if not results:
        soup = BeautifulSoup(html, "lxml")
        for p in soup.find_all("p"):
            c = clean_text(p.get_text())
            if is_usable(c) and is_english(c):
                results.append(c)
    return results


async def scrape_html(
    client: httpx.AsyncClient,
    url: str,
    lesson_type: str,
    age_group_hint: str,
    notes: str,
) -> list[dict]:
    print(f"  Fetching {url}")
    html = await fetch(client, url)
    if not html:
        return []
    chunks = extract_text(html)
    items = [{
        "lesson_type":      lesson_type,
        "age_group_hint":   age_group_hint,
        "source_url":       url,
        "source_notes":     notes,
        "raw_text":         chunk,
        "word_count":       word_count(chunk),
        "scraped_at":       datetime.now(timezone.utc).isoformat(),
        "labeled":          False,
        "source_type":      "scraped",
    } for chunk in chunks]
    print(f"  → {len(items)} chunks")
    return items


# ── Curated loader ─────────────────────────────────────────────────────────

def load_curated(lesson_type: str) -> list[dict]:
    seeds = CURATED_SEEDS.get(lesson_type, [])
    items = []
    for seed in seeds:
        text = seed.get("text") or seed.get("topic") or seed.get("prompt", "")
        base = {
            "lesson_type":      lesson_type,
            "age_group_hint":   seed.get("age_group_hint", "teen"),
            "source_url":       "curated",
            "source_notes":     "hand-curated seed",
            "raw_text":         text,
            "word_count":       word_count(text),
            "scraped_at":       datetime.now(timezone.utc).isoformat(),
            "labeled":          False,
            "source_type":      "curated",
        }
        extras = {k: v for k, v in seed.items()
                  if k not in ("text", "topic", "prompt", "age_group_hint")}
        items.append({**base, **extras})
    return items


# ── Main pipeline ──────────────────────────────────────────────────────────

async def run_scraper(lesson_types: list[str] | None = None):
    all_types = set(SOURCES.keys()) | CURATED_ONLY
    targets = lesson_types or list(all_types)

    # SSL verify=False only for sites with cert issues
    async with httpx.AsyncClient(verify=False) as client:
        for lt in targets:
            print(f"\n[{lt}]")
            all_items = []

            if lt in CURATED_ONLY:
                all_items = load_curated(lt)
                print(f"  Loaded {len(all_items)} curated seeds")
            else:
                for (url, age_hint, notes) in SOURCES.get(lt, []):
                    if notes.startswith("gutenberg:"):
                        items = await fetch_gutenberg(client, url, lt, age_hint)
                    else:
                        items = await scrape_html(client, url, lt, age_hint, notes)
                    all_items.extend(items)
                    await asyncio.sleep(1)

                curated = load_curated(lt)
                if curated:
                    all_items.extend(curated)
                    print(f"  + {len(curated)} curated seeds")

            out_path = RAW_DIR / f"{lt}.jsonl"
            async with aiofiles.open(out_path, "w", encoding="utf-8") as f:
                for item in all_items:
                    await f.write(json.dumps(item) + "\n")

            print(f"  ✓ {len(all_items)} total → {out_path}")


if __name__ == "__main__":
    asyncio.run(run_scraper())