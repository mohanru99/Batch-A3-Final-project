"""
Multi-source scrapers tested to work from cloud IPs (Railway, AWS, etc).

The key insight: most "review sites" block cloud IPs at the WAF layer.
What does NOT block: RSS feeds, Algolia/public search APIs, and Google News.
We use these to pull *real* opinion text about any company/product/topic.

Sources (all free, no API key needed):
  • Google News RSS         — news + opinion articles, never blocks RSS readers
  • DuckDuckGo News         — backup news source
  • Hacker News (Algolia)   — JSON API, public, fast
  • Reddit (old.reddit.com) — JSON API via the old.* host (less aggressive than www.*)
  • YouTube comments via   — Invidious public RSS instances (best-effort)
"""
import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import quote_plus, urlparse

import httpx
from bs4 import BeautifulSoup

log = logging.getLogger("scrapers")

TIMEOUT = 12

UA_DESKTOP = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
UA_FEED_READER = "Mozilla/5.0 (compatible; SentimentAnalyzer/2.0; +https://example.com/bot)"


def _clean(text: str) -> str:
    if not text:
        return ""
    text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_long_enough(text: str, min_len: int = 30) -> bool:
    return text and len(text) >= min_len


# ─────────────────────────────────────────────────────────────────────
# Google News RSS — primary news source. Works from any IP.
# ─────────────────────────────────────────────────────────────────────
async def scrape_google_news(query: str, limit: int = 25) -> list[dict]:
    """
    Google News RSS: returns recent news articles + opinion pieces about the query.
    Works from any IP. Returns title + description as the analyzable text.
    """
    out = []
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    )
    try:
        async with httpx.AsyncClient(
            timeout=TIMEOUT,
            headers={"User-Agent": UA_FEED_READER},
            follow_redirects=True,
        ) as c:
            r = await c.get(url)
            if r.status_code != 200:
                log.warning("google_news %d for %r", r.status_code, query)
                return out
            try:
                root = ET.fromstring(r.text)
            except ET.ParseError as e:
                log.warning("google_news parse: %s", e)
                return out
            for item in root.iter("item"):
                title = _clean(getattr(item.find("title"), "text", "") or "")
                desc = _clean(getattr(item.find("description"), "text", "") or "")
                link = getattr(item.find("link"), "text", "") or ""
                pub = getattr(item.find("pubDate"), "text", "") or ""
                source_el = item.find("source")
                source_name = _clean(getattr(source_el, "text", "") or "Google News")
                # Build analyzable text: title carries sentiment, description adds context
                text = title
                if desc and desc.lower() != title.lower():
                    text = f"{title}. {desc}"
                if not _is_long_enough(text, 25):
                    continue
                out.append({
                    "text": text[:2000],
                    "rating": None,
                    "author": source_name,
                    "source": f"news:{source_name}",
                    "url": link,
                    "score": 0,
                    "ts": pub,
                })
                if len(out) >= limit:
                    break
    except Exception as e:
        log.warning("scrape_google_news error: %s", e)
    log.info("google_news: %d items for %r", len(out), query)
    return out


# ─────────────────────────────────────────────────────────────────────
# Hacker News via Algolia — works everywhere
# ─────────────────────────────────────────────────────────────────────
async def scrape_hn(query: str, limit: int = 25) -> list[dict]:
    out = []
    url = "https://hn.algolia.com/api/v1/search"
    params = {"query": query, "tags": "(comment,story)", "hitsPerPage": min(limit, 50)}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, headers={"User-Agent": UA_DESKTOP}) as c:
            r = await c.get(url, params=params)
            if r.status_code != 200:
                log.warning("hn %d for %r", r.status_code, query)
                return out
            for hit in r.json().get("hits", []):
                text = hit.get("comment_text") or hit.get("story_text") or hit.get("title") or ""
                text = _clean(text)
                if not _is_long_enough(text, 30):
                    continue
                out.append({
                    "text": text[:2000],
                    "rating": None,
                    "author": hit.get("author", "anon"),
                    "source": "hackernews",
                    "url": f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                    "score": hit.get("points") or 0,
                    "ts": hit.get("created_at_i", 0),
                })
    except Exception as e:
        log.warning("scrape_hn error: %s", e)
    log.info("hackernews: %d items for %r", len(out), query)
    return out


# ─────────────────────────────────────────────────────────────────────
# Reddit — try multiple host variants and search modes; best-effort.
# ─────────────────────────────────────────────────────────────────────
async def scrape_reddit(query: str, limit: int = 25) -> list[dict]:
    """
    Reddit blocks `www.reddit.com/search.json` from many cloud IPs.
    `old.reddit.com` and the .json suffix on subreddit pages tend to work better,
    and search.rss never gets blocked. We try in order until something returns.
    """
    out = []
    headers = {"User-Agent": UA_FEED_READER}

    # Strategy 1: Reddit search RSS — most permissive
    rss_url = f"https://www.reddit.com/search.rss?q={quote_plus(query)}&sort=relevance&t=year"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, headers=headers, follow_redirects=True) as c:
            r = await c.get(rss_url)
            if r.status_code == 200 and r.text.strip().startswith("<"):
                try:
                    # Atom feed namespace
                    ns = {"a": "http://www.w3.org/2005/Atom"}
                    root = ET.fromstring(r.text)
                    for entry in root.findall("a:entry", ns):
                        title = _clean(getattr(entry.find("a:title", ns), "text", "") or "")
                        content = _clean(getattr(entry.find("a:content", ns), "text", "") or "")
                        link_el = entry.find("a:link", ns)
                        link = link_el.get("href") if link_el is not None else ""
                        author_el = entry.find("a:author/a:name", ns)
                        author = _clean(getattr(author_el, "text", "") or "anon")
                        text = title
                        if content and content.lower() != title.lower():
                            text = f"{title}. {content}"
                        if not _is_long_enough(text, 25):
                            continue
                        out.append({
                            "text": text[:2000],
                            "rating": None,
                            "author": author.replace("/u/", "u/"),
                            "source": "reddit",
                            "url": link,
                            "score": 0,
                            "ts": 0,
                        })
                        if len(out) >= limit:
                            break
                except ET.ParseError as e:
                    log.warning("reddit rss parse: %s", e)
            else:
                log.info("reddit rss status %d", r.status_code)
    except Exception as e:
        log.warning("reddit rss error: %s", e)

    # Strategy 2: old.reddit.com JSON if RSS gave us nothing
    if not out:
        json_url = f"https://old.reddit.com/search.json?q={quote_plus(query)}&limit={limit}&sort=relevance&t=year"
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT, headers=headers, follow_redirects=True) as c:
                r = await c.get(json_url)
                if r.status_code == 200:
                    for child in r.json().get("data", {}).get("children", []):
                        d = child.get("data", {})
                        title = d.get("title", "")
                        body = d.get("selftext", "") or ""
                        text = (title + ". " + body).strip(" .")
                        if not _is_long_enough(text, 25):
                            continue
                        out.append({
                            "text": text[:2000],
                            "rating": None,
                            "author": "u/" + d.get("author", "anon"),
                            "source": f"reddit:r/{d.get('subreddit', '?')}",
                            "url": "https://reddit.com" + d.get("permalink", ""),
                            "score": d.get("score", 0),
                            "ts": d.get("created_utc", 0),
                        })
                else:
                    log.info("old.reddit json status %d", r.status_code)
        except Exception as e:
            log.warning("old.reddit error: %s", e)

    log.info("reddit: %d items for %r", len(out), query)
    return out


# ─────────────────────────────────────────────────────────────────────
# DuckDuckGo News (no API key, works everywhere)
# ─────────────────────────────────────────────────────────────────────
async def scrape_ddg_news(query: str, limit: int = 20) -> list[dict]:
    """
    DDG's HTML news endpoint. Acts like a search aggregator — gives us a
    second wave of news beyond Google News.
    """
    out = []
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query + ' review opinion')}"
    try:
        async with httpx.AsyncClient(
            timeout=TIMEOUT,
            headers={"User-Agent": UA_DESKTOP},
            follow_redirects=True,
        ) as c:
            r = await c.post("https://html.duckduckgo.com/html/", data={"q": query + " review opinion"})
            if r.status_code != 200:
                log.info("ddg status %d", r.status_code)
                return out
            soup = BeautifulSoup(r.text, "html.parser")
            for result in soup.select(".result")[:limit]:
                title_el = result.select_one(".result__title")
                snippet_el = result.select_one(".result__snippet")
                link_el = result.select_one(".result__url")
                title = _clean(title_el.get_text() if title_el else "")
                snippet = _clean(snippet_el.get_text() if snippet_el else "")
                link = link_el.get("href", "") if link_el else ""
                text = title
                if snippet and snippet.lower() != title.lower():
                    text = f"{title}. {snippet}"
                if not _is_long_enough(text, 30):
                    continue
                domain = ""
                try:
                    domain = urlparse(link).netloc.replace("www.", "") if link else ""
                except Exception:
                    pass
                out.append({
                    "text": text[:2000],
                    "rating": None,
                    "author": domain or "DuckDuckGo",
                    "source": f"ddg:{domain or 'web'}",
                    "url": link,
                    "score": 0,
                    "ts": 0,
                })
    except Exception as e:
        log.warning("scrape_ddg_news error: %s", e)
    log.info("ddg: %d items for %r", len(out), query)
    return out


# ─────────────────────────────────────────────────────────────────────
# Smart router — runs all sources in parallel, dedups, returns combined
# ─────────────────────────────────────────────────────────────────────
SOURCE_FNS = {
    "news": scrape_google_news,
    "hackernews": scrape_hn,
    "reddit": scrape_reddit,
    "ddg": scrape_ddg_news,
}


async def scrape_all(query: str, sources: list[str], limit: int = 30) -> list[dict]:
    """Run all requested sources concurrently, dedup, return combined."""
    valid = [s for s in sources if s in SOURCE_FNS]
    if not valid:
        valid = ["news", "hackernews", "reddit"]

    # Per-source limit so we don't overshoot
    per = max(8, limit // max(len(valid), 1) + 5)

    tasks = [SOURCE_FNS[s](query, per) for s in valid]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    combined = []
    for src, res in zip(valid, results):
        if isinstance(res, Exception):
            log.warning("source %s raised: %s", src, res)
            continue
        combined.extend(res)

    # Dedup by first 120 chars (catches reposts and identical headlines)
    seen, uniq = set(), []
    for r in combined:
        k = (r.get("text") or "")[:120].lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)

    return uniq[:limit]


async def scrape_stream(query: str, sources: list[str], limit: int = 30):
    """
    Async-generator version: yields each review as soon as a source completes.
    Better UX — UI updates progressively instead of waiting for all sources.
    """
    valid = [s for s in sources if s in SOURCE_FNS]
    if not valid:
        valid = ["news", "hackernews", "reddit"]
    per = max(8, limit // max(len(valid), 1) + 5)

    tasks = {asyncio.create_task(SOURCE_FNS[s](query, per)): s for s in valid}
    seen = set()
    yielded = 0

    pending = set(tasks.keys())
    while pending and yielded < limit:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                results = task.result()
            except Exception as e:
                log.warning("source %s failed: %s", tasks[task], e)
                continue
            for r in results:
                k = (r.get("text") or "")[:120].lower()
                if k in seen:
                    continue
                seen.add(k)
                yield r
                yielded += 1
                if yielded >= limit:
                    break
