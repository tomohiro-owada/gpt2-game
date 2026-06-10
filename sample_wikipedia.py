#!/usr/bin/env python3
"""Build the corpus from random English Wikipedia article intros (CC BY-SA 4.0).

Replaces sample_owt.py's OpenWebText source with freely licensed text. Two tiers,
same shape as before, written straight to corpus.json:

  * "good"  (rough: false) — clean English intro, starts at the article's first
    sentence, whitespace-normalized, well-formed, high-ASCII.
  * "raw"   (rough: true)  — a window from a random spot in the intro (usually
    mid-sentence), newlines kept.

Each entry also records the source article title and URL so attribution
(CC BY-SA 4.0) can be surfaced in the app. Both tiers pass the same safety
blocklist as the original sampler. Deterministic given the fetched pool order
is shuffled with a fixed seed.
"""

import json
import os
import random
import re
import time
import urllib.parse
import urllib.request

N_GOOD = 45
N_RAW = 40
SEED = 11
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "corpus.json")

API = ("https://en.wikipedia.org/w/api.php?action=query&format=json"
       "&generator=random&grnnamespace=0&grnlimit=20"
       "&prop=extracts|info&explaintext=1&exintro=1&inprop=url")
UA = {"User-Agent": "gpt2-game-corpus-sampler/1.0 (one-off corpus build)"}

BLOCK = [
    "nigger", "nigga", "faggot", "fag", "retard", "retarded", "spic", "chink",
    "kike", "wetback", "tranny", "coon", "dyke", "cunt",
    "fuck", "shit", "bitch", "porn", "cum", "cock", "dick", "pussy", "blowjob",
    "rape", "rapist", "molest", "pedophile", "incest", "bestiality", "nsfw",
]
BLOCK_RE = re.compile(r"(?i)\b(" + "|".join(re.escape(w) for w in BLOCK) + r")\b")

EN_COMMON = {"the", "and", "of", "to", "in", "is", "that", "it", "for", "on",
             "with", "as", "was", "are", "this", "but", "not", "you", "have",
             "be", "at", "or", "from", "they", "he", "she", "we", "his", "her"}


def ascii_ratio(s):
    if not s:
        return 0.0
    return sum(1 for c in s if (32 <= ord(c) < 127) or c in "\n\t") / len(s)


def looks_english(t):
    words = re.findall(r"[a-z']+", t.lower())
    if len(words) < 18:
        return False
    return sum(1 for w in set(words) if w in EN_COMMON) >= 5


def safe(t):
    return bool(t) and not BLOCK_RE.search(t)


def cut_words(t, lo=260, hi=380):
    """Cut at a word boundary somewhere in [lo, hi] chars."""
    if len(t) <= hi:
        return t
    cut = t.rfind(" ", lo, hi)
    return t[:cut] if cut > 0 else t[:hi]


def fetch_pool(target=250):
    pool = []
    seen = set()
    while len(pool) < target:
        for attempt in range(6):
            try:
                with urllib.request.urlopen(
                        urllib.request.Request(API, headers=UA), timeout=30) as r:
                    data = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == 5:
                    raise
                wait = 5 * (2 ** attempt)
                print(f"  429, waiting {wait}s...")
                time.sleep(wait)
        for page in data.get("query", {}).get("pages", {}).values():
            title = page.get("title", "")
            text = (page.get("extract") or "").strip()
            url = page.get("fullurl") or (
                "https://en.wikipedia.org/wiki/" +
                urllib.parse.quote(title.replace(" ", "_")))
            if title in seen or len(text) < 350:
                continue
            seen.add(title)
            pool.append({"title": title, "url": url, "text": text})
        print(f"  pool: {len(pool)}", end="\r")
        time.sleep(1.5)
    print()
    return pool


def main():
    rng = random.Random(SEED)
    pool = fetch_pool()
    rng.shuffle(pool)

    out = []
    used = set()

    # good tier: article intro from its first sentence
    for art in pool:
        if len([e for e in out if not e["rough"]]) >= N_GOOD:
            break
        t = re.sub(r"\s+", " ", art["text"]).strip()
        if not t or not t[0].isupper():
            continue
        t = cut_words(t)
        if len(t) < 240 or ascii_ratio(t) < 0.97 or not looks_english(t) or not safe(t):
            continue
        used.add(art["title"])
        out.append({"text": t, "rough": False,
                    "source": art["title"], "source_url": art["url"],
                    "license": "CC BY-SA 4.0"})

    # raw tier: window from a random spot, newlines kept
    for art in pool:
        if len([e for e in out if e["rough"]]) >= N_RAW:
            break
        if art["title"] in used:
            continue
        text = art["text"]
        if len(text) < 420:
            continue
        start = rng.randrange(0, max(1, len(text) - 400))
        # snap to a whitespace so we never start mid-word (mid-sentence is fine)
        ws = text.find(" ", start)
        if ws < 0 or ws + 240 > len(text):
            continue
        t = cut_words(text[ws + 1:].strip())
        if len(t) < 240 or ascii_ratio(t) < 0.92 or not looks_english(t) or not safe(t):
            continue
        used.add(art["title"])
        out.append({"text": t, "rough": True,
                    "source": art["title"], "source_url": art["url"],
                    "license": "CC BY-SA 4.0"})

    good = sum(1 for e in out if not e["rough"])
    raw = len(out) - good
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {OUT}: {good} good + {raw} raw = {len(out)} passages "
          f"(all English Wikipedia intros, CC BY-SA 4.0)")


if __name__ == "__main__":
    main()
