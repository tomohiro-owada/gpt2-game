#!/usr/bin/env python3
"""Build the whole corpus from real OpenWebText (GPT-2's WebText training data).

Two tiers, both from OWT, written straight to corpus.json:

  * "good"  (rough: false) — English, **good start**: begins at a real sentence
    boundary (capitalized), whitespace-normalized, well-formed, high-ASCII.
  * "raw"   (rough: true)  — **anything permitted**: a window from a random spot
    in the document (usually mid-sentence), newlines kept, any latin-script text.

Both pass a safety blocklist (slurs / explicit terms) — we won't ship those into a
game even under "anything". Deterministic (fixed seed).
"""

import json
import os
import random
import re

from datasets import load_dataset

N_GOOD = 45     # English, good sentence start  -> rough: False
N_RAW = 40      # anything permitted, raw        -> rough: True
SEED = 11
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "corpus.json")

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


def good_chunk(text):
    """A clean English passage that starts at a capitalized sentence."""
    text = re.sub(r"\s+", " ", text.strip())
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    starts = [i for i, s in enumerate(sents) if s[:1].isupper() and len(s) > 25]
    random.shuffle(starts)
    for st in starts:
        buf = ""
        for s in sents[st:]:
            buf = (buf + " " + s).strip()
            if len(buf) >= 210:
                break
        if len(buf) > 360:
            c = buf.rfind(" ", 0, 340)
            buf = buf[:c if c > 200 else 340].rstrip()
        if 190 <= len(buf) <= 360 and looks_english(buf) and ascii_ratio(buf) > 0.985:
            return buf
    return None


def raw_chunk(text):
    """A raw window from a random spot in the document (anything permitted)."""
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) < 240:
        return None
    start = random.randint(0, int(len(text) * 0.6))
    sp = text.find(" ", start)
    start = sp + 1 if sp != -1 else 0
    win = text[start:start + 340]
    cut = win.rfind(" ")
    if cut > 200:
        win = win[:cut]
    win = win.strip()
    if 150 <= len(win) <= 360 and win.count(" ") >= 12 and ascii_ratio(win) > 0.90:
        return win
    return None


def load_owt():
    try:
        return load_dataset("stas/openwebtext-10k", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"  10k subset failed ({e}); streaming full openwebtext...")
        return load_dataset("Skylion007/openwebtext", split="train",
                            streaming=True, trust_remote_code=True)


def main():
    random.seed(SEED)
    ds = load_owt()
    good, raw, seen = [], [], set()
    for i, row in enumerate(ds):
        if i > 9000 or (len(good) >= N_GOOD and len(raw) >= N_RAW):
            break
        text = row.get("text", "")
        if len(good) < N_GOOD:
            g = good_chunk(text)
            if g and safe(g) and g[:40] not in seen:
                seen.add(g[:40]); good.append(g); continue
        if len(raw) < N_RAW:
            r = raw_chunk(text)
            if r and safe(r) and r[:40] not in seen:
                seen.add(r[:40]); raw.append(r)

    corpus = ([{"text": g, "rough": False} for g in good] +
              [{"text": r, "rough": True} for r in raw])
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(good)} good + {len(raw)} raw = {len(corpus)} OWT passages to {OUT}")


if __name__ == "__main__":
    main()
