#!/usr/bin/env python3
"""Build a Japanese corpus from real Japanese Wikipedia (REST/Action API).

The Japanese counterpart of sample_owt.py. Two tiers, written to corpus.json:

  * "good" (rough: false) — a clean article INTRO: plain-text lead section,
    whitespace-normalised, started at a sentence and cut at a sentence end (。!?).
  * "raw"  (rough: true)  — a window from a random spot deeper in the article
    (usually mid-sentence), newlines kept.

Both pass a small safety blocklist. Deterministic given the seed (the random
title fill uses Wikipedia's own randomness, so the exact set can vary; the
curated seeds are stable). No heavyweight `datasets` dependency — just stdlib
urllib against ja.wikipedia.org.
"""

import json
import os
import random
import re
import time
import urllib.parse
import urllib.request

N_GOOD = 50     # clean article intros        -> rough: False
N_RAW = 20      # raw mid-article windows      -> rough: True
# the playable model (rinna) tokenizer: we reject any passage whose playable
# window tokenizes to an <unk> (out-of-vocab Latin runs / rare kanji), so the
# game never asks you to guess "<unk>" and passages stay full length.
MODEL_TOKENIZER = "rinna/japanese-gpt2-medium"
UNK_CHECK_TOKENS = 70   # START(≤5) + MAX_STEPS(60) walked, with headroom
SEED = 11
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "corpus.json")
API = "https://ja.wikipedia.org/w/api.php"
UA = "gpt2-game-jp/1.0 (corpus builder; https://localhost)"

# A spread of well-known topics for clean, high-quality intros. Random articles
# fill the rest (and supply most of the rough windows).
SEEDS = [
    "京都市", "源氏物語", "富士山", "日本", "東京都", "徳川家康", "織田信長",
    "夏目漱石", "宮沢賢治", "黒澤明", "宮崎駿", "新幹線", "和食", "寿司", "茶道",
    "桜", "紅葉", "琵琶湖", "鴨川", "金閣寺", "清水寺", "伏見稲荷大社", "祇園祭",
    "大阪市", "奈良市", "北海道", "沖縄県", "富岡製糸場", "浮世絵", "歌舞伎",
    "能", "落語", "相撲", "柔道", "剣道", "将棋", "囲碁", "俳句", "短歌",
    "万葉集", "平家物語", "枕草子", "徒然草", "明治維新", "戦国時代", "江戸時代",
    "縄文時代", "弥生時代", "卑弥呼", "聖徳太子", "坂本龍馬", "西郷隆盛",
    "野口英世", "湯川秀樹", "本田宗一郎", "豊臣秀吉", "紫式部", "清少納言",
    "松尾芭蕉", "葛飾北斎", "伊能忠敬", "天体", "太陽系", "ブラックホール",
    "量子力学", "進化論", "DNA", "光合成", "火山", "地震", "台風", "オーロラ",
]

# crude safety blocklist (Japanese + a few latin terms); we won't ship these.
BLOCK = ["セックス", "ポルノ", "強姦", "レイプ", "売春", "児童ポルノ", "わいせつ",
         "性的暴行", "ペドフィリア", "近親相姦", "porn", "rape", "nsfw"]
BLOCK_RE = re.compile("|".join(re.escape(w) for w in BLOCK), re.IGNORECASE)

JP_RE = re.compile(r"[ぁ-んァ-ヶ一-龯]")          # has hiragana/katakana/kanji
SENT_END = re.compile(r"(?<=[。！？])")           # split keeping the delimiter


def api_get(params):
    params = {**params, "format": "json", "formatversion": "2"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_extract(title, intro_only):
    """Plain-text extract for a title (lead section only when intro_only)."""
    params = {"action": "query", "prop": "extracts", "explaintext": "1",
              "redirects": "1", "titles": title}
    if intro_only:
        params["exintro"] = "1"
    pages = api_get(params).get("query", {}).get("pages", [])
    if not pages:
        return ""
    return pages[0].get("extract", "") or ""


def random_titles(n):
    out = []
    while len(out) < n:
        batch = api_get({"action": "query", "list": "random",
                         "rnnamespace": "0", "rnlimit": "20"})
        out += [r["title"] for r in batch.get("query", {}).get("random", [])]
        time.sleep(0.2)
    return out[:n]


def clean(text):
    # drop parentheticals that carry Latin/ASCII (English names, romanizations,
    # symbols) — the rinna SentencePiece vocab maps those to <unk>
    text = re.sub(r"[（(][^（()）]*[A-Za-z0-9][^（()）]*[）)]", "", text)
    # drop parenthetical pronunciation clutter and collapse whitespace
    text = re.sub(r"[ \t　]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text.strip())
    return text


_TOK = None


def model_clean(text):
    """True if the playable token window has NO <unk> for the rinna tokenizer."""
    global _TOK
    if _TOK is None:
        from transformers import T5Tokenizer
        _TOK = T5Tokenizer.from_pretrained(MODEL_TOKENIZER)
    ids = _TOK.encode(text, add_special_tokens=False)[:UNK_CHECK_TOKENS]
    return _TOK.unk_token_id not in ids


def safe(t):
    return bool(t) and not BLOCK_RE.search(t)


def good_chunk(extract):
    """A clean intro passage: whole sentences, ~190–360 chars, started clean."""
    text = re.sub(r"\s+", "", clean(extract))   # JP needs no inter-word spaces
    # drop the leading "タイトル（よみがな…）" gloss so play starts on real prose,
    # not a kana-reading puzzle that every Wikipedia lead opens with
    text = re.sub(r"^([^（(]{1,20})[（(][^）)]{0,80}[）)]", r"\1", text)
    sents = [s for s in SENT_END.split(text) if s.strip()]
    buf = ""
    for s in sents:
        buf += s
        if len(buf) >= 200:
            break
    if len(buf) > 360:
        return None
    if 150 <= len(buf) <= 360 and len(JP_RE.findall(buf)) >= 60:
        return buf
    return None


def raw_chunk(extract):
    """A raw window from a random spot deeper in the article."""
    text = clean(extract)
    if len(text) < 300:
        return None
    start = random.randint(int(len(text) * 0.2), int(len(text) * 0.6))
    win = text[start:start + 300]
    # don't start mid-... we deliberately allow mid-sentence (that's the point)
    if 150 <= len(win) <= 360 and len(JP_RE.findall(win)) >= 60:
        return win.strip()
    return None


def main():
    random.seed(SEED)
    good, raw, seen = [], [], set()

    titles = SEEDS[:]
    # pull extra random titles to fill both tiers (the <unk> filter rejects many,
    # so over-fetch to leave enough clean candidates)
    titles += random_titles(2 * (N_GOOD + N_RAW))

    for title in titles:
        if len(good) >= N_GOOD and len(raw) >= N_RAW:
            break
        try:
            ex = fetch_extract(title, intro_only=False)
        except Exception as e:
            print(f"  skip {title}: {e}")
            continue
        time.sleep(0.15)
        if not ex:
            continue
        if len(good) < N_GOOD:
            g = good_chunk(ex)
            if g and safe(g) and model_clean(g) and g[:24] not in seen:
                seen.add(g[:24]); good.append(g); continue
        if len(raw) < N_RAW:
            r = raw_chunk(ex)
            if r and safe(r) and model_clean(r) and r[:24] not in seen:
                seen.add(r[:24]); raw.append(r)

    corpus = ([{"text": g, "rough": False} for g in good] +
              [{"text": r, "rough": True} for r in raw])
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(good)} good + {len(raw)} raw = {len(corpus)} JA-Wikipedia "
          f"passages to {OUT}")


if __name__ == "__main__":
    main()
