# Play as GPT-2 🤖

A tiny web game: a real passage unfolds **one word at a time** from a tiny opening,
and you guess the word that *actually* comes next. GPT-2's most-confident predictions
are mixed in as the tempting **decoys** — so you're racing the model's intuition
against the real text. Every correct word is revealed and the passage grows.

Four modes (chosen on the start screen):

| Mode   | What you do                | Random-guess baseline |
|--------|----------------------------|-----------------------|
| Easy   | pick from 3 choices        | 33%                   |
| Medium | pick from 6 choices        | 17%                   |
| Hard   | pick from 9 choices        | 11%                   |
| Insane | **type the word** yourself | ~0%                   |

In the choice modes you can press **`1`–`9`** to pick (or click). The correct answer
is always the *real* next token in the source text — which is often **not** GPT-2's
top guess, so the app always force-includes it among the choices. After each guess
the game **auto-advances** to the next word (no button to click — press Enter/Space
to skip the brief pause), with the previous word's result pinned just below.

Don't like a passage? Hit **skip** to draw a different one. Every passage is real
**OpenWebText** (the corpus GPT-2 trained on); an **English / Mixed / Anything**
control on the start screen filters it: *English* = clean, sentence-aligned English
chunks; *Anything* = raw chunks from anywhere in a document (mid-sentence, newlines,
any latin-script text). Defaults to Mixed, remembered across visits. In Insane mode,
the right letters with the wrong leading space (e.g. typing `ing` for the token
` ing`) scores a **yellow near-miss**.

Finished games are saved to a **personal leaderboard** (browser `localStorage`
only — nothing leaves your machine), shown ranked by accuracy on the start and end
screens, with a **clear** button.

## Families (pick one on the start screen)

You pick a **model family**; **every token is graded by all sizes in it**. The
largest size supplies the decoys (your choices), and after each guess you see how
every size did. Two families:

| Family   | Sizes                         | Notes                                   |
|----------|-------------------------------|-----------------------------------------|
| **GPT-2**| 124M / 355M / 774M / 1.5B     | OpenAI, 2019. Trained on this kind of web text. |
| **Qwen3**| 0.6B / 1.7B / 4B / 8B / 14B   | Alibaba, 2025, Apache-2.0. 14B runs 8-bit to fit a 24GB GPU. |

`cache_family.py` builds one `web/rounds.<family>.json` per family (listed in
`web/families.json`). Per step it stores the real token, the largest size's top-k as
decoys, and **every size's `{rank, top, prob}`** for that token (one teacher-forced
forward per passage per size). Each size's corpus-wide rank-1 rate becomes the
**size ladder** shown live in-game and on the end screen ("you predicted like GPT-2
medium"). Fair comparison only in **Insane** mode (choice modes give you fewer
options than the models' full vocabulary). Big sizes fit a 3090 via `QUANT=8bit` /
`QUANT=4bit` (needs `bitsandbytes` + `accelerate`).

All models run via 🤗 `transformers` (pinned `>=4,<5`) and fit comfortably on a
single RTX 3090 (24 GB) in fp16.

## Layout

```
gpt2-game/
├── corpus.json          # plain-text passages walked one token at a time
├── cache_family.py      # grades every passage with all sizes of a family -> web/rounds.<family>.json
├── sample_owt.py        # builds corpus.json from real OpenWebText
├── requirements.txt
├── serve.py             # zero-dependency static server (stdlib)
├── run.sh               # cache (if needed) + serve
└── web/
    ├── index.html
    ├── style.css
    ├── app.js
    ├── families.json     # family registry (key, label, file)
    ├── rounds.gpt2.json  # GPT-2 family, every token graded by all 4 sizes (generated)
    └── rounds.qwen3.json # Qwen3 family, every token graded by all 5 sizes (generated)
```

## Quick start

### 1. Cache the families (needs the GPU box, a few min on a 3090)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt bitsandbytes accelerate
FAMILY=gpt2 LABEL="GPT-2" TOKENIZER=gpt2 DECOY_FROM="GPT-2 XL" \
  MODELS='[{"id":"gpt2","name":"GPT-2 small","params":"124M"}, ... ]' python cache_family.py
FAMILY=qwen3 LABEL="Qwen3" TOKENIZER=Qwen/Qwen3-0.6B-Base DECOY_FROM="Qwen3 14B" \
  QUANT8="Qwen/Qwen3-14B-Base" MODELS='[ ... ]' python cache_family.py
```

`DECOY_FROM` = the size whose top-k become the choices (use the largest). `QUANT8` /
`QUANT_BIG` list ids to load in 8-bit / 4-bit so big sizes fit a 24GB GPU.

### 2. Play

```bash
python serve.py                   # http://localhost:8000
```

No build step, no JS framework — just static files + one JSON.

## Corpus

`corpus.json` is entirely real **OpenWebText** — the open reproduction of the WebText
corpus GPT-2 was actually trained on (Reddit-outbound links; news/blog-heavy and
genuinely messy). `sample_owt.py` builds it on the GPU box (needs `datasets`) in two
tiers, each entry `{"text": "...", "rough": <bool>}`:

- `rough: false` — **English, good start**: a chunk that begins at a capitalized
  sentence, whitespace-normalized, high-ASCII, passing a common-English-word check.
- `rough: true` — **anything permitted**: a raw window from a random spot in a
  document (usually mid-sentence), newlines kept, any latin-script text.

Both pass a slur/explicit blocklist. The `rough` flag rides through to each passage
in the `rounds.<family>.json` files, driving the start-screen filter and skip.

## How a game is built

`cache_family.py` takes each corpus entry and:

1. opens with the first **3 whole words** (`START_WORDS`) so the context never cuts
   mid-word;
2. walks forward through the **original** text up to `MAX_STEPS` (60) tokens (~50
   words) via one teacher-forced forward per size; at each step it records the **real
   next token**, the largest size's top `DECOY_POOL` (9) tokens as decoys, and **every
   size's** rank of the real token + its own #1 token;
3. writes one entry per passage to `web/rounds.<family>.json`.

A game picks one random passage and walks it: the browser shows the true token plus
`k-1` decoys (`k` = 3 / 6 / 9), shuffled, and asks you to find the real one — or, in
Insane mode, to type it. After each guess the real token is revealed and appended (so
the passage grows into a paragraph), the readout shows **real · you · largest size**,
and a strip grades the token across **all sizes**.

## Deploying with nginx

It's all static — copy `web/` into your webroot:

```bash
sudo cp -r web/. /var/www/gpt2-game/
```

Serve the `rounds.*.json` and `models.json` files with `Cache-Control: no-store` so
updated data isn't cached stale.
