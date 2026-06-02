#!/usr/bin/env python3
"""Cache a token-by-token "guess the real next word" walk for each corpus snippet.

The game starts each passage with a *tiny* prefix and then walks forward through
the ORIGINAL text one token at a time. At every step we record:

  * the real next token (the ground truth the player must find), and
  * GPT-2's most-likely tokens, which become the tempting *decoys*.

So the player is asked "what word actually comes next?" while GPT-2's confident
predictions sit alongside it as plausible wrong answers. The real token is often
*not* in GPT-2's top-k, so the web app always force-includes it among the shown
choices (true token + top decoys, shuffled, sliced to k = 3 / 6 / 9).

Output is written to web/rounds.json. Runs on a single RTX 3090 in fp16 in a few
minutes. transformers must be >=4,<5.
"""

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# pick any causal LM: `MODEL=Qwen/Qwen3-0.6B-Base python cache_logits.py`
MODEL_NAME = os.environ.get("MODEL", os.environ.get("GPT2_MODEL", "gpt2-xl"))
START_WORDS = 3         # passage opens with this many whole words (very little!)
MAX_STEPS = 60          # how many next-token guesses to walk per passage (~50 words)
DECOY_POOL = 9          # model decoys cached per step (enough for Hard = 9 choices)
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(HERE, "corpus.json")


def slugify(name: str) -> str:
    """huggingface id -> file-friendly slug, e.g. Qwen/Qwen3-0.6B-Base -> qwen3-0.6b-base"""
    return name.split("/")[-1].lower().replace("_", "-")


# one file per model, so the web app can offer a choice of opponent
OUT_PATH = os.path.join(HERE, "web", f"rounds.{slugify(MODEL_NAME)}.json")


def word_start_cut(ids, tokenizer, n_words):
    """Token index where word #(n_words+1) begins, so the prefix holds exactly
    n_words whole words. Word starts = first token, or any token GPT-2 renders
    with a leading space. Keeps the opening context from cutting mid-word."""
    words = 0
    for idx, tid in enumerate(ids):
        if idx == 0 or tokenizer.decode([tid]).startswith(" "):
            words += 1
        if words > n_words:
            return idx
    return len(ids)


def main() -> None:
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    quant = os.environ.get("QUANT")        # "4bit" / "8bit" to fit big models on a 3090
    print(f"Loading {MODEL_NAME} on {device} ({quant or dtype})...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if quant in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        bnb = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.float16)
               if quant == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype)
        model.to(device)
    model.eval()

    passages = []
    for i, entry in enumerate(corpus):
        # a corpus entry is either a plain string (clean prose) or
        # {"text": ..., "rough": true} for messier real-internet-style text
        if isinstance(entry, dict):
            text = entry.get("text", "").strip()
            rough = bool(entry.get("rough", False))
        else:
            text = str(entry).strip()
            rough = False
        ids = tokenizer.encode(text)
        start = word_start_cut(ids, tokenizer, START_WORDS)
        # need the opening prefix plus a few tokens to actually predict
        if len(ids) < start + 4:
            continue

        last = min(len(ids), start + MAX_STEPS)

        # one teacher-forced forward over the whole prefix gives every step's
        # next-token distribution: logits[t] is the prediction for token t+1,
        # conditioned on the true tokens up to t — identical to feeding each
        # growing prefix separately, but ~MAX_STEPS times faster.
        with torch.no_grad():
            inp = torch.tensor([ids[:last]], device=device)
            all_logits = model(inp).logits[0].float()       # [last, vocab]

        steps = []
        for pos in range(start, last):
            true_id = ids[pos]

            probs = torch.softmax(all_logits[pos - 1], dim=-1)
            top_probs, top_ids = torch.topk(probs, DECOY_POOL + 1)

            true_token = tokenizer.decode([true_id])
            # GPT-2's view of the *real* token: its probability and full-vocab rank
            true_prob = float(probs[true_id])
            true_rank = int((probs > probs[true_id]).sum().item()) + 1

            # decoys = GPT-2's top tokens, minus any that collide with the true
            # token by id OR by rendered text (so no two identical-looking buttons)
            decoys = []
            for rank, (p, tid) in enumerate(zip(top_probs.tolist(), top_ids.tolist()), start=1):
                tok = tokenizer.decode([tid])
                if tid == true_id or tok == true_token:
                    continue
                decoys.append({
                    "token": tok,
                    "token_id": tid,
                    "prob": round(float(p), 6),
                    "rank": rank,
                })
                if len(decoys) >= DECOY_POOL:
                    break

            steps.append({
                "true_token": true_token,
                "true_token_id": true_id,
                "true_prob": round(true_prob, 6),
                "true_rank": true_rank,
                "decoys": decoys,
            })

        passages.append({
            "id": i,
            "rough": rough,   # messier text the player can filter out / skip
            # the opening context as a string AND tokenized, so the web app can
            # render each starting token (with a neutral "given" style)
            "prefix": tokenizer.decode(ids[:start]),
            "prefix_tokens": [tokenizer.decode([t]) for t in ids[:start]],
            "steps": steps,
        })
        if (i + 1) % 10 == 0:
            print(f"  walked {i + 1}/{len(corpus)} passages")

    out = {
        "model": MODEL_NAME,
        "start_words": START_WORDS,
        "max_steps": MAX_STEPS,
        "decoy_pool": DECOY_POOL,
        "count": len(passages),
        "passages": passages,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"Wrote {len(passages)} passages to {OUT_PATH}")


if __name__ == "__main__":
    main()
