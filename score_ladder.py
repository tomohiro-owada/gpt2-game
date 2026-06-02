#!/usr/bin/env python3
"""Compute the rank-1 hit rate of several model sizes from ONE family (they must
share a tokenizer) on the corpus — the "are you as good as GPT-2 small / XL?"
ladder. Merges results into web/ladders.json, keyed by family.

Fast: one teacher-forced forward per passage gives every step's prediction at
once (we follow the real tokens, so logits[t] predicts token t+1).

Env:
  FAMILY     family key (e.g. "gpt2", "qwen3")
  LABEL      display label
  TOKENIZER  shared tokenizer id used to define the walk
  MODELS     JSON list of {"id","name","params"}
  QUANT_BIG  comma-separated model ids to load in 4-bit (optional)
"""

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FAMILY = os.environ["FAMILY"]
LABEL = os.environ.get("LABEL", FAMILY)
TOKENIZER_ID = os.environ["TOKENIZER"]
MODELS = json.loads(os.environ["MODELS"])
QUANT_BIG = set(filter(None, os.environ.get("QUANT_BIG", "").split(",")))   # 4-bit
QUANT8 = set(filter(None, os.environ.get("QUANT8", "").split(",")))         # 8-bit
START_WORDS, MAX_STEPS = 3, 60
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "web", "ladders.json")


def word_start_cut(ids, tok, n):
    w = 0
    for i, t in enumerate(ids):
        if i == 0 or tok.decode([t]).startswith(" "):
            w += 1
        if w > n:
            return i
    return len(ids)


def main():
    corpus = json.load(open(os.path.join(HERE, "corpus.json")))
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # precompute the walks once (shared tokenizer => identical across sizes)
    walks = []
    for entry in corpus:
        text = (entry["text"] if isinstance(entry, dict) else entry).strip()
        ids = tok.encode(text)
        start = word_start_cut(ids, tok, START_WORDS)
        last = min(len(ids), start + MAX_STEPS)
        if last - start >= 4:
            walks.append((ids, start, last))

    rungs = []
    for m in MODELS:
        mid = m["id"]
        if mid in QUANT_BIG or mid in QUANT8:
            from transformers import BitsAndBytesConfig
            bnb = (BitsAndBytesConfig(load_in_8bit=True) if mid in QUANT8
                   else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.float16))
            model = AutoModelForCausalLM.from_pretrained(mid, quantization_config=bnb,
                                                        device_map="auto")
        else:
            dt = torch.float16 if device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(mid, dtype=dt).to(device)
        model.eval()

        hits = tot = 0
        with torch.no_grad():
            for ids, start, last in walks:
                inp = torch.tensor([ids[:last]], device=device)
                logits = model(inp).logits[0]               # [last, vocab]
                preds = logits[start - 1:last - 1].argmax(-1)
                truth = torch.tensor(ids[start:last], device=preds.device)
                hits += int((preds == truth).sum())
                tot += truth.numel()
        pct = round(100 * hits / tot, 1)
        rungs.append({"name": m["name"], "params": m["params"], "pct": pct})
        print(f"  {m['name']} ({m['params']}): {pct}%")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    rungs.sort(key=lambda r: r["pct"])
    data = json.load(open(OUT)) if os.path.exists(OUT) else {}
    data[FAMILY] = {"label": LABEL, "rungs": rungs}
    json.dump(data, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {FAMILY} ladder ({len(rungs)} rungs) to {OUT}")


if __name__ == "__main__":
    main()
