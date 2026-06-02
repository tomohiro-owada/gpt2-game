#!/usr/bin/env python3
"""Cache a whole MODEL FAMILY's per-token grading on the corpus into ONE file.

The sizes in a family share a tokenizer, so they grade the exact same walk. For
each passage step we record: the real next token, the decoys (top-k of the
LARGEST size, force-excluding the true token), and EVERY size's prediction —
the rank it gave the true token and its own #1 token. One teacher-forced forward
per passage per size (logits[t] predicts token t+1 given the true prefix).

Output: web/rounds.<family>.json
Env: FAMILY, LABEL, TOKENIZER, MODELS (json [{id,name,params}]),
     DECOY_FROM (name of the size whose top-k are the choices),
     QUANT8 / QUANT_BIG (comma-separated ids for 8-bit / 4-bit).
"""

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FAMILY = os.environ["FAMILY"]
LABEL = os.environ.get("LABEL", FAMILY)
TOKENIZER_ID = os.environ["TOKENIZER"]
MODELS = json.loads(os.environ["MODELS"])
DECOY_FROM = os.environ["DECOY_FROM"]
QUANT8 = set(filter(None, os.environ.get("QUANT8", "").split(",")))
QUANT_BIG = set(filter(None, os.environ.get("QUANT_BIG", "").split(",")))
START_WORDS, MAX_STEPS, DECOY_POOL = 3, 60, 9
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "web", f"rounds.{FAMILY}.json")


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
    decoy_idx = next(i for i, m in enumerate(MODELS) if m["name"] == DECOY_FROM)

    # passage/step skeleton (private "_" fields stripped before writing)
    passages = []
    for i, entry in enumerate(corpus):
        if isinstance(entry, dict):
            text, rough = entry.get("text", "").strip(), bool(entry.get("rough"))
        else:
            text, rough = str(entry).strip(), False
        ids = tok.encode(text)
        start = word_start_cut(ids, tok, START_WORDS)
        if len(ids) < start + 4:
            continue
        last = min(len(ids), start + MAX_STEPS)
        # byte-level BPE splits multibyte UTF-8 into byte tokens that decode to
        # the replacement char "�"; drop passages whose opening has one, and
        # truncate the walk at the first such token so we never grade/show "�"
        if any("�" in tok.decode([t]) for t in ids[:start]):
            continue
        for pos in range(start, last):
            if "�" in tok.decode([ids[pos]]):
                last = pos
                break
        if last - start < 4:
            continue
        steps = [{"true_token": tok.decode([ids[pos]]), "true_token_id": ids[pos],
                  "_pos": pos, "decoys": [], "models": [None] * len(MODELS)}
                 for pos in range(start, last)]
        passages.append({"id": i, "rough": rough, "_ids": ids, "_last": last,
                         "prefix": tok.decode(ids[:start]),
                         "prefix_tokens": [tok.decode([t]) for t in ids[:start]],
                         "steps": steps})

    hits = [0] * len(MODELS)
    total = 0
    for mi, m in enumerate(MODELS):
        mid = m["id"]
        if mid in QUANT8 or mid in QUANT_BIG:
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
        with torch.no_grad():
            for p in passages:
                logits = model(torch.tensor([p["_ids"][:p["_last"]]],
                                            device=device)).logits[0].float()
                for s in p["steps"]:
                    row = logits[s["_pos"] - 1]
                    dist = torch.softmax(row, dim=-1)
                    tid = s["true_token_id"]
                    rank = int((dist > dist[tid]).sum()) + 1
                    s["models"][mi] = {"rank": rank,
                                       "top": tok.decode([int(row.argmax())]),
                                       "prob": round(float(dist[tid]), 6)}
                    if mi == decoy_idx:
                        hits[mi] += (rank == 1)
                        tp, ti = torch.topk(dist, DECOY_POOL + 1)
                        for r, (pr, did) in enumerate(zip(tp.tolist(), ti.tolist()), 1):
                            dtok = tok.decode([did])
                            if did == tid or dtok == s["true_token"]:
                                continue
                            s["decoys"].append({"token": dtok, "token_id": did,
                                                "prob": round(float(pr), 6), "rank": r})
                            if len(s["decoys"]) >= DECOY_POOL:
                                break
                    else:
                        hits[mi] += (rank == 1)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        n = sum(len(p["steps"]) for p in passages)
        print(f"  {m['name']} ({m['params']}): {round(100 * hits[mi] / n, 1)}%")

    n = sum(len(p["steps"]) for p in passages)
    sizes = [{"name": m["name"], "params": m["params"], "pct": round(100 * hits[i] / n, 1)}
             for i, m in enumerate(MODELS)]
    for p in passages:
        p.pop("_ids", None); p.pop("_last", None)
        for s in p["steps"]:
            s.pop("_pos", None)
    out = {"family": FAMILY, "label": LABEL, "sizes": sizes, "decoy_from": decoy_idx,
           "start_words": START_WORDS, "max_steps": MAX_STEPS, "decoy_pool": DECOY_POOL,
           "count": len(passages), "passages": passages}
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {FAMILY}: {len(passages)} passages × {len(MODELS)} sizes -> {OUT}")


if __name__ == "__main__":
    main()
