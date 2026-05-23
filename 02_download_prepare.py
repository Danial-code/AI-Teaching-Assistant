#!/usr/bin/env python3
# scripts/02_download_prepare.py
# Downloads multiple lecture/meeting/paper datasets, cleans & segments them,
# deduplicates, stratifies by domain, and writes train/val/test JSONL files.

import os, re, json, random, hashlib
from pathlib import Path
import yaml, jsonlines
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# --------- Paths & config ---------
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data_prepared"
OUT_DIR.mkdir(exist_ok=True)

CFG_DS_PATH = ROOT / "config" / "datasets.yaml"
CFG_PP_PATH = ROOT / "config" / "preprocessing.yaml"

cfg_ds = yaml.safe_load(CFG_DS_PATH.read_text())
cfg_pp = yaml.safe_load(CFG_PP_PATH.read_text())

limits = cfg_ds.get("limits", {})
splits = cfg_ds.get("splits", {"train": 0.80, "val": 0.10, "test": 0.10})

MAX_IN_TOK = int(limits.get("max_input_tokens", 2000))
MIN_IN_TOK = int(limits.get("min_input_tokens", 200))
MAX_OUT_CH = int(limits.get("max_output_chars", 1200))

random.seed(42)

# --------- Tokenizer (with SAFE fallback) ---------
def get_tokenizer():
    """
    Try LLaMA tokenizer (requires HF access). If not available, fall back to GPT-2,
    which is totally fine for counting tokens and chunking during preprocessing.
    """
    try:
        return AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", use_fast=True
        )
    except Exception:
        return AutoTokenizer.from_pretrained("gpt2", use_fast=True)

tok = get_tokenizer()

# --------- Cleaning helpers ---------
CTRL  = re.compile(r"[\u0000-\u001F\u007F]")
STAMP = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")  # 12:34 or 01:02:03
STAGE = re.compile(r"\[(?:applause|laughter|music|silence|noise|pause)\]", re.I)

def clean_text(s: str) -> str:
    if not s: return ""
    s = s.replace("\ufeff", "").replace("\u200b", "")
    if cfg_pp["cleaning"].get("strip_stage_directions", True):
        s = STAGE.sub("", s)
    if cfg_pp["cleaning"].get("strip_timestamps", True):
        s = STAMP.sub("", s)
    s = CTRL.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def count_toks(s: str) -> int:
    return len(tok(s, add_special_tokens=False).input_ids)

def chunk_by_tokens(text: str, max_toks=2000, min_toks=200):
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, buf, cur = [], [], 0
    for sent in sents:
        t = count_toks(sent)
        if t > max_toks:
            # drop ultra-long single sentences
            continue
        if cur + t > max_toks and cur >= min_toks:
            chunks.append(" ".join(buf)); buf, cur = [], 0
        buf.append(sent); cur += t
    if cur >= min_toks:
        chunks.append(" ".join(buf))
    return chunks

def bullets_from_summary(s: str, max_items=7) -> str:
    if not s: return ""
    parts = re.split(r"(?<=[.!?])\s+", s)
    items = [p.strip() for p in parts if 3 < len(p.strip()) < 300][:max_items]
    return "\n".join(f"- {x}" for x in items) if items else ""

def pack_rows(source: str, domain: str, inp: str, summ: str):
    inp, summ = clean_text(inp), clean_text(summ)
    if len(inp) < cfg_pp["cleaning"]["min_input_chars"]:
        return []
    if len(summ) < cfg_pp["cleaning"]["min_output_chars"]:
        return []
    out = bullets_from_summary(summ) or summ[:MAX_OUT_CH]
    rows = []
    if count_toks(inp) > MAX_IN_TOK:
        for seg in chunk_by_tokens(inp, MAX_IN_TOK, MIN_IN_TOK):
            rows.append({"source": source, "domain": domain, "input": seg, "output": out})
    else:
        rows.append({"source": source, "domain": domain, "input": inp, "output": out})
    return rows

def safe_sample(lst, k):
    if k >= len(lst): return list(lst)
    idx = list(range(len(lst))); random.shuffle(idx)
    return [lst[i] for i in idx[:k]]

# --------- Dataset loaders & field mapping ---------
def load_and_normalize(d):
    name   = d["name"]
    subset = d.get("subset")
    split  = d.get("split", "train")
    target = int(d.get("target", 1000))
    domain = d.get("domain", "GEN")

    print(f"\n==> Loading {name} {subset or ''} [{split}] target={target}")
    try:
        ds = load_dataset(name, subset, split=split) if subset else load_dataset(name, split=split)
    except Exception as e:
        print(f"⚠️  Could not load {name}: {e}")
        return []

    rows = []
    pool = safe_sample(list(ds), target)

    for ex in tqdm(pool, desc=name):
        inp, summ = "", ""
        low = name.lower()

        # ---- new stable datasets ----
        if "samsum" in low:
            # dialogue (like class discussion)
            inp  = ex.get("dialogue") or ""
            summ = ex.get("summary") or ""

        elif "multi_news" in low:
            # multi-doc news → long-form summary
            inp  = ex.get("document") or ""
            summ = ex.get("summary") or ""

        elif "wikihow" in low:
            # instructional tone
            inp  = ex.get("text") or ex.get("article") or ""
            summ = ex.get("headline") or ex.get("summary") or ex.get("title") or ""

        elif "cnn_dailymail" in low:
            # broad-topic summarization
            inp  = ex.get("article") or ""
            summ = ex.get("highlights") or ""

        # ---- academic tone / papers ----
        elif "arxiv" in low:
            inp  = ex.get("article") or ex.get("input") or ""
            summ = ex.get("abstract") or ex.get("summary") or ""

        elif "pubmed" in low:
            inp  = ex.get("article") or ex.get("input") or ""
            summ = ex.get("abstract") or ex.get("summary") or ""

        elif "govreport" in low:
            # gov reports; fields vary across mirrors
            inp  = ex.get("report") or ex.get("text") or ex.get("document") or ""
            summ = ex.get("summary") or ex.get("abstract") or ""

        # ---- legacy mappings kept (in case you add them back later) ----
        elif "lecturebank" in low:
            inp  = ex.get("text") or ex.get("content") or ""
            summ = ex.get("summary") or ex.get("abstract") or ""

        elif "tedlium" in low:
            raw = ex.get("text")
            inp  = " ".join(raw) if isinstance(raw, list) else (raw or "")
            summ = "Talk overview. Key points discussed."

        elif "edinburghcstr/ami" in low:
            inp  = ex.get("transcript") or ""
            summ = ex.get("summary") or ""

        elif "qmsum" in low:
            inp  = ex.get("meeting_transcripts") or ex.get("source_dialogue") or ""
            summ = ex.get("general_summary") or ex.get("target") or ""

        else:
            # fallback
            inp  = ex.get("text") or ex.get("document") or ex.get("content") or ""
            summ = ex.get("summary") or ex.get("abstract") or ""

        rows.extend(pack_rows(name, domain, inp, summ))

    return rows

# --------- Split & save ---------
def sha1(s: str) -> str:
    return hashlib.sha1(s.lower().encode()).hexdigest()

def stratified_split(rows):
    # deduplicate by input text
    if cfg_pp["cleaning"].get("deduplicate", True):
        seen, uniq = set(), []
        for r in rows:
            key = sha1(r["input"])
            if key in seen:
                continue
            seen.add(key); uniq.append(r)
        rows = uniq

    random.shuffle(rows)
    by_dom = {}
    for r in rows:
        by_dom.setdefault(r["domain"], []).append(r)

    train, val, test = [], [], []
    tr_p, va_p, te_p = splits["train"], splits["val"], splits["test"]
    for dom, items in by_dom.items():
        n = len(items)
        n_val  = max(1, int(va_p * n))
        n_test = max(1, int(te_p * n))
        train += items[: n - n_val - n_test]
        val   += items[n - n_val - n_test : n - n_test]
        test  += items[n - n_test :]

    return train, val, test, rows

def main():
    all_rows = []
    for d in cfg_ds["datasets"]:
        all_rows += load_and_normalize(d)

    if not all_rows:
        print("❌ No rows prepared. Check dataset names/Internet and try again.")
        return

    train, val, test, deduped = stratified_split(all_rows)

    with jsonlines.open(OUT_DIR / "train.jsonl", "w") as w: w.write_all(train)
    with jsonlines.open(OUT_DIR / "val.jsonl", "w") as w: w.write_all(val)
    with jsonlines.open(OUT_DIR / "test.jsonl", "w") as w: w.write_all(test)

    stats = {
        "total_raw": len(all_rows),
        "total_deduped": len(deduped),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "domains": {d: sum(1 for r in deduped if r["domain"] == d) for d in set(r["domain"] for r in deduped)},
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print("✅ Wrote:", OUT_DIR / "train.jsonl", OUT_DIR / "val.jsonl", OUT_DIR / "test.jsonl")
    print("📊 Stats:", json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
