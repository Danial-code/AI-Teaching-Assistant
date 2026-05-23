import os
import json
import random
import hashlib
from datasets import load_dataset

OUT_DIR = "data_lecture"
os.makedirs(OUT_DIR, exist_ok=True)

# Reproducibility (important for thesis / experiments)
SEED = 42


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean(text):
    text = (text or "").replace("\n", " ").strip()
    return " ".join(text.split())


def row_hash(inp: str, out: str) -> str:
    return hashlib.sha256((inp + "\n" + out).encode("utf-8")).hexdigest()


def load_and_collect(name: str, in_field: str, out_field: str, limit: int, seen: set[str]):
    """Load a HF dataset split, shuffle deterministically, filter, dedupe, and return rows."""
    print(f"→ Loading {name} ...")
    ds = load_dataset(name, split="train")

    # Avoid order bias: shuffle before selecting a subset
    ds = ds.shuffle(seed=SEED)

    if len(ds) > limit:
        ds = ds.select(range(limit))

    rows = []
    kept = 0
    for ex in ds:
        src = clean(ex.get(in_field, ""))
        tgt = clean(ex.get(out_field, ""))

        # Basic quality filters
        if len(src) <= 200 or len(tgt) <= 20:
            continue

        key = row_hash(src, tgt)
        if key in seen:
            continue
        seen.add(key)

        rows.append({
            "input": src,
            "output": tgt,
            "source": name,
        })
        kept += 1

    print(f"  ✅ kept {kept} samples")
    return rows


print("📘 Loading datasets...")

datasets_info = [
    ("ccdv/arxiv-summarization", "article", "abstract", 1500),
    ("ccdv/pubmed-summarization", "article", "abstract", 1500),
    ("MeetingBank/MeetingBank", "transcript", "summary", 1000),
]

all_rows: list[dict] = []
seen_hashes: set[str] = set()

for name, in_field, out_field, limit in datasets_info:
    try:
        rows = load_and_collect(name, in_field, out_field, limit, seen_hashes)
        all_rows.extend(rows)
    except Exception as e:
        print(f"⚠️ Could not load {name}: {e}")

# Reproducible shuffle of the combined pool
random.Random(SEED).shuffle(all_rows)

print(f"✅ Loaded {len(all_rows)} total examples (deduped)")

# --- Split into train/val/test (80/10/10) ---
n = len(all_rows)
train = all_rows[: int(0.8 * n)]
val = all_rows[int(0.8 * n) : int(0.9 * n)]
test = all_rows[int(0.9 * n) :]

write_jsonl(os.path.join(OUT_DIR, "train.jsonl"), train)
write_jsonl(os.path.join(OUT_DIR, "val.jsonl"), val)
write_jsonl(os.path.join(OUT_DIR, "test.jsonl"), test)

# Metadata for reporting / thesis reproducibility
meta = {
    "seed": SEED,
    "total": n,
    "train": len(train),
    "val": len(val),
    "test": len(test),
    "sources": {name: sum(1 for r in all_rows if r["source"] == name) for name, *_ in datasets_info},
}
with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"📁 Saved {len(train)} train, {len(val)} val, {len(test)} test samples to {OUT_DIR}/")
print(f"🧾 Wrote metadata to {OUT_DIR}/meta.json")
