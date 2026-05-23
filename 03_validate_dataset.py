#!/usr/bin/env python3
import jsonlines, statistics
from pathlib import Path

p = Path("data_prepared")
def read(name):
    with jsonlines.open(p/name) as r: return list(r)

train, val, test = read("train.jsonl"), read("val.jsonl"), read("test.jsonl")

def lens(rows): return [len(x["input"]) for x in rows]
print("sizes:", len(train), len(val), len(test))
for name, rows in [("train",train),("val",val),("test",test)]:
    L = lens(rows)
    print(name, "avg_chars=", int(statistics.mean(L)), "p90=", int(sorted(L)[int(0.9*len(L))-1]))
    bad = [i for i,x in enumerate(rows) if not isinstance(x.get("input",""),str) or not isinstance(x.get("output",""),str)]
    if bad: print(f"⚠️ {name} bad rows:", len(bad))
