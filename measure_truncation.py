"""Measure reference-chain truncation against the 4x max_tokens ceiling.

Diagnostic only. Generates reference chains and counts TOKENS. It computes no
entropy, no monotonicity, no correctness: nothing that bears on H1-H5.
"""
import sys, statistics, collections
sys.path.insert(0, "src")
from entropydrift.backends import TransformersBackend
from entropydrift.datasets import load_examples
from entropydrift.config import load_config

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
MODEL = sys.argv[2]
# arg 4 sets sampling max_tokens; reference_chain() generates at 4x that.
# Raise it to lift the ceiling and read the UNCENSORED chain-length distribution.
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 150
CAP = MAXTOK * 4

cfg = load_config("configs/smoke-local.yaml")
cfg.dataset.limit = N
if len(sys.argv) > 3:
    cfg.dataset.name = sys.argv[3]
print("dataset:", cfg.dataset.name)
examples = load_examples(cfg)

be = TransformersBackend(name=MODEL, temperature=0.7, max_tokens=MAXTOK,
                         quantization="4bit", revision="")
tok = be.tokenizer

lens, truncated = [], 0
for i, ex in enumerate(examples):
    chain = be.reference_chain(ex.question)
    n = len(tok.encode(chain, add_special_tokens=False))
    lens.append(n)
    if n >= CAP - 2:
        truncated += 1
    if (i + 1) % 10 == 0:
        print(f"  ...{i+1}/{N}", flush=True)

import json, os
os.makedirs("lengths", exist_ok=True)
tag = MODEL.split("/")[-1] + "-" + cfg.dataset.name + "-cap" + str(CAP)
json.dump({"model": MODEL, "dataset": cfg.dataset.name, "cap": CAP, "lengths": lens},
          open(f"lengths/{tag}.json", "w"))
print("  raw lengths ->", f"lengths/{tag}.json")
lens.sort()
print(f"\nMODEL: {MODEL}")
print(f"  n                 : {len(lens)}")
print(f"  cap (4x max_tokens): {CAP}")
print(f"  TRUNCATED at cap  : {truncated}/{len(lens)}  ({100*truncated/len(lens):.0f}%)")
print(f"  tokens min/med/max: {lens[0]} / {statistics.median(lens):.0f} / {lens[-1]}")
print(f"  mean              : {statistics.mean(lens):.0f}")
def pct(p):
    return lens[min(len(lens)-1, int(round(p/100*len(lens)))-1)]
print("  percentiles p50/p75/p90/p95/p99:", pct(50), pct(75), pct(90), pct(95), pct(99))
buckets = collections.Counter(min(n // 250 * 250, CAP) for n in lens)
print("  histogram (250-tok bins):", {f"{k}": v for k, v in sorted(buckets.items())})
