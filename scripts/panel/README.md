# Panel analysis scripts

The ad-hoc scripts that produced the reported numbers for the September 2026 panel,
kept so every published table has the code that generated it. They were written on the
run host and are preserved close to as-run: the only changes are resolving `src/` from
the script's own location and accepting an optional results directory, so they no longer
depend on being launched from the repo root.

```bash
python scripts/panel/llama_summary.py [results-dir]   # per-cell report, Llama-3.1 pair
python scripts/panel/distilled.py     [results-dir]   # per-cell report, R1-Distill pair
python scripts/panel/recompute.py     [results-dir]   # deviation check, dedup vs raw
python scripts/panel/vllm_check.py    [config.yaml]   # stage 0: vLLM loads and generates
```

`results-dir` defaults to `results`.

Two notes for anyone reading the output:

- `llama_summary.py` and `distilled.py` are the same report over different cells. They
  are kept separate rather than merged because each backs a specific published table.
- **`len(trajectory)` counts prefixes that yielded an extractable answer, not
  segmentation units.** These scripts print it as `steps`, which is what the panel
  wrote up, and reading it as a unit count is a mistake that reached the write-up once.
  Runs from 2026-09-02 record `raw_units`, `prefixes` and `extracted_prefixes`; prefer
  those. See the record-fields table in the top-level README.
- `recompute.py` needs `records.jsonl.raw-with-duplicate` beside `records.jsonl`, which
  exists only for the `mistral7b-math500` cell.
