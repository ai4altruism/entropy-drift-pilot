#!/usr/bin/env bash
# Drive the confirmatory panel: 8 cells (4 models x 2 datasets), fp16 via vLLM.
#
# Parameters are fixed by OSF registration osf.io/8w2q3. This script chooses the
# ORDER and does the bookkeeping; it never changes a registered parameter.
#
#   ./scripts/run_panel.sh preflight   # checks only, spends nothing
#   ./scripts/run_panel.sh calibrate   # fp16 anchor at n=300, measures throughput
#   ./scripts/run_panel.sh panel       # all 8 cells, resumable
#   ./scripts/run_panel.sh panel qwen7b-gsm8k mistral7b-gsm8k   # named cells only
# NOT `set -e`: a batch that abandons six remaining cells because one died is worse
# than one that records the failure and continues. Cells are independent.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
# vLLM shells out to `ninja` during torch.compile. Invoking .venv/bin/python directly
# does NOT put the venv's console scripts on PATH, so ninja is not found and the engine
# dies with FileNotFoundError after loading the weights. Prepend it explicitly.
export PATH="$PWD/.venv/bin:$PATH"
PY=${PY:-.venv/bin/python}
LOGDIR=logs; mkdir -p "$LOGDIR"

CELLS=(
  qwen7b-gsm8k qwen7b-math500
  mistral7b-gsm8k mistral7b-math500
  llama31-8b-gsm8k llama31-8b-math500
  r1-distill-7b-gsm8k r1-distill-7b-math500
)

say() { printf '\n== %s\n' "$*"; }

preflight() {
  local ok=0
  say "preflight"
  $PY -c 'import torch;assert torch.cuda.is_available();print("  GPU:",torch.cuda.get_device_name(0))' || ok=1
  $PY -c 'import vllm;print("  vLLM:",vllm.__version__)' \
    || { echo "  FAIL: vLLM not importable. Install: pip install -r requirements-vllm.txt"; ok=1; }
  if [ -s "${HF_HOME:-$HOME/.cache/huggingface}/token" ] || [ -n "${HF_TOKEN:-}" ]; then
    echo "  HF token: present"
  else
    echo "  FAIL: no HF token. Llama-3.1-8B-Instruct is gated and will 401."; ok=1
  fi
  # Every cell must carry a pinned revision; an unpinned confirmatory run is not registered.
  for c in "${CELLS[@]}"; do
    grep -qE '^  revision: "[0-9a-f]{40}"' "configs/panel/$c.yaml" \
      || { echo "  FAIL: $c.yaml has no pinned 40-char revision"; ok=1; }
  done
  [ $ok -eq 0 ] && echo "  preflight OK" || { echo "  preflight FAILED"; return 1; }
}

# Expected record count for a cell, from its dataset.
target_for() { case "$1" in *math500) echo 500;; *) echo 1319;; esac; }
records_for() { local f="results/$1/records.jsonl"; [ -f "$f" ] && wc -l < "$f" || echo 0; }

run_cell() {
  local cfg=$1 name; name=$(basename "$cfg" .yaml)
  local log="$LOGDIR/$name.log" t0 t1 rc n target attempt
  target=$(target_for "$name")
  # One retry. --resume continues from records already written, so a retry costs
  # only the work lost since the last flush, not the whole cell.
  for attempt in 1 2; do
    n=$(records_for "$name")
    [ "$n" -ge "$target" ] && break
    say "cell $name (attempt $attempt, $n/$target done)"
    t0=$(date +%s)
    $PY -m entropydrift.run --config "$cfg" --resume >>"$log" 2>&1
    rc=$?
    t1=$(date +%s)
    n=$(records_for "$name")
    printf '   exit=%d after %dm %ds, %s/%s records\n' "$rc" $(( (t1-t0)/60 )) $(( (t1-t0)%60 )) "$n" "$target"
    printf '%s\t%d\t%s\t%d\n' "$name" $((t1-t0)) "$n" "$rc" >> "$LOGDIR/timings.tsv"
    [ "$n" -ge "$target" ] && break
  done
  n=$(records_for "$name")
  if [ "$n" -ge "$target" ]; then echo "   OK $name"; else
    echo "   *** INCOMPLETE $name: $n/$target -- continuing to the next cell"
    FAILED="$FAILED $name"
  fi
}

case "${1:-panel}" in
  preflight) preflight ;;
  calibrate)
    preflight
    say "calibration: fp16 anchor, GSM8K n=300 (configs/smoke.yaml)"
    echo "   Measure this before committing to the panel. It is the plan's first"
    echo "   confirmatory milestone and it converts hours into a real budget."
    run_cell configs/smoke.yaml
    ;;
  panel)
    preflight
    shift || true
    sel=("$@"); [ ${#sel[@]} -eq 0 ] && sel=("${CELLS[@]}")
    FAILED=""
    for c in "${sel[@]}"; do run_cell "configs/panel/$c.yaml"; done
    # Never claim completion the run did not achieve: say which cells finished.
    if [ -n "$FAILED" ]; then
      say "QUEUE FINISHED WITH FAILURES:$FAILED"
      echo "   Re-run those cells; --resume continues from what is already written."
      exit 1
    fi
    say "all requested cells complete. Timings in $LOGDIR/timings.tsv"
    echo "   Next: $PY scripts/fp_reduce.py results/<run-name>   # H5"
    ;;
  *) echo "usage: $0 {preflight|calibrate|panel [cell ...]}"; exit 2 ;;
esac
