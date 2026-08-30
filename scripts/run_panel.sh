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
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
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

run_cell() {
  local cfg=$1 name; name=$(basename "$cfg" .yaml)
  local log="$LOGDIR/$name.log" t0 t1
  say "cell $name"
  t0=$(date +%s)
  # --resume makes an interrupted or spot-preempted cell continue rather than restart.
  $PY -m entropydrift.run --config "$cfg" --resume >>"$log" 2>&1
  t1=$(date +%s)
  local n; n=$(wc -l < "results/$name/records.jsonl" 2>/dev/null || echo 0)
  printf '   done in %dm %ds, %s records -> results/%s\n' $(( (t1-t0)/60 )) $(( (t1-t0)%60 )) "$n" "$name"
  printf '%s\t%d\t%s\n' "$name" $((t1-t0)) "$n" >> "$LOGDIR/timings.tsv"
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
    for c in "${sel[@]}"; do run_cell "configs/panel/$c.yaml"; done
    say "panel complete. Timings in $LOGDIR/timings.tsv"
    echo "   Next: $PY scripts/fp_reduce.py results/<run-name>   # H5"
    ;;
  *) echo "usage: $0 {preflight|calibrate|panel [cell ...]}"; exit 2 ;;
esac
