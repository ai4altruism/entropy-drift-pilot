"""Stage 0: prove vLLM loads a model and generates, through the harness's own path."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from entropydrift.config import load_config
from entropydrift.backends import make_backend

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "configs/panel/qwen7b-gsm8k.yaml"))
print(f"backend={cfg.backend} model={cfg.model.name}")
print(f"revision={cfg.model.revision}")
print(f"quantization={cfg.model.quantization!r} (must be 'none' for fp16)")

t0 = time.time()
be = make_backend(cfg)
print(f"model loaded in {time.time()-t0:.0f}s")

q = "Natalia sold clips to 48 friends in April, and half as many in May. How many did she sell altogether?"
t0 = time.time()
chain = be.reference_chain(q)
t1 = time.time()
print(f"\nreference_chain: {len(chain)} chars in {t1-t0:.1f}s")
print("  first 200 chars:", repr(chain[:200]))

t0 = time.time()
comps = be.continue_from(q, "", 5)
t1 = time.time()
print(f"\ncontinue_from n=5: {len(comps)} completions in {t1-t0:.1f}s")
for i, c in enumerate(comps[:2]):
    print(f"  [{i}] {len(c)} chars: {c[:90]!r}")
print("\nSTAGE 0 PASS" if chain and len(comps) == 5 else "\nSTAGE 0 FAIL")
