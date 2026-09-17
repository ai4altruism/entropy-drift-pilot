# Paper analysis scripts

The scripts behind the analyses in the preprint that the panel scripts in
`scripts/panel/` do not already cover. Every one recomputes from the stored records
or scan summaries, so none needs a GPU and none can perturb a registered run. They are
preserved close to as-run. The only changes are resolving `src/` from the script's own
location, taking the data directory as an argument, and rewording docstrings that
described where the code was kept before release.

```bash
python scripts/paper/paper_analyses.py      [results-dir] [group ...]
python scripts/paper/registered_family.py   [results-dir]
python scripts/paper/h5_exact_k.py          [results-dir]
python scripts/paper/aurc_baselines.py      [results-dir]
python scripts/paper/violation_magnitude.py [results-dir]
python scripts/paper/label_coupling.py      [trace-dir]
python scripts/paper/refscan_compare.py     <scan-dir-A> <scan-dir-B>
```

`results-dir` defaults to `results` and `trace-dir` to `trace`. They need numpy and
scipy.

| script | paper location | status |
|---|---|---|
| `paper_analyses.py tolerance` | Appendix I, the monotonicity tolerance | pre-specified secondary |
| `paper_analyses.py prefix` | Appendix D, early exit | pre-specified exploratory |
| `paper_analyses.py baselines` | Section 6.1 and Appendix F, confounder control, ROC areas and logistic regression | exploratory |
| `paper_analyses.py holm` | Appendix C, the sixteen-test panel-wide family | exploratory robustness check |
| `registered_family.py` | Section 3.4 and Appendix C, Holm over the registered family | registered, applied late |
| `h5_exact_k.py` | Section 4.5 and Appendix A, H5 at exact coverage | registered, rescored |
| `aurc_baselines.py` | Appendix F, risk-coverage area over the full curve and truncated at 73.7% | exploratory |
| `violation_magnitude.py` | Section 6.1, the violation-magnitude check | exploratory |
| `label_coupling.py` | Section 6.1, the correctness label recomputed from the reference chain | exploratory, traced run |
| `refscan_compare.py` | Section 7, the two temperature scans | diagnostic |

One reading note carries over from `scripts/panel/`: `len(trajectory)` counts prefixes
that yielded an extractable answer, not segmentation units, so the length variable in
the confounder control approximates the original's chain length rather than
reproducing it.
