# Silent Second Opinion

Does MedGemma 1.5 4B admit when misleading context changes its answer? Definitions and the three
pre-specified comparisons are in [`SPEC.md`](SPEC.md). Read that first.

```
SPEC.md                  RQ, operationalization, claims, backlog
ssoc/
  config.py              frozen constants (model ids, conditions, modes, token limits)
  data.py                load MedMisBench MedMisQA, filter, sample, pick targets
  prompts.py             cue text + prompt construction (user turn identical across modes)
  parse.py               split trace/final, parse the chosen letter
  generate.py            batched, resumable greedy generation (prompt-hash cache key)
  judge.py               LLM disclosure judge (Qwen3-14B)
  analysis.py            follow rate, disclosure, P1-P3
  audit.py               blind human validation sheet + Cohen's kappa
notebooks/ssoc_run.ipynb thin runner: 11 cells, no logic
tests/test_core.py       22 pure-logic tests, no GPU
```

**Run locally (tests):** `python -m unittest discover -s tests`

**Run on Colab:** open `notebooks/ssoc_run.ipynb` from GitHub on an L4/A100. Add Colab Secrets `HF_TOKEN` and
`GITHUB_TOKEN` (fine-grained, read-only Contents on this repo). The setup cell clones the repo; outputs go to
`MyDrive/silent_second_opinion/`. Run with `KILL_TEST=True`, then `False`, with `REF` pinned to a commit hash.

Edit nothing in `prompts.py` or `config.py` after generation starts; use a new `RUN_NAME`.
