# Silent Second Opinion

**When misleading context changes a medical model's answer, does it say so?**
A study of MedGemma 1.5 4B on 200 MedMisBench questions: a fabricated claim versus a bare "the correct answer is C",
with the model's reasoning on and off. Both cues move the model to the wrong answer about 4 times in 10, and the
bare assertion is much less often mentioned in the response (62% of followed runs are silent, against 21% for claims).

Temesgen Tewolde, Duke University, AIPI 590 (Explainable AI). Poster, talk and explorer are included.

| Read | For |
|---|---|
| [`SPEC.md`](SPEC.md) | the research question, definitions and the three pre-specified comparisons |
| [`DESIGN.md`](DESIGN.md) | why each design choice was made, validation, limitations, misuse risks, and an erratum |
| [`poster/`](poster/) | the printed poster (as printed) and its build script |
| [`talk/`](talk/) | an 18-slide deck and a 5-minute poster script |
| [`explorer/`](explorer/) | a single page for browsing every question, trace, response and judge verdict |

## Reproduce the numbers in one minute (no GPU)

```
pip install -r requirements.txt
python reproduce.py                      # prints every table and ends with: CHECK vs results/expected.json: PASS
python -m unittest discover -s tests     # 34 tests, including the check above
```

`results/` holds the 1,600 rollouts (parsed answers and flags) and the 962 judge verdicts with the quoted words.
It holds **no MedMisBench question text and no full model responses**: the dataset licence is "other", so those
are regenerated rather than redistributed.

## Rerun everything from scratch (Colab, GPU)

1. Open `notebooks/ssoc_run.ipynb` from GitHub on an L4 or A100.
2. Add Colab Secrets `HF_TOKEN` (needs access to `google/medgemma-1.5-4b-it`, a gated model) and, if the repo is
   private, `GITHUB_TOKEN` (fine-grained, read-only Contents).
3. Pin `REF` to a commit hash. Run with `KILL_TEST=True`, then `False`. Outputs go to `MyDrive/silent_second_opinion/`.
4. Copy `generations.jsonl`, `judge.jsonl` and `items.json` into `data_cache/`, then `python -m ssoc.bundle make`
   to refresh `results/`, or `python explorer/build_explorer.py` for the explorer.

Models and data downloaded at run time: MedGemma 1.5 4B (generation), Qwen3-14B (judge, 4-bit on L4),
MedMisBench `MEDMISQA` from Hugging Face (`HongjianZhou/MedMisBench`). Greedy decoding, seed 590. About 1,600
generations plus about 1,000 judge calls. Edit nothing in `prompts.py` or `config.py` after generation starts; use a new `RUN_NAME`.

## Layout

```
ssoc/
  config.py       frozen constants (model ids, conditions, modes, token limits)
  data.py         load MedMisBench MedMisQA, filter, sample, pick targets
  prompts.py      cue text and prompt construction (user turn identical across modes)
  parse.py        split trace/final, parse the chosen letter
  generate.py     batched, resumable greedy generation (prompt-hash cache key)
  judge.py        LLM disclosure judge (Qwen3-14B)
  analysis.py     follow rate, disclosure, P1-P3
  audit.py        validation sheets and Cohen's kappa
  bundle.py       builds the text-free results/ bundle
reproduce.py      reproduces the reported numbers from results/
results/          rollouts and judge verdicts (text-free) plus expected.json
validation/       LLM-annotator labels used for the judge-agreement check
notebooks/        thin Colab runner, no logic
tests/            pure-logic tests plus the reproduction check
```

## Known limitations (full list in [`DESIGN.md`](DESIGN.md))
One small model, 200 questions, English multiple choice. The judge was checked only against LLM annotators
(κ 0.78 response, 0.76 trace), not human labels. Source framing is not randomized. The prompts contain
deliberately false medical statements: for evaluation only, not clinical advice.

## Note on the printed poster
A parsing bug found after the poster was printed moves a few poster figures by one point; no conclusion changes.
See the erratum in `DESIGN.md`. This README, the talk and the explorer use the corrected numbers.
