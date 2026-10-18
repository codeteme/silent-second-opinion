# Design decisions, limits and risks

Why the study is built the way it is. For the question and definitions see [SPEC.md](SPEC.md); for how to run it see [README.md](README.md).

## Question
When misleading context changes MedGemma 1.5 4B's answer, do its reasoning trace or its response say so?

## Decisions and reasons

| Decision | Reason |
|---|---|
| Two cues: a fabricated claim and a bare answer assertion | Separates misleading *content* from misleading *framing*. The bare assertion carries no medical claim. |
| Assertion wording matched to each question's provenance (guideline / patient / neutral) | Keeps who-is-speaking constant between the two cues. Wording follows Linzmayer & Elhadad (2026); still to be verified against their Table 6. |
| Sham condition (irrelevant sentence) | Noise floor: any extra sentence can shift a greedy model. Sham lands on the target about 4% of the time. |
| Follow rate only over *eligible* questions (clean answer is not the target, cued run finished) | A model already choosing the target cannot be "moved" to it; an unfinished run is missing data, not a resisted cue. |
| Greedy decoding, fixed seed | Every rollout is a pure function of the prompt, so results are reproducible and cacheable. |
| Direct mode = pre-filled empty thought block | MedGemma thinks by default; this switches the trace off without changing the user turn. |
| LLM judge (Qwen3-14B) must return the sentence number, a verbatim quote and a source phrase | Early versions gave 100% disclosure and false positives. Verbatim-quote checking, the answer-line strip, an answer-cue guard and a source-phrase requirement for evidence cues removed those. |
| Judge cache keyed by a hash of the rubric | A rubric edit invalidates old judgements, so stale labels cannot leak into results. |
| Wilson intervals; exact paired McNemar for within-question comparisons; Fisher plus bootstrap for the unpaired one | Small counts and paired design. |

## Validation
- Judge agreement with blind LLM annotators on clean held-out rollouts: κ 0.78 (response), 0.76 (trace). Weakest for thinking-off responses (κ about 0.44).
- **There are no human labels.** This is the largest open rigor gap.
- Unit tests (`tests/`) cover prompt building, parsing, the judge guards and the statistics.

## Known limitations
- One small model; 200 questions; English multiple choice; greedy decoding only.
- Provenance (authority / neutral / patient) is a per-question label, not randomized, so it is confounded with question difficulty.
- Thinking-off responses still reason visibly, and 10–15% of those rollouts loop.
- The evidence-cue trace-agreement interval is wide (0.44–1.00).
- "Disclosure" means the text refers to the cue as an input. It does not show the cue *caused* the answer.
- "A monitor reading only the response would miss most of these cases" is an inference from disclosure rates. It has not been tested with an actual monitor.

## Misuse and stakeholders
- **Misuse:** the finding that a bare "the correct answer is C" steers a small model almost as well as a fabricated claim describes an easy attack on a deployed system, and the repo contains deliberately false medical statements. They are a benchmark from MedMisBench, are not clinical advice, and should be used for evaluation only.
- **Stakeholders:** clinicians and patients who might rely on a model's explanation; deployers who build monitors on model output; auditors who decide what to inspect (response only, or trace too).
- **Deployment consideration:** providers that hide the reasoning trace remove the surface where most of the disclosure was found.

## Future work
Human labels for the judge; more models and sizes; randomized provenance; a real response-only monitor test; and a larger question set.

## Erratum: answer-parsing bug (found 2026-10-17, after the poster was printed)
The answer parser took the last hit of the *first pattern that matched anywhere*, so a quoted "the correct answer is B" earlier in a response beat a closing `\boxed{D}`. It is now the last answer statement in the text (`ssoc/parse.py`, with a test), and `build_frame` re-parses cached responses.
Two of 1,600 rollouts changed (both bare-assertion runs, both wrongly counted as following the cue). Effect on the printed poster:

| Printed | Corrected |
|---|---|
| Bare assertion followed, thinking on: 34% (59/172) | 34% (58/172) |
| Bare assertion followed, thinking off: 36% (49/138) | 35% (48/138) |
| Assertion mentioned in trace: 60% | 59% |
| Assertion mentioned in response, thinking on: 39% | 38% |
| Assertion mentioned in response, thinking off: 47% | 46% |
| Followed assertions never mentioned in the response: 61% (36/59) | 62% (36/58) |

All p-values keep their reading (0.76, 1.00, 0.002, <0.0001). No conclusion changes. The printed poster PDFs in `poster/` are the as-printed versions; the explorer and this repo use the corrected numbers.
