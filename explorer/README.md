# Explorer

A single-page browser for every question, its four versions (clean, sham, claim, assertion), the
model's reasoning trace and response, and the judge's verdict with the matched words highlighted.
Its counts are computed by the same `ssoc.analysis` code as the poster, so they agree with it.

```
python explorer/build_explorer.py     # needs data_cache/ (generations.jsonl, judge.jsonl, items.json)
open explorer/index.html              # works offline; keep it next to ../poster/fonts for the Duke fonts
```

`index.html` is not committed: it embeds the model outputs and MedMisBench questions (about 10 MB).
Rebuild it locally, or publish it only after confirming MedMisBench's licence allows redistribution.
Keyboard: up/down arrows move between questions. The URL hash stores the current question, mode and version.
