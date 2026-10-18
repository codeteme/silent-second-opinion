"""Slim, text-free results bundle: enough to reproduce every reported number without a GPU.

    python -m ssoc.bundle make      # data_cache/ -> results/   (needs the full run outputs)

Holds each rollout's parsed answer and flags, and the judge's verdicts with the quoted words. It does
NOT hold MedMisBench question text or the models' full traces/responses (the dataset licence is "other";
the full texts regenerate with the notebook).
"""
import gzip
import json
import sys
from pathlib import Path

from .judge import load_judged
from .parse import parse_answer

ROOT = Path(__file__).resolve().parent.parent
KEEP = ("qid", "mode", "condition", "target", "gold", "provenance", "answer", "finished", "used_thinking",
        "trace_complete")


def _write(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make(src=ROOT / "data_cache", out=ROOT / "results"):
    out.mkdir(exist_ok=True)
    rows = []
    for line in open(src / "generations.jsonl"):
        r = json.loads(line)
        if r.get("answer") is not None and r.get("finished"):
            r["answer"] = parse_answer(r["final"])           # current parser, not the cached one
        rows.append({k: r.get(k) for k in KEEP})
    _write(out / "rollouts.jsonl.gz", rows)
    judged = load_judged(str(src / "judge.jsonl"))
    _write(out / "judge.jsonl.gz", [{k: v.get(k) for k in ("key", "used", "quote", "source_phrase", "rules")}
                                    for v in judged.values()])
    print(f"wrote {len(rows)} rollouts and {len(judged)} judge records to {out}")


if __name__ == "__main__":
    if sys.argv[1:] == ["make"]:
        make()
    else:
        print(__doc__)
