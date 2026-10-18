"""Reproduce the reported numbers from the committed results bundle. No GPU, no downloads, seconds on a laptop.

    pip install -r requirements.txt
    python reproduce.py

Reads results/rollouts.jsonl.gz and results/judge.jsonl.gz and prints follow rates, disclosure rates and the
three pre-specified comparisons (P1-P3), then checks them against results/expected.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd

from ssoc import analysis as A
from ssoc.bundle import read

ROOT = Path(__file__).resolve().parent
pd.set_option("display.width", 160)


def numbers(rollouts, judge):
    judged = {r["key"]: r for r in judge}
    cue = A.attach_judge(A.cue_frame(A.build_frame(rollouts)), judged)
    df = A.build_frame(rollouts)
    return cue, df


def summary(cue, df):
    ft, dt = A.follow_table(cue), A.disclosure_table(cue)
    out = {"sham_to_target": round(float(A.sham_target_rate(df)["thinking"]["mean"]), 3)}
    for _, r in ft.iterrows():
        out[f"follow|{r['mode']}|{r['cond']}"] = [int(r.n_followed), int(r.n_eligible)]
    for _, r in dt.iterrows():
        out[f"silent|{r['mode']}|{r['cond']}"] = [int(r.silent_and_steered), int(r.n_followed)]
        out[f"final_rate|{r['mode']}|{r['cond']}"] = float(r.final_rate)
    p3 = A.p3(cue).set_index("cond").mcnemar_p
    out["p3|answer"], out["p3|evidence"] = round(float(p3["answer"]), 3), round(float(p3["evidence"]), 3)
    p2 = A.p2(cue).set_index("cond").mcnemar_p
    out["p2|answer"] = round(float(p2["answer"]), 4)
    out["p1_fisher"] = float(f"{A.p1(cue)['fisher_p']:.3g}")
    return out


def main():
    rollouts = read(ROOT / "results/rollouts.jsonl.gz")
    judge = read(ROOT / "results/judge.jsonl.gz")
    cue, df = numbers(rollouts, judge)
    print(f"{len(rollouts)} rollouts, {len(judge)} judge records\n")
    print("Sham sentence lands on the target:", A.sham_target_rate(df), "\n")
    print("FOLLOW RATE (eligible = clean answer != target)\n", A.follow_table(cue).to_string(index=False), "\n")
    print("DISCLOSURE among followed rollouts\n", A.disclosure_table(cue).to_string(index=False), "\n")
    print("P1 response disclosure, claim vs assertion:", A.p1(cue))
    print("\nP2 trace vs response (paired)\n", A.p2(cue).to_string(index=False))
    print("\nP3 thinking vs direct follow rate (paired)\n", A.p3(cue).to_string(index=False))
    got = summary(cue, df)
    want = json.load(open(ROOT / "results/expected.json"))
    bad = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
    print("\nCHECK vs results/expected.json:", "PASS" if not bad else f"FAIL {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
