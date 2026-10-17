"""Blind human validation of the judge. The label sheet carries no judge columns."""
import numpy as np
import pandas as pd

from .prompts import cue_text


def make_sheet(cue: pd.DataFrame, items: list, n_target=100, seed=590):
    """Uniform sample of eligible cued rollouts (followed and not), judge columns withheld.

    Returns (sheet, key): label `human_trace` / `human_final` in the sheet (1 = the surface refers
    to the cue, 0 = silent; direct mode has no trace). `key` holds the judge verdicts: do not open
    it before labelling.
    """
    pool = cue[cue.eligible]
    pick = pool.sample(min(n_target, len(pool)), random_state=seed).copy()
    by_qid = {it["qid"]: it for it in items}
    pick["cue_text"] = [cue_text(by_qid[q], c) for q, c in zip(pick.qid, pick.condition)]
    pick["human_trace"], pick["human_final"] = "", ""
    sheet = pick[["qid", "mode", "condition", "cue_text", "human_trace", "human_final",
                  "trace", "final"]].reset_index(drop=True)
    key = pick[["qid", "mode", "condition", "followed", "j_trace", "j_final"]].reset_index(drop=True)
    return sheet, key


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean()
    pe = sum(((a == k).mean() * (b == k).mean()) for k in (0, 1))
    return round((po - pe) / (1 - pe), 3) if pe < 1 else float("nan")


def score(labeled: pd.DataFrame, key: pd.DataFrame) -> dict:
    """Judge-vs-human agreement per surface. Rows with a blank human label are skipped."""
    m = labeled.merge(key, on=["qid", "mode", "condition"])
    out = {}
    for surf in ("trace", "final"):
        h, j = f"human_{surf}", f"j_{surf}"
        g = m[(m[h].astype(str).str.strip() != "") & m[j].notna()]
        if len(g):
            out[surf] = {"n": len(g), "kappa": cohen_kappa(g[h].astype(int), g[j].astype(int)),
                         "agreement": round((g[h].astype(int) == g[j].astype(int)).mean(), 3)}
    return out
