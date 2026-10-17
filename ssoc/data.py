"""Load MedMisBench MedMisQA and draw the item sample."""
import json
import random
import re

from .config import MAX_OPTIONS, SEED

REPOS = ["HongjianZhou/MedMisBench", "AI4HealthResearch/MedMisBench"]
OPT_RX = re.compile(r"^op_?([a-t])$")
INJ_RX = re.compile(r"^inject(?:ion)?_?([a-t])$")
# Image-dependent items: the model cannot see the figure.
FIG_OPT = re.compile(r"^\s*(figure|fig\.?|image|panel|graph|photo(graph)?|picture)\s*[A-T0-9]?\s*$", re.I)
FIG_STEM = re.compile(r"\b(figure|photograph|image|micrograph|picture)s?\b|shown (below|in the)|is shown\b", re.I)


def _clean(v) -> str:
    return str(v).strip().strip('"“”').strip()


def normalise(ex: dict, i: int) -> dict:
    opts = {OPT_RX.match(k).group(1).upper(): _clean(v) for k, v in ex.items()
            if OPT_RX.match(k) and v not in (None, "")}
    inj = {INJ_RX.match(k).group(1).upper(): _clean(v) for k, v in ex.items()
           if INJ_RX.match(k) and v not in (None, "")}
    a = str(ex.get("answer", "")).strip()
    a = a[-1] if re.fullmatch(r"(op_?)?[a-tA-T]", a) else a
    gold = a.upper() if a.upper() in opts else next((k for k, v in opts.items() if v == a), None)
    tgt = ex.get("target_wrong_answer")
    tgt = str(tgt).strip()[-1].upper() if tgt else None
    return {"qid": str(ex.get("id", f"medmisqa_{i}")), "question": _clean(ex["question"]),
            "options": dict(sorted(opts.items())), "gold": gold, "inject": inj, "target_given": tgt,
            "content": ex.get("injection_content"), "provenance": ex.get("injection_provenance")}


def usable(it: dict) -> bool:
    return bool(it["gold"] and 4 <= len(it["options"]) <= MAX_OPTIONS
                and any(k != it["gold"] for k in it["inject"])
                and not any(FIG_OPT.match(v) for v in it["options"].values())
                and not FIG_STEM.search(it["question"]))


def pick_target(it: dict) -> str:
    """The benchmark's own Type-1 target if released, else a seeded wrong option with an injection."""
    wrong = [k for k in it["inject"] if k != it["gold"] and k in it["options"]]
    if it["target_given"] in wrong:
        return it["target_given"]
    return random.Random(f"{SEED}-{it['qid']}").choice(wrong)


def load_items(n: int, out_dir: str = None) -> list:
    from datasets import load_dataset
    raw = None
    for repo in REPOS:
        try:
            raw = load_dataset(repo, "MEDMISQA")["MEDMISQA"]
            print("loaded", repo)
            break
        except Exception as e:                      # noqa: BLE001 - try the next repo id
            print(f"{repo}: {type(e).__name__}: {str(e)[:120]}")
    assert raw is not None, "Could not load MedMisBench; see errors above."
    print("keys:", list(raw[0].keys()))
    pool = [it for it in (normalise(raw[i], i) for i in range(len(raw))) if usable(it)]
    assert pool, "No usable items: schema not recognised. Check the printed keys."
    print(f"{len(pool)} usable items")
    items = random.Random(SEED).sample(pool, n)
    for it in items:
        it["target"] = pick_target(it)
    if out_dir:
        json.dump(items, open(f"{out_dir}/items.json", "w"), indent=1)
    return items
