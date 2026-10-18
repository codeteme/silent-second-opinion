"""Build explorer/index.html: a single page for browsing every question, its four versions, the model's
trace and response, and the judge's verdict.

Flags (eligible / followed / disclosed) come from ssoc.analysis, the same code that produced the poster
numbers, so the explorer cannot disagree with them.

    python explorer/build_explorer.py        # reads data_cache/, writes explorer/index.html
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ssoc import analysis as A                      # noqa: E402
from ssoc.judge import load_judged                  # noqa: E402
from ssoc.prompts import cue_text                   # noqa: E402

items = json.load(open(ROOT / "data_cache/items.json"))
recs = [json.loads(line) for line in open(ROOT / "data_cache/generations.jsonl")]
J = load_judged(str(ROOT / "data_cache/judge.jsonl"))
df = A.build_frame(recs)
cue = A.attach_judge(A.cue_frame(df), J).set_index(["qid", "mode", "condition"])


def b(v):
    return None if v is None or v != v else bool(v)


runs = {}
for r in recs:
    k = f"{r['qid']}|{r['mode']}|{r['condition']}"
    run = {"a": r.get("answer"), "tr": r.get("trace") if r["mode"] == "thinking" else None,
           "fi": r.get("final"), "ok": bool(r.get("finished"))}
    idx = (r["qid"], r["mode"], r["condition"])
    if idx in cue.index:
        c = cue.loc[idx]
        run.update(el=bool(c.eligible), fo=bool(c.followed), jt=b(c.j_trace), jf=b(c.j_final))
        for surf, f in (("trace", "qt"), ("final", "qf")):
            rec = J.get(f"{k}|{surf}")
            if rec and rec.get("used"):
                run[f] = rec.get("quote", "")
    runs[k] = run

out_items = []
for it in items:
    out_items.append({"id": it["qid"], "q": it["question"], "opts": it["options"], "gold": it["gold"],
                      "tgt": it["target"], "prov": it.get("provenance"),
                      "cues": {c: cue_text(it, c) for c in ("sham", "evidence", "answer")}})

data = {"items": out_items, "runs": runs, "stats": {"sham": round(float(A.sham_target_rate(df)["thinking"]["mean"]), 3)}}
tpl = (ROOT / "explorer/template.html").read_text()
html = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
(ROOT / "explorer/index.html").write_text(html)
print(f"wrote explorer/index.html: {len(out_items)} questions, {len(runs)} runs, {len(html)/1e6:.1f} MB")
