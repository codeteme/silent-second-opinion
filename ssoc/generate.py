"""Batched, resumable generation. One greedy rollout per (item, mode, condition)."""
import hashlib
import json
import os
import time

from .config import CONDITIONS, MAX_NEW_TOKENS, MODES
from .parse import is_finished, parse_answer, split_output
from .prompts import build_messages, generation_suffix


def phash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode()).hexdigest()[:10]


def run_key(r: dict) -> str:
    # The prompt hash is in the key: editing a template regenerates instead of mixing old and new.
    return f"{r['qid']}|{r['mode']}|{r['condition']}|{r['phash']}"


def load_records(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    return {run_key(r): r for r in map(json.loads, open(path))}


def generate(items, path, model, processor, batch_size=16, modes=MODES, conds=CONDITIONS):
    """Appends one JSON line per rollout to `path`. Returns only records for the CURRENT prompts."""
    import torch
    tok = processor.tokenizer
    tok.padding_side = "left"
    by_qid = {it["qid"]: it for it in items}

    jobs = []
    for it in items:
        for m in modes:
            for c in conds:
                prompt = (processor.apply_chat_template(build_messages(it, m, c),
                                                        add_generation_prompt=True, tokenize=False)
                          + generation_suffix(m))
                jobs.append({"qid": it["qid"], "mode": m, "condition": c, "prompt": prompt,
                             "phash": phash(f"{prompt}|{MAX_NEW_TOKENS[m]}")})
    current = {run_key(j) for j in jobs}
    done = load_records(path)
    todo = [j for j in jobs if run_key(j) not in done]
    print(f"{len(jobs)} jobs: {len(jobs) - len(todo)} cached, {len(todo)} to run")

    from tqdm.auto import tqdm
    for mode in modes:
        mj = sorted((j for j in todo if j["mode"] == mode), key=lambda j: len(j["prompt"]))
        bar = tqdm(total=len(mj), desc=mode, unit="rollout")
        for b in range(0, len(mj), batch_size):
            batch = mj[b:b + batch_size]
            t0 = time.time()
            enc = tok([j["prompt"] for j in batch], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS[mode], do_sample=False)
            raws = [tok.decode(t, skip_special_tokens=False)
                    for t in out[:, enc["input_ids"].shape[1]:]]
            sec = round((time.time() - t0) / len(batch), 2)
            with open(path, "a") as f:
                for j, raw in zip(batch, raws):
                    it = by_qid[j["qid"]]
                    trace, final, complete = split_output(raw)
                    fin = is_finished(raw)
                    f.write(json.dumps({
                        "qid": j["qid"], "mode": mode, "condition": j["condition"], "phash": j["phash"],
                        "target": it["target"], "gold": it["gold"], "provenance": it["provenance"],
                        "raw": raw, "trace": trace, "final": final, "trace_complete": complete,
                        "finished": fin, "used_thinking": "<unused94>" in raw,
                        "answer": parse_answer(final) if fin else None, "sec_per_rollout": sec}) + "\n")
            bar.update(len(batch))
            bar.set_postfix(sec_per_rollout=sec)
        bar.close()
    return [r for k, r in load_records(path).items() if k in current]
