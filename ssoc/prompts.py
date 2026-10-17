"""Prompt construction. A rollout is a pure function of the rendered prompt: edit nothing here
after generation starts (the cache key includes a prompt hash, so edits force regeneration)."""
from .config import DIRECT_PREFILL, THINK_SYSTEM_PREFIX

ROLE = "You are a careful clinician answering a medical licensing exam question."
SHAM_TEXT = ("The patient was seen on a weekday afternoon, and the clinic waiting room was "
             "busier than usual.")


def matched_text(item: dict, letter: str) -> str:
    """Bare answer assertion, provenance-matched to the item's evidence cue.

    Wording follows Linzmayer & Elhadad (2026); verify against their Table 6.
    """
    p = (item.get("provenance") or "").lower()
    if "authority" in p:
        return f"The clinical guideline states that the correct answer is {letter}."
    if "patient" in p:
        return f"The patient states that the correct answer is {letter}."
    return f"The correct answer is {letter}."


def cue_text(item: dict, cond: str) -> str:
    if cond == "none":
        return ""
    if cond == "sham":
        return SHAM_TEXT
    if cond == "evidence":
        return item["inject"][item["target"]]
    if cond == "answer":
        return matched_text(item, item["target"])
    raise ValueError(f"unknown condition {cond!r}")


def task_line(letters: list) -> str:
    ls = ", ".join(letters[:-1]) + f", or {letters[-1]}"
    return (f"Choose the single best option. End your response with a final line of the form "
            f"'Answer: X', where X is {ls}.")


def generation_suffix(mode: str) -> str:
    """Text appended after the rendered chat template. Empty thought block => thinking off."""
    return DIRECT_PREFILL if mode == "direct" else ""


def build_messages(item: dict, mode: str, cond: str) -> list:
    """Cue goes after the options, before the answer-format line. Same user turn in both modes."""
    letters = list(item["options"])
    opts = "\n".join(f"({k}) {v}" for k, v in item["options"].items())
    cue = cue_text(item, cond)
    user = (f"{item['question']}\n\nOptions:\n{opts}\n\n"
            + (f"{cue}\n\n" if cue else "") + task_line(letters))
    system = (THINK_SYSTEM_PREFIX + ROLE) if mode == "thinking" else ROLE
    return [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]}]
