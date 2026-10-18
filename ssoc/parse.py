"""Split a raw generation into (trace, final) and parse the chosen option letter."""
import re

from .config import THINK_CLOSE, THINK_OPEN

_STRIP = re.compile(r"<end_of_turn>|<eos>|<pad>|<bos>")
_CLEAN = re.compile(r"[*_`$]|\\text\{|\\textbf\{")
_ANS = [re.compile(r"(?i:answer)(?:\s+(?i:is))?\s*[:\-]?\s*\(?\s*([A-T])\b"),
        re.compile(r"boxed\{\s*\(?([A-T])\)?\s*\}"),
        re.compile(r"(?i:correct (?:option|choice) is)\s*\(?([A-T])\b")]


def is_finished(raw: str) -> bool:
    return "<end_of_turn>" in raw or "<eos>" in raw


def split_output(raw: str):
    """Returns (trace, final, trace_complete). Direct-mode outputs have an empty trace."""
    raw = _STRIP.sub("", raw)
    if THINK_CLOSE in raw:
        trace, final = raw.split(THINK_CLOSE, 1)
        return trace.replace(THINK_OPEN, "").removeprefix("thought").strip(), final.strip(), True
    if THINK_OPEN in raw:   # opened a thought and never closed it
        return raw.replace(THINK_OPEN, "").removeprefix("thought").strip(), "", False
    return "", raw.strip(), True


def parse_answer(final: str):
    """The option letter the response commits to: the LAST answer statement in the text, whatever its form.

    (An earlier version took the last hit of the first pattern that matched anywhere, so a quoted
    "the correct answer is B" in the middle beat a closing "\\boxed{D}".)"""
    text = _CLEAN.sub("", final or "")
    best = None
    for pat in _ANS:
        for m in pat.finditer(text):
            if best is None or m.start(1) > best[0]:
                best = (m.start(1), m.group(1))
    return best[1] if best else None
