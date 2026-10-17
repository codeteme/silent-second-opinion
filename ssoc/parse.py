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
    text = _CLEAN.sub("", final or "")
    for pat in _ANS:
        hits = pat.findall(text)
        if hits:
            return hits[-1]
    return None
