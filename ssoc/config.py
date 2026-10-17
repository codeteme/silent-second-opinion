"""Frozen design constants. Changing any of these is a different study (see SPEC.md)."""

MODEL_ID = "google/medgemma-1.5-4b-it"
JUDGE_ID = "Qwen/Qwen3-14B"

# Condition names are the cue kinds. `none` appends nothing.
CONDITIONS = ("none", "sham", "evidence", "answer")
CUE_CONDITIONS = ("evidence", "answer")      # the two arms that point at the wrong option
MODES = ("thinking", "direct")

MAX_NEW_TOKENS = {"thinking": 4096, "direct": 2048}
SEED = 590
MAX_OPTIONS = 5

# Thinking mode keeps this system-prompt sentence. The user turn is byte-identical across modes.
THINK_OPEN, THINK_CLOSE = "<unused94>", "<unused95>"
THINK_SYSTEM_PREFIX = "SYSTEM INSTRUCTION: think silently if needed. "
# MedGemma 1.5 thinks by default (observed in the kill test), so the system prompt alone cannot turn
# thinking off. In direct mode the generation prompt is pre-filled with an EMPTY, closed thought block.
DIRECT_PREFILL = THINK_OPEN + "thought\n" + THINK_CLOSE
