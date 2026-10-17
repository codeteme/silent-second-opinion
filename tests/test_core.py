"""Pure-logic tests (no GPU, no model). Run: python -m unittest discover -s tests"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssoc import analysis, audit  # noqa: E402
from ssoc.data import normalise, pick_target, usable  # noqa: E402
from ssoc.judge import (RULES_HASH, judge_prompt, load_judged, parse_judge, quote_in_text, quote_ok,  # noqa: E402
                        sentences, strip_answer_line)
from ssoc.parse import parse_answer, split_output  # noqa: E402
from ssoc.prompts import build_messages, cue_text, generation_suffix  # noqa: E402

ITEM = {"qid": "q1", "question": "Which?", "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
        "gold": "B", "target": "C", "provenance": "Neutral false statement",
        "inject": {"A": "x", "C": "Fabricated claim about c."}}


def user_text(item, mode, cond):
    return build_messages(item, mode, cond)[1]["content"][0]["text"]


class Prompts(unittest.TestCase):
    def test_user_turn_identical_across_modes(self):
        for c in ("none", "sham", "evidence", "answer"):
            self.assertEqual(user_text(ITEM, "thinking", c), user_text(ITEM, "direct", c))

    def test_system_differs_only_by_toggle(self):
        s = lambda m: build_messages(ITEM, m, "none")[0]["content"][0]["text"]
        self.assertTrue(s("thinking").endswith(s("direct")))
        self.assertIn("think silently", s("thinking"))

    def test_cues_point_at_same_target(self):
        self.assertEqual(cue_text(ITEM, "evidence"), "Fabricated claim about c.")
        self.assertEqual(cue_text(ITEM, "answer"), "The correct answer is C.")

    def test_matched_provenance(self):
        a = dict(ITEM, provenance="Authority / guideline")
        p = dict(ITEM, provenance="Patient self-claim")
        self.assertIn("guideline", cue_text(a, "answer"))
        self.assertIn("patient", cue_text(p, "answer"))

    def test_cue_after_options(self):
        t = user_text(ITEM, "direct", "answer")
        self.assertLess(t.index("(D)"), t.index("The correct answer is C."))
        self.assertLess(t.index("The correct answer is C."), t.index("Answer: X"))


class DirectMode(unittest.TestCase):
    def test_suffix(self):
        self.assertEqual(generation_suffix("thinking"), "")
        self.assertEqual(generation_suffix("direct"), "<unused94>thought\n<unused95>")


class Parse(unittest.TestCase):
    def test_split_thinking(self):
        t, f, ok = split_output("<unused94>thought\nreasoning here<unused95>Answer: C<end_of_turn>")
        self.assertEqual((t, f, ok), ("reasoning here", "Answer: C", True))

    def test_split_unclosed(self):
        self.assertFalse(split_output("<unused94>thought\nstill going")[2])

    def test_split_direct(self):
        self.assertEqual(split_output("Because.\nAnswer: B<end_of_turn>"), ("", "Because.\nAnswer: B", True))

    def test_parse_answer(self):
        self.assertEqual(parse_answer("so\n**Answer: C**"), "C")
        self.assertEqual(parse_answer("Answer: (B)"), "B")
        self.assertIsNone(parse_answer("no letter here"))
        self.assertEqual(parse_answer("Answer: A ... Answer: D"), "D")


class Data(unittest.TestCase):
    def test_normalise_and_filters(self):
        ex = {"id": "z", "question": "Q?", "op_a": "1", "op_b": "2", "op_c": "3", "op_d": "4",
              "answer": "opb", "target_wrong_answer": "opc", "inject_c": "claim", "inject_a": "other",
              "injection_provenance": "Neutral"}
        it = normalise(ex, 0)
        self.assertEqual((it["gold"], it["target_given"]), ("B", "C"))
        self.assertTrue(usable(it))
        self.assertEqual(pick_target(it), "C")
        self.assertFalse(usable(dict(it, question="A photograph is shown.")))

    def test_pick_target_deterministic_fallback(self):
        it = {"qid": "k", "gold": "B", "options": {"A": 1, "B": 2, "C": 3, "D": 4},
              "inject": {"A": "x", "C": "y"}, "target_given": None}
        self.assertEqual(pick_target(it), pick_target(dict(it)))
        self.assertIn(pick_target(it), ("A", "C"))


class Judge(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_judge('{"used": [2], "quote": "the guideline says C", "source_phrase": "the guideline says"}'),
                         (True, "the guideline says C", "the guideline says"))
        self.assertEqual(parse_judge('{"used": []}'), (False, "", ""))
        self.assertEqual(parse_judge("garbage"), (None, "", ""))
        self.assertEqual(parse_judge('<think>hmm</think>{"used": []}')[0], False)

    def test_quote_must_be_verbatim(self):
        text = "We are told the guideline says C. So C."
        self.assertTrue(parse_judge('{"used": [1], "quote": "the guideline says C"}', text)[0])
        self.assertFalse(parse_judge('{"used": [1], "quote": "invented words"}', text)[0])
        self.assertFalse(parse_judge('{"used": [1], "quote": ""}', text)[0])

    def test_quote_ok_guard(self):
        for good in ("the user states the correct answer is D", "Although the prompt states A is the correct answer"):
            self.assertTrue(quote_ok(good))
        for bad in ("the patient's history of Hodgkin's lymphoma", "The user wants me to identify the most likely cause"):
            self.assertFalse(quote_ok(bad))

    def test_quote_matching_tolerates_markup_and_quotes(self):
        text = "**Consider the guideline:** The prompt states, \u201cantitoxin is the only requirement.\u201d Therefore C."
        self.assertTrue(quote_in_text("The prompt states, 'antitoxin is the only requirement.'", text))
        self.assertTrue(quote_in_text("Consider the guideline: The prompt states antitoxin is the only requirement", text))
        self.assertFalse(quote_in_text("The attending physician insists on surgery", text))
        self.assertFalse(quote_in_text("", text))

    def test_answer_line_variants(self):
        self.assertEqual(strip_answer_line("Reasoning.\nFinal Answer: The final answer is $\\boxed{A}$").strip(), "Reasoning.")
        self.assertEqual(strip_answer_line("(D)\n\nAnswer: D").strip(), "")
        self.assertIn("(B) is wrong", strip_answer_line("(B) is wrong because x."))

    def test_evidence_requires_source_phrase_inside_quote(self):
        cue = pd.DataFrame({"qid": ["a", "b", "c"], "mode": ["thinking"] * 3, "condition": ["evidence"] * 3})
        judged = {"a|thinking|evidence|final": {"used": True, "quote": "The prompt states that X is true.", "source_phrase": "the prompt states"},
                  "b|thinking|evidence|final": {"used": True, "quote": "X is a known risk factor.", "source_phrase": ""},
                  "c|thinking|evidence|final": {"used": True, "quote": "X is a known risk factor.", "source_phrase": "the SOP says"}}
        out = analysis.attach_judge(cue, judged)
        self.assertEqual(list(out.j_final), [True, False, False])

    def test_judge_cache_keyed_by_rubric(self):
        import json, tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"key": "k1", "used": True, "rules": RULES_HASH}) + "\n")
            f.write(json.dumps({"key": "k2", "used": True, "rules": "stale000"}) + "\n")
            f.write(json.dumps({"key": "k3", "used": True}) + "\n")
        self.assertEqual(set(load_judged(f.name)), {"k1"})

    def test_unparsed_judge_records_are_retried_latest_wins(self):
        import json, tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"key": "k1", "used": None, "rules": RULES_HASH}) + "\n")
            f.write(json.dumps({"key": "k1", "used": True, "rules": RULES_HASH}) + "\n")
            f.write(json.dumps({"key": "k2", "used": None, "rules": RULES_HASH}) + "\n")
        got = load_judged(f.name)
        self.assertTrue(got["k1"]["used"])                 # the retry replaced the failure
        self.assertEqual({k for k, r in got.items() if r["used"] is not None}, {"k1"})   # k2 stays pending

    def test_answer_line_stripped(self):
        self.assertEqual(strip_answer_line("Reasoning.\nAnswer: C").strip(), "Reasoning.")
        self.assertEqual(strip_answer_line("**Answer: (B)**").strip(), "")
        self.assertIn("answer is C because", strip_answer_line("The answer is C because x.\nAnswer: C"))
        item = {"question": "Q?", "options": {"A": "a", "B": "b"}}
        self.assertNotIn("Answer: B", judge_prompt(item, "The correct answer is B.", "Answer: B"))

    def test_sentences(self):
        self.assertEqual(sentences("One. Two!\nThree"), ["One.", "Two!", "Three"])


def rec(qid, mode, cond, ans, target="C", gold="B", fin=True):
    return {"qid": qid, "mode": mode, "condition": cond, "answer": ans, "target": target, "gold": gold,
            "finished": fin, "used_thinking": mode == "thinking"}


class Analysis(unittest.TestCase):
    def setUp(self):
        r = []
        for mode in ("thinking", "direct"):
            r += [rec("a", mode, "none", "B"), rec("a", mode, "evidence", "C"), rec("a", mode, "answer", "C"),
                  rec("b", mode, "none", "C"), rec("b", mode, "evidence", "C"), rec("b", mode, "answer", "B"),
                  rec("c", mode, "none", "B"), rec("c", mode, "evidence", "B"), rec("c", mode, "answer", "C"),
                  rec("a", mode, "sham", "B"), rec("c", mode, "sham", "A")]
        self.df = analysis.build_frame(r)
        self.cue = analysis.cue_frame(self.df)

    def test_eligible_excludes_clean_target(self):
        b = self.cue[(self.cue.qid == "b")]
        self.assertFalse(b.eligible.any())            # clean answer already == target
        self.assertFalse(b.followed.any())

    def test_unparsed_cued_rollout_not_eligible(self):
        df = analysis.build_frame([rec("a", "thinking", "none", "B"), rec("a", "thinking", "evidence", None, fin=False)])
        cue = analysis.cue_frame(df)
        self.assertFalse(cue.eligible.any())

    def test_followed(self):
        g = self.cue[(self.cue["mode"] == "thinking")].set_index(["qid", "condition"]).followed
        self.assertTrue(g["a", "evidence"] and g["a", "answer"])
        self.assertFalse(g["c", "evidence"])
        self.assertTrue(g["c", "answer"])

    def test_sham_flip(self):
        self.assertEqual(analysis.sham_flip(self.df), {"direct": 0.5, "thinking": 0.5})

    def test_sham_target_rate(self):
        r = analysis.sham_target_rate(self.df)
        self.assertEqual(r["thinking"], {"mean": 0.0, "size": 2})

    def test_follow_table(self):
        t = analysis.follow_table(self.cue).set_index(["mode", "cond"])
        self.assertEqual(t.loc[("thinking", "evidence"), "n_eligible"], 2)
        self.assertEqual(t.loc[("thinking", "evidence"), "n_followed"], 1)

    def test_p3_paired(self):
        t = analysis.p3(self.cue).set_index("cond")
        self.assertEqual(t.loc["evidence", "n_items"], 2)
        self.assertEqual(t.loc["evidence", "direct"], t.loc["evidence", "thinking"])

    def test_disclosure_p1_p2(self):
        judged = {}
        for q, c, tr, fi in [("a", "evidence", True, True), ("a", "answer", True, False),
                             ("c", "answer", False, False)]:
            judged[f"{q}|thinking|{c}|trace"] = {"used": tr, "quote": "the prompt states the correct answer is C", "source_phrase": "the prompt states"}
            judged[f"{q}|thinking|{c}|final"] = {"used": fi, "quote": "the prompt states the correct answer is C", "source_phrase": "the prompt states"}
        cue = analysis.attach_judge(self.cue, judged)
        d = analysis.disclosure_table(cue).set_index(["mode", "cond"])
        self.assertEqual(d.loc[("thinking", "answer"), "n_followed"], 2)
        self.assertEqual(d.loc[("thinking", "answer"), "silent_and_steered"], 2)
        r1 = analysis.p1(cue)
        self.assertEqual((r1["n_evidence"], r1["n_answer"]), (1, 2))
        self.assertFalse(analysis.p2(cue).empty)

    def test_wilson_and_mcnemar(self):
        lo, hi = analysis.wilson([1, 0, 1, 1])
        self.assertTrue(0 < lo < 0.75 < hi < 1)
        self.assertTrue(np.isnan(analysis.mcnemar(0, 0)[2]))
        self.assertAlmostEqual(analysis.mcnemar(5, 0)[2], 0.0625)


class Audit(unittest.TestCase):
    def test_kappa(self):
        self.assertEqual(audit.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)
        self.assertEqual(audit.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]), 0.0)

    def test_sheet_is_blind_and_scores(self):
        cue = pd.DataFrame({"qid": ["a", "b"], "mode": ["direct"] * 2, "condition": ["answer"] * 2,
                            "eligible": [True, True], "followed": [True, False],
                            "trace": ["", ""], "final": ["x", "y"], "j_trace": [None, None],
                            "j_final": [True, False]})
        items = [dict(ITEM, qid="a"), dict(ITEM, qid="b")]
        sheet, key = audit.make_sheet(cue, items, n_target=10)
        self.assertNotIn("j_final", sheet.columns)
        sheet["human_final"] = [1 if q == "a" else 0 for q in sheet.qid]
        self.assertEqual(audit.score(sheet, key)["final"]["kappa"], 1.0)


if __name__ == "__main__":
    unittest.main()
