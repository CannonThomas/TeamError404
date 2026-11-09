import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import runner


class RunnerTests(unittest.TestCase):
    def test_answer_formats(self):
        examples = {
            "**FINAL: 18**": "18",
            "FINAL: 18 dollars": "18",
            "FINAL: 50%": "50",
            "FINAL: 18\nThanks": "18",
            "FINAL: $1,200.00": "1200",
            "FINAL: 4/2": "2",
            "FINAL: 1/0": None,
            "FINAL: 12,3": None,
            "FINAL: 18 or 19": None,
            "": None,
        }
        for response, expected in examples.items():
            with self.subTest(response=response):
                self.assertEqual(runner.final_answer(response), expected)

    def test_batched_exchanges_and_baseline_reuse(self):
        prompts = []

        def generate(pair):
            prompts.append(pair)
            return [f"agent {agent}, batch {len(prompts)}\nFINAL: 2" for agent in range(2)]

        row = runner.evaluate("1+1?", "2", generate)
        self.assertEqual(len(prompts), 6)
        self.assertTrue(all(len(pair) == 2 for pair in prompts))
        for round_number, pair in enumerate(prompts[1:], 1):
            for agent, prompt in enumerate(pair):
                self.assertIn(f"Other solver's prior solution:\nagent {1-agent}, batch {round_number}", prompt)
        self.assertEqual(row["baseline_response"], row["debate_responses"][0][0])
        self.assertEqual(row["unique_model_completions"], 12)
        self.assertTrue(row["debate_correct"])
        self.assertEqual(row["first_agreement_round"], 0)

    def test_outcomes_and_exact_mcnemar(self):
        def run(final_pair, first="FINAL: 2"):
            batches = iter([[first, "FINAL: 3"]] + [["FINAL: 2"] * 2] * 4 + [final_pair])
            return runner.evaluate("1+1?", "2", lambda _: next(batches))

        wrong = run(["FINAL: 3"] * 2)
        disagree = run(["FINAL: 2", "FINAL: 3"])
        missing = run(["FINAL: 2", ""])
        improved = run(["FINAL: 2"] * 2, first="FINAL: 3")
        self.assertTrue(wrong["wrong_agreement"])
        self.assertTrue(disagree["debate_final_disagreement"])
        self.assertFalse(disagree["debate_final_parse_failure"])
        self.assertTrue(missing["debate_final_parse_failure"])
        self.assertFalse(missing["debate_final_disagreement"])
        self.assertTrue(missing["unresolved"])
        self.assertEqual(missing["empty_completions"], 1)
        summary = runner.summarize([wrong, disagree, missing, improved])
        self.assertEqual(summary["mcnemar_exact"], {
            "baseline_only_correct": 3, "debate_only_correct": 1, "two_sided_p": 0.625,
        })
        self.assertEqual(summary["total_model_calls"], 48)
        self.assertEqual(summary["generation_batches"], 24)
        self.assertEqual(summary["unresolved_rate_pct"], 50)

    def test_resume_skips_completed_question(self):
        api = SimpleNamespace(model_info=lambda _: SimpleNamespace(sha="model-test"),
                              dataset_info=lambda _: SimpleNamespace(sha="dataset-test"))
        sources = {
            "datasets": SimpleNamespace(load_dataset=lambda *a, **k: [
                {"question": "1+1?", "answer": "#### 2"}
            ] * 1319),
            "huggingface_hub": SimpleNamespace(HfApi=lambda: api),
        }
        batches = 0

        def generate(prompts):
            nonlocal batches
            batches += 1
            if batches == 7:
                raise InterruptedError
            return ["FINAL: 2"] * len(prompts)

        with tempfile.TemporaryDirectory() as directory:
            argv = ["runner.py", "--limit", "2", "--output", directory]
            with (patch.dict(sys.modules, sources), patch.object(sys, "argv", argv),
                  patch.object(runner, "load_generator", return_value=generate),
                  contextlib.redirect_stdout(io.StringIO())):
                with self.assertRaises(InterruptedError):
                    runner.main()
                self.assertEqual(len(list(Path(directory).glob("gsm8k-*.json"))), 1)
                runner.main()
                runner.main()
            self.assertEqual(batches, 13)
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(summary["total_model_calls"], 24)
            self.assertEqual(summary["generation_batches"], 12)


if __name__ == "__main__":
    unittest.main()
