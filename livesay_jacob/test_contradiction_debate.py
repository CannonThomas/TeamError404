import json
import unittest

from contradiction_debate import (
    contradiction_steps, parse_pairs, revision_prompts,
)
from runner import evaluate_group


class ContradictionTests(unittest.TestCase):
    def setUp(self):
        self.prior = ["2 + 3 = 5\nFINAL: 5", "2 + 3 = 6\nFINAL: 6"]
        self.pair = {"claim_a": "2 + 3 = 5", "claim_b": "2 + 3 = 6",
                     "relation": "contradiction", "reason": "Different sums"}

    def test_grounded_conflict(self):
        pairs = parse_pairs(json.dumps({"pairs": [self.pair]}), self.prior)
        self.assertEqual(pairs, [self.pair])
        for text in revision_prompts("2+3?", self.prior, pairs):
            self.assertIn("Different sums", text)
            self.assertIn("2 + 3 = 5", text)
            self.assertIn("2 + 3 = 6", text)

    def test_invalid_analysis(self):
        self.assertIsNone(parse_pairs("not JSON", self.prior))
        self.assertIsNone(parse_pairs(json.dumps({"pairs": [
            {**self.pair, "claim_a": "unquoted invented claim"}
        ]}), self.prior))
        self.assertIn("Review both prior solutions",
                      revision_prompts("2+3?", self.prior, None)[0])

    def test_neutral_not_treated_as_conflict(self):
        pair = {**self.pair, "relation": "neutral"}
        pairs = parse_pairs(json.dumps({"pairs": [pair]}), self.prior)
        text = revision_prompts("2+3?", self.prior, pairs)[0]
        self.assertNotIn("Different sums", text)
        self.assertIn("No grounded contradiction", text)

    def test_all_rounds_and_call_counts(self):
        sizes = []

        def fake_generate(prompts):
            sizes.append(len(prompts))
            if len(prompts) == 1:
                return ['{"pairs":[]}']
            return ["FINAL: 5"] * len(prompts)

        result, = evaluate_group([("2+3?", "5")], fake_generate,
                                 steps=contradiction_steps)
        self.assertEqual(sizes, [2] + [1, 2] * 5)
        self.assertEqual(result["debate_calls"], 17)
        self.assertEqual(result["answer_completions"], 12)
        self.assertEqual(len(result["debate_responses"]), 6)
        self.assertTrue(result["debate_correct"])


if __name__ == "__main__":
    unittest.main()
