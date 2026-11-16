"""Experimental claim-pair feedback variant for the shared GSM8K runner."""

import json

from runner import ROUNDS, exchange_prompts, main, prompt

LABELS = {"entailment", "contradiction", "neutral"}


def analysis_prompt(question, previous):
    """Ask the shared LLM to extract and compare claims, not solve the problem."""
    return (
        "You are analyzing two independent mathematical solutions. "
        "Extract up to two related pairs of short, verbatim claims, one "
        "from each solution. Prioritize potentially conflicting calculations. "
        "Use entailment when claim A implies claim B, contradiction when both "
        "cannot be true in the given problem, and neutral otherwise. "
        "Do not call unrelated numbers contradictory. Do not choose a winner "
        "or solve the problem. Return only valid JSON in this format:\n"
        '{"pairs":[{"claim_a":"exact quote","claim_b":"exact quote",'
        '"relation":"contradiction","reason":"short explanation"}]}\n'
        'Use {"pairs":[]} if no useful pair can be found.\n\n'
        f"Problem: {question}\n\n"
        f"Solution A:\n{previous[0]}\n\n"
        f"Solution B:\n{previous[1]}"
    )


def normalize(text):
    return " ".join(text.split()).casefold()


def parse_pairs(response, previous):
    """Validate the JSON and ensure each claim is grounded in its own response.

    Return None for invalid analysis, or a list of verified claim pairs.
    Validation checks the format and quotation, not the correctness of labels.
    """
    start, end = response.find("{"), response.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(response[start:end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("pairs"), list):
        return None
    pairs = []
    for pair in data["pairs"][:2]:
        if not isinstance(pair, dict):
            continue
        if not all(isinstance(pair.get(key), str) for key in
                   ("claim_a", "claim_b", "relation", "reason")):
            continue
        a, b = pair["claim_a"].strip(), pair["claim_b"].strip()
        relation = pair["relation"].strip().lower()
        if (not a or not b or relation not in LABELS
                or normalize(a) not in normalize(previous[0])
                or normalize(b) not in normalize(previous[1])):
            continue
        pairs.append({"claim_a": a, "claim_b": b, "relation": relation,
                      "reason": pair["reason"].strip()})
    if data["pairs"] and not pairs:
        return None
    return pairs


def revision_prompts(question, previous, pairs):
    """Use only validated contradiction candidates as additional feedback."""
    if pairs is None:
        return exchange_prompts(question, previous)
    conflicts = [pair for pair in pairs if pair["relation"] == "contradiction"]
    if conflicts:
        feedback = "\n".join(
            f'- Solution A: {pair["claim_a"]}\n'
            f'  Solution B: {pair["claim_b"]}\n'
            f'  Possible conflict: {pair["reason"]}'
            for pair in conflicts
        )
    else:
        feedback = "No grounded contradiction was flagged. Agreement is not proof."
    return [
        prompt(
            question,
            f"Independently revise your solution. You are solver {'AB'[agent]}.\n"
            f"Your prior solution:\n{previous[agent]}\n\n"
            f"Other solver's prior solution:\n{previous[1-agent]}\n\n"
            f"Claim comparison feedback (may be wrong):\n{feedback}\n\n"
            "Check the conflicting calculations against the original problem. "
            "Recompute independently; do not accept feedback just because it "
            "was flagged. Keep your answer if it is justified."
        )
        for agent in range(2)
    ]


def contradiction_steps(question, initial):
    """Run five simultaneous exchanges, each preceded by an analysis call."""
    turns = [initial]
    for _ in range(ROUNDS):
        previous = turns[-1]
        analysis, = yield [analysis_prompt(question, previous)]
        pairs = parse_pairs(analysis, previous)
        replies = yield revision_prompts(question, previous, pairs)
        turns.append(replies)
    return turns


if __name__ == "__main__":
    main(steps=contradiction_steps, variant="contradiction-feedback-v1")
