"""Run a paired GSM8K baseline and five-round debate experiment."""

import argparse
import json
import re
from fractions import Fraction
from pathlib import Path

MODEL = "Qwen/Qwen2.5-3B-Instruct"
DATASET = "openai/gsm8k"
ROUNDS = 5
FORMAT = "Explain your reasoning briefly. End with a separate line: FINAL: <number>"
NUMBER = r"[+-]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*\d+)?"
FINAL = re.compile(r"^\s*FINAL:\s*(" + NUMBER + r")\.?\s*$", re.I)


def number(value):
    cleaned = re.sub(r"[,\s$]", "", value)
    return Fraction(cleaned)


def final_answer(response):
    lines = response.strip().splitlines()
    match = FINAL.fullmatch(lines[-1]) if lines else None
    if not match:
        return None
    try:
        return str(number(match[1]))
    except (ValueError, ZeroDivisionError):
        return None


def ask(generate, question, task):
    return generate(f"{task}\n\nProblem: {question}\n\n{FORMAT}")


def solve_debate(question, generate):
    """Return the initial pair and one pair of responses per exchange round."""
    turns = [[
        ask(generate, question, "Solve independently using a direct calculation."),
        ask(generate, question, "Solve independently using an alternate approach and check your arithmetic."),
    ]]
    for _ in range(ROUNDS):
        previous = turns[-1]
        turns.append([
            ask(generate, question,
                f"Review both prior solutions, then independently revise your answer.\n"
                f"Your prior solution:\n{previous[agent]}\n\n"
                f"Other solver's prior solution:\n{previous[1 - agent]}")
            for agent in range(2)
        ])
    return turns


def evaluate(question, expected, generate, debate=solve_debate):
    """A variant can supply a callable with the same transcript format."""
    baseline = ask(generate, question, "Solve independently using a direct calculation.")
    baseline_answer = final_answer(baseline)
    turns = debate(question, generate)
    answers = [[final_answer(response) for response in pair] for pair in turns]
    agreements = [a is not None and a == b for a, b in answers]
    final = answers[-1][0] if agreements[-1] else None
    return {
        "expected": expected,
        "baseline_response": baseline,
        "baseline_answer": baseline_answer,
        "baseline_correct": baseline_answer == expected,
        "debate_responses": turns,
        "debate_answers": answers,
        "debate_correct": final == expected,
        "first_agreement_round": next((i for i, agreed in enumerate(agreements) if agreed), None),
        "unresolved": final is None,
        "wrong_agreement": final is not None and final != expected,
        "baseline_calls": 1,
        "debate_calls": sum(map(len, turns)),
    }


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize(rows):
    count = len(rows)
    agreements = [row["first_agreement_round"] for row in rows
                  if row["first_agreement_round"] is not None]
    baseline = sum(row["baseline_correct"] for row in rows)
    debate = sum(row["debate_correct"] for row in rows)
    return {
        "questions": count,
        "baseline_accuracy_pct": 100 * baseline / count,
        "debate_accuracy_pct": 100 * debate / count,
        "gain_percentage_points": 100 * (debate - baseline) / count,
        "mean_rounds_until_first_agreement": sum(agreements) / len(agreements) if agreements else None,
        "questions_with_any_agreement": len(agreements),
        "unresolved_rate_pct": 100 * sum(row["unresolved"] for row in rows) / count,
        "wrong_agreement_rate_pct": 100 * sum(row["wrong_agreement"] for row in rows) / count,
        "baseline_calls": sum(row["baseline_calls"] for row in rows),
        "debate_calls": sum(row["debate_calls"] for row in rows),
        "total_model_calls": sum(row["baseline_calls"] + row["debate_calls"] for row in rows),
    }


def load_generator(revision, max_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision, torch_dtype=dtype)
    model = model.to(device).eval()

    def generate(prompt):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("results/pilot"))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--model-revision")
    parser.add_argument("--dataset-revision")
    args = parser.parse_args()
    ids = json.loads(Path(__file__).with_name("question_ids.json").read_text(encoding="utf-8"))
    if not 1 <= args.limit <= len(ids) or args.max_tokens < 1:
        parser.error(f"Use a limit between 1 and {len(ids)} and a positive token limit.")
    ids = ids[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "run.json"

    from datasets import load_dataset
    from huggingface_hub import HfApi

    previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else None
    api = HfApi()
    config = {
        "model": MODEL,
        "model_revision": args.model_revision or (previous["model_revision"] if previous else api.model_info(MODEL).sha),
        "dataset": DATASET,
        "dataset_revision": args.dataset_revision or (previous["dataset_revision"] if previous else api.dataset_info(DATASET).sha),
        "question_ids": ids,
        "max_tokens": args.max_tokens,
        "exchange_rounds": ROUNDS,
        "protocol_version": 1,
    }
    if previous is not None and previous != config:
        parser.error("This output directory has a different run configuration. Choose another.")
    if previous is None:
        write_json(manifest, config)

    rows = []
    generate = None
    dataset = None
    for index in ids:
        path = args.output / f"gsm8k-{index:04d}.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            if generate is None:
                dataset = load_dataset(DATASET, "main", split="test", revision=config["dataset_revision"])
                generate = load_generator(config["model_revision"], args.max_tokens)
            item = dataset[index]
            expected = str(number(item["answer"].rsplit("####", 1)[1].strip()))
            row = {"question_id": index, "question": item["question"],
                   **evaluate(item["question"], expected, generate)}
            write_json(path, row)
        rows.append(row)
        print(f"{len(rows)}/{len(ids)} saved", flush=True)
    summary = summarize(rows)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
