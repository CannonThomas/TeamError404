"""Run a paired GSM8K baseline and five-round debate experiment."""

import argparse
import json
import re
from math import comb
from time import perf_counter
from fractions import Fraction
from pathlib import Path

MODEL = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
DATASET = "openai/gsm8k"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
TEAM_PRECISION = "float16"
TEAM_BATCH_SIZE = 4
ROUNDS = 5
FORMAT = "Explain your reasoning briefly. End with a separate line: FINAL: <number>"
NUMBER = r"[+-]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*/\s*\d+)?"
FINAL = re.compile(r"^FINAL:\s*(.*)$", re.I)
NUMERIC_ANSWER = re.compile(
    r"(" + NUMBER + r")(?:\s*%|\s+[a-z]+(?:[- ][a-z]+)*)?\.?", re.I
)


def number(value):
    cleaned = re.sub(r"[,\s$]", "", value)
    return Fraction(cleaned)


def final_answer(response):
    for line in reversed(response.splitlines()):
        marker = FINAL.fullmatch(line.strip().strip("*_`").strip())
        if marker:
            match = NUMERIC_ANSWER.fullmatch(marker[1].strip().strip("*_`").strip())
            if match:
                try:
                    return str(number(match[1]))
                except (ValueError, ZeroDivisionError):
                    return None
            return None
    return None


def prompt(question, task):
    return f"{task}\n\nProblem: {question}\n\n{FORMAT}"


def initial_prompts(question):
    return [
        prompt(question, "Solve independently using a direct calculation."),
        prompt(question, "Solve independently using an alternate approach and check your arithmetic."),
    ]


def exchange_prompts(question, previous):
    return [
        prompt(question,
               f"Review both prior solutions, then independently revise your answer.\n"
               f"Your prior solution:\n{previous[agent]}\n\n"
               f"Other solver's prior solution:\n{previous[1 - agent]}")
        for agent in range(2)
    ]


def debate_steps(question, initial):
    """Yield the prompts for each exchange and return the answer transcript."""
    turns = [initial]
    for _ in range(ROUNDS):
        replies = yield exchange_prompts(question, turns[-1])
        turns.append(replies)
    return turns


def solve_debate(question, generate, initial):
    """Run the base exchange steps using the single-question interface."""
    steps = debate_steps(question, initial)
    try:
        prompts = next(steps)
        while True:
            prompts = steps.send(generate(prompts))
    except StopIteration as finished:
        return finished.value


def generate_batch(generate, prompts):
    responses = generate(prompts)
    if len(responses) != len(prompts):
        raise ValueError("Expected one model response per prompt.")
    return responses


def evaluate_group(items, generate, steps=debate_steps):
    """Batch the next requests from active questions and route their replies."""
    started = perf_counter()
    initial = generate_batch(generate, [
        text for question, _ in items for text in initial_prompts(question)
    ])
    completions = [initial[i:i + 2] for i in range(0, len(initial), 2)]
    active = [
        (index, steps(question, pair.copy()), None)
        for index, ((question, _), pair) in enumerate(zip(items, completions))
    ]
    transcripts = {}
    batches = 1
    while active:
        requests = []
        prompts = []
        for index, session, replies in active:
            try:
                request = session.send(replies)
            except StopIteration as finished:
                turns = finished.value
                if not turns or any(len(pair) != 2 for pair in turns):
                    raise ValueError("A debate must return at least one pair of agent answers.")
                transcripts[index] = turns
                continue
            if not isinstance(request, (list, tuple)) or not request or not all(
                isinstance(text, str) for text in request
            ):
                raise ValueError("A debate step must yield a nonempty list of prompts.")
            requests.append((index, session, len(prompts), len(request)))
            prompts.extend(request)
        if not requests:
            break
        responses = generate_batch(generate, prompts)
        batches += 1
        active = []
        for index, session, offset, count in requests:
            replies = responses[offset:offset + count]
            completions[index].extend(replies)
            active.append((index, session, replies))

    elapsed = perf_counter() - started
    return [
        score(expected, completions[index][0], transcripts[index], completions[index],
              batches if index == 0 else 0, elapsed if index == 0 else 0)
        for index, (_, expected) in enumerate(items)
    ]


def evaluate(question, expected, generate, debate=solve_debate):
    """Run the baseline and a debate variant on the same question."""
    started = perf_counter()
    batches = 0
    completions = []

    def batch(prompts):
        nonlocal batches
        responses = generate_batch(generate, prompts)
        batches += 1
        completions.extend(responses)
        return responses

    initial = batch(initial_prompts(question))
    turns = debate(question, batch, initial)
    return score(expected, initial[0], turns, completions, batches, perf_counter() - started)


def score(expected, baseline, turns, completions, batches, elapsed):
    baseline_answer = final_answer(baseline)
    answers = [[final_answer(response) for response in pair] for pair in turns]
    agreements = [a is not None and a == b for a, b in answers]
    final = answers[-1][0] if agreements[-1] else None
    parse_failure = None in answers[-1]
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
        "baseline_parse_failure": baseline_answer is None,
        "debate_final_parse_failure": parse_failure,
        "debate_final_disagreement": final is None and not parse_failure,
        "wrong_agreement": final is not None and final != expected,
        "baseline_calls": 1,
        "debate_calls": len(completions),
        "unique_model_completions": len(completions),
        "generation_batches": batches,
        "answer_completions": sum(map(len, answers)),
        "parse_failed_completions": sum(answer is None for pair in answers for answer in pair),
        "empty_completions": sum(not response.strip() for response in completions),
        "elapsed_seconds": elapsed,
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
    baseline_only = sum(row["baseline_correct"] and not row["debate_correct"] for row in rows)
    debate_only = sum(row["debate_correct"] and not row["baseline_correct"] for row in rows)
    discordant = baseline_only + debate_only
    tail = sum(comb(discordant, k) for k in range(min(baseline_only, debate_only) + 1))
    completions = sum(row["unique_model_completions"] for row in rows)
    answer_completions = sum(row.get("answer_completions", row["unique_model_completions"]) for row in rows)
    return {
        "questions": count,
        "mcnemar_exact": {
            "baseline_only_correct": baseline_only,
            "debate_only_correct": debate_only,
            "two_sided_p": min(1.0, 2 * tail / (2 ** discordant)),
        },
        "baseline_accuracy_pct": 100 * baseline / count,
        "debate_accuracy_pct": 100 * debate / count,
        "gain_percentage_points": 100 * (debate - baseline) / count,
        "mean_rounds_until_first_agreement": sum(agreements) / len(agreements) if agreements else None,
        "questions_with_any_agreement": len(agreements),
        "unresolved_rate_pct": 100 * sum(row["unresolved"] for row in rows) / count,
        "wrong_agreement_rate_pct": 100 * sum(row["wrong_agreement"] for row in rows) / count,
        "baseline_calls": sum(row["baseline_calls"] for row in rows),
        "debate_calls": sum(row["debate_calls"] for row in rows),
        "total_model_calls": completions,
        "generation_batches": sum(row["generation_batches"] for row in rows),
        "baseline_parse_failure_rate_pct": 100 * sum(row["baseline_parse_failure"] for row in rows) / count,
        "debate_final_parse_failure_rate_pct": 100 * sum(row["debate_final_parse_failure"] for row in rows) / count,
        "debate_final_disagreement_rate_pct": 100 * sum(row["debate_final_disagreement"] for row in rows) / count,
        "completion_parse_failure_rate_pct": 100 * sum(row["parse_failed_completions"] for row in rows) / answer_completions,
        "empty_completions": sum(row["empty_completions"] for row in rows),
        "evaluation_seconds": sum(row["elapsed_seconds"] for row in rows),
    }


def load_generator(revision, max_tokens, precision):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision == "auto":
        precision = ("bfloat16" if torch.cuda.is_bf16_supported(including_emulation=False)
                     else "float16") if device == "cuda" else "float32"
    if precision == "bfloat16" and device == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("This GPU does not support native bfloat16. Use float32 or auto.")
    dtype = getattr(torch, precision)
    print(f"Loading {MODEL} on {device} in {precision}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=revision, torch_dtype=dtype)
    model = model.to(device).eval()

    def generate(prompts):
        chats = [[{"role": "user", "content": text}] for text in prompts]
        formatted = tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, add_special_tokens=False, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
        return tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def main(debate=None, variant="base", *, steps=None):
    if debate is not None and steps is not None:
        raise ValueError("Choose either debate or steps.")
    if debate is None:
        steps = steps or debate_steps
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("results/pilot"))
    parser.add_argument("--question-batch-size", type=int, default=1 if debate is not None else TEAM_BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--precision", choices=("auto", "float32", "float16", "bfloat16"), default=TEAM_PRECISION)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    args = parser.parse_args()
    ids = json.loads(Path(__file__).with_name("question_ids.json").read_text(encoding="utf-8"))
    if not 1 <= args.limit <= len(ids) or args.max_tokens < 1:
        parser.error(f"Use a limit between 1 and {len(ids)} and a positive token limit.")
    if args.question_batch_size < 1:
        parser.error("The question batch size must be positive.")
    if args.question_batch_size > 1 and debate is not None:
        parser.error("Use steps= for cross-question batching with a variant.")
    ids = ids[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "run.json"

    from datasets import load_dataset

    previous = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else None
    config = {
        "model": MODEL,
        "model_revision": args.model_revision,
        "dataset": DATASET,
        "dataset_revision": args.dataset_revision,
        "question_ids": ids,
        "max_tokens": args.max_tokens,
        "base_exchange_rounds": ROUNDS,
        "precision": args.precision,
        "variant": variant,
        "protocol_version": 4,
        "question_batch_size": args.question_batch_size,
    }
    if previous is not None and previous != config:
        parser.error("This output directory has a different run configuration. Choose another.")
    if previous is None:
        write_json(manifest, config)

    rows = []
    generate = None
    dataset = None
    size = args.question_batch_size
    for offset in range(0, len(ids), size):
        group = ids[offset:offset + size]
        filename = f"gsm8k-{group[0]:04d}.json" if size == 1 else f"batch-{offset:04d}.json"
        path = args.output / filename
        if path.exists():
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
            group_rows = [checkpoint] if size == 1 else checkpoint
        else:
            if generate is None:
                dataset = load_dataset(DATASET, "main", split="test", revision=config["dataset_revision"])
                generate = load_generator(config["model_revision"], args.max_tokens, args.precision)
            items = []
            for index in group:
                item = dataset[index]
                expected = str(number(item["answer"].rsplit("####", 1)[1].strip()))
                items.append((item["question"], expected))
            if debate is not None:
                outcomes = [evaluate(*items[0], generate, debate=debate)]
            else:
                outcomes = evaluate_group(items, generate, steps=steps)
            group_rows = [
                {"question_id": index, "question": question, **outcome}
                for index, (question, _), outcome in zip(group, items, outcomes)
            ]
            write_json(path, group_rows[0] if size == 1 else group_rows)
        rows.extend(group_rows)
        print(f"{len(rows)}/{len(ids)} saved", flush=True)
    summary = summarize(rows)
    summary["last_session_seconds"] = perf_counter() - started
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
