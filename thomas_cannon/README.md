# Cannon Thomas | Base debate benchmark

Run from `thomas_cannon/` with Python 3.10–3.12. In Colab select a GPU runtime and upload this folder or clone the repository using your own access. Local CPU execution works but needs enough RAM for float32 weights and will be slower. Generation runs locally without a hosted inference API.

```bash
python -m pip install -r requirements.txt
python -m unittest -v
python runner.py --limit 25 --output results/pilot-v2
```

For Colab, mount Drive and set `--output` to a folder on Drive so results survive a reset. Repeat the same command and output path to resume. The runner saves one JSON file per completed question and a summary after all requested questions finish. Use one process per output folder. A question interrupted mid-run will be rerun. Call counts cover completed questions.

Check `summary.json` for session time, cumulative evaluation time, parse failure rates, and empty outputs before expanding the run. Inspect the saved responses when answers are missing or garbled. The shared settings are Qwen2.5-3B-Instruct and GSM8K pinned to the pilot revisions, the existing ordered test IDs, float16, four questions per batch, and a 512-token cap. Use the same settings for all reported variants. On a CPU, use `--precision float32` for a separate diagnostic run. Use a new output directory for each protocol or variant.

For the full experiment, run `python runner.py --limit 200 --output results/full`.

## Protocol

- The shared IDs are zero-based rows in the GSM8K `main/test` split. This proposed list was sampled without replacement with seed 404. The pilot uses its first 25 IDs.
- The first agent’s initial answer is reused as the single-agent baseline. The two initial prompts are independent, use different instructions, and run in one batch. All calls use greedy decoding and allow up to 512 new tokens per response.
- Each of five exchanges uses both responses from the previous turn, so neither agent sees the other agent's update from the current round.
- Round 0 means the initial answers already agreed. The runner completes all five rounds, even after agreement. Mean time to first agreement is calculated over questions that agree at least once.
- The final round determines correctness. Missing or unparsable final answers and different numeric answers are unresolved and scored as failures. The summary separates these causes. Percent signs are stripped when comparing numeric magnitudes against GSM8K’s unitless answer keys. Wrong agreement and unresolved rates use all evaluated questions as the denominator.
- The baseline uses one completion that is also counted among the 12 debate completions. The combined total counts it once. Each group runs six generation batches for the base model. At batch size four the pilot uses 300 completions across 42 batches, and the full run uses 2,400 completions across 300 batches.
- The exact two-sided McNemar test uses paired baseline-only and debate-only correct counts. Both modes score missing answers as incorrect.
## Standalone software archives

Each teammate’s individual software archive should include its own copy of `runner.py`, `question_ids.json`, `requirements.txt`, and its own setup instructions and variant files. Extract each archive into a separate folder and run it there. Imports from sibling team folders will break when a grader extracts an individual archive.

### Adding a variant

Put your variant in its own Python file and pass a generator function to the shared runner. The runner supplies the question and two initial answers. Yield a list of ready prompts, receive the replies in the same order, and return the list of answer pairs. Add a pair for each completed exchange. Extra critique calls count toward compute but their replies belong outside the answer transcript unless they also contain an agent's final answer. Early stopping returns the pairs collected so far.

```python
from runner import exchange_prompts, main

def my_steps(question, initial):
    turns = [initial]
    for _ in range(5):
        replies = yield exchange_prompts(question, turns[-1])
        turns.append(replies)
    return turns

if __name__ == "__main__":
    main(steps=my_steps, variant="my-variant")
```

Replace the example exchange with your variant's logic. A synchronous variant can still use `main(debate=...)` for single-question debugging. For the team comparison, use the batched interface and record the shared settings. Padding and float16 can change greedy responses when batch shapes differ, so rerun the base with the same settings and report any differences in batch shape or hardware. The pilot had one baseline answer reach the token cap without a final answer. If the team changes prompts or the token cap, apply that decision across variants and rerun the baseline.


## Sources

- [Qwen2.5-3B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [GSM8K dataset card](https://huggingface.co/datasets/openai/gsm8k)

## Colab pilot results

On 25 fixed GSM8K test questions, the single-agent baseline and five-round debate each scored 21/25 (84%). Debate gained 0 percentage points (exact paired McNemar p = 1.00). One baseline output reached the token cap before giving a final answer. All final debate answers parsed, and none of the 300 completions were empty. The float16 Tesla T4 run took 18 minutes 21 seconds of evaluation across 42 generation batches. See [the pilot summary](results/pilot-batch4/summary.json).

## Colab batch pilot

Open `colab_pilot.ipynb` and select a GPU runtime. Use a fresh software ZIP
from the commit you are testing. The notebook runs the files from that ZIP and
prints progress in the output cell. Allow Drive mounting for a dedicated output
folder that survives runtime loss. Keep one notebook tab open while running.

Start with four questions per group (eight prompts per generation batch):

```bash
python -u runner.py --limit 25 --question-batch-size 4 --output results/pilot-batched
```

The base debate performs all five exchanges. Variants using `steps=` share the same batching and checkpointing code.
Each group is saved as one checkpoint with a row for each question. On restart,
completed groups are skipped and any interrupted group is rerun. Keep the same
batch size when resuming. Use a fresh output folder when changing batch size,
model revisions, precision or the token cap. Check the printed GPU and precision
and compare seconds per question and parse failure rates before increasing size.
The loader prints the precision it actually uses. Colab may show dependency
warnings for unrelated preinstalled packages. Check that the runner imports and
the tests pass before proceeding.
