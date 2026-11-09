# Cannon Thomas | Base debate benchmark

Run from `thomas_cannon/` with Python 3.10–3.12. In Colab select a GPU runtime and upload this folder or clone the repository using your own access. Local CPU execution works but needs enough RAM for float32 weights and will be slower. Generation runs locally without a hosted inference API.

```bash
python -m pip install -r requirements.txt
python -m unittest -v
python runner.py --limit 25 --output results/pilot-v2
```

For Colab, mount Drive and set `--output` to a folder on Drive so results survive a reset. Repeat the same command and output path to resume. The runner saves one JSON file per completed question and a summary after all requested questions finish. Use one process per output folder. A question interrupted mid-run will be rerun. Call counts cover completed questions.

Check `summary.json` for session time, cumulative evaluation time, parse failure rates, and empty outputs before expanding the run. Inspect the saved responses when answers are missing or garbled. On GPUs with native bfloat16 support, automatic precision uses bfloat16. On older GPUs it uses float16. If float16 outputs are garbled, retry with `--precision float32` in a new output directory, provided the GPU has enough memory. Keep device and precision consistent when resuming or comparing variants.

For the full experiment, run `python runner.py --limit 200 --output results/full`. Keep the same model, model revision, dataset revision, ordered question IDs, and token limit across variants. The manifest records those settings. Share it with the team before collecting full results. Use `--model-revision` and `--dataset-revision` with the commit hashes in the manifest to pin new runs to those revisions.

## Protocol

- The shared IDs are zero-based rows in the GSM8K `main/test` split. This proposed list was sampled without replacement with seed 404. The pilot uses its first 25 IDs.
- The first agent’s initial answer is reused as the single-agent baseline. The two initial prompts are independent, use different instructions, and run in one batch. All calls use greedy decoding and allow up to 512 new tokens per response.
- Each of five exchanges uses both responses from the previous turn, so neither agent sees the other agent's update from the current round.
- Round 0 means the initial answers already agreed. The runner completes all five rounds, even after agreement. Mean time to first agreement is calculated over questions that agree at least once.
- The final round determines correctness. Missing or unparsable final answers and different numeric answers are unresolved and scored as failures. The summary separates these causes. Percent signs are stripped when comparing numeric magnitudes against GSM8K’s unitless answer keys. Wrong agreement and unresolved rates use all evaluated questions as the denominator.
- The baseline uses one completion that is also counted among the 12 debate completions. The combined total counts it once. Each question uses six generation batches. The pilot uses 300 completions across 150 batches, and the full run uses 2,400 completions across 1,200 batches.
- The exact two-sided McNemar test uses paired baseline-only and debate-only correct counts. Both modes score missing answers as incorrect.
## Standalone software archives

Each teammate’s individual software archive should include its own copy of `runner.py`, `question_ids.json`, `requirements.txt`, and its own setup instructions and variant files. Extract each archive into a separate folder and run it there. Imports from sibling team folders will break when a grader extracts an individual archive.

A variant can call `runner.main(debate=my_debate, variant="my-variant")`. The function receives `(question, generate, initial)`, where `generate` takes a list of prompts and returns a list of responses. Return a transcript starting with `initial` and followed by response pairs. Record each variant’s differences and exact run command in its README.

## Sources

- [Qwen2.5-3B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [GSM8K dataset card](https://huggingface.co/datasets/openai/gsm8k)

The runner records actual outputs when executed. Benchmark results are pending.
