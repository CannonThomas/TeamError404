# Cannon Thomas | Base debate benchmark

Run from `thomas_cannon/` with Python 3.10–3.12. In Colab select a GPU runtime and upload this folder or clone the repository using your own access. Local CPU execution works but needs enough RAM for float32 weights and will be slower. Generation runs locally without a hosted inference API.

```bash
python -m pip install -r requirements.txt
python runner.py --limit 25 --output results/pilot
```

For Colab, mount Drive and set `--output` to a folder on Drive so results survive a reset. Repeat the same command and output path to resume. The runner saves one JSON file per completed question and a summary after all requested questions finish. Use one process per output folder. A question interrupted mid-run will be rerun. Call counts cover completed questions.

For the full experiment, run `python runner.py --limit 200 --output results/full`. Keep the same model, model revision, dataset revision, ordered question IDs, and token limit across variants. The manifest records those settings. Share it with the team before collecting full results. Use `--model-revision` and `--dataset-revision` with the commit hashes in the manifest to pin new runs to those revisions.

## Protocol

- The shared IDs are zero-based rows in the GSM8K `main/test` split. This proposed list was sampled without replacement with seed 404. The pilot uses its first 25 IDs.
- The baseline makes one call. Two independent initial responses use different instructions and share no responses until the exchange. The first agent uses the baseline prompt. All calls use greedy decoding.
- Each of five exchanges uses both responses from the previous turn, so neither agent sees the other agent's update from the current round.
- Round 0 means the initial answers already agreed. The runner completes all five rounds, even after agreement. Mean time to first agreement is calculated over questions that agree at least once.
- The final round determines correctness. Missing or unparsable final answers and different numeric answers are unresolved and scored as failures. Wrong agreement and unresolved rates use all evaluated questions as the denominator.
- Each question costs 1 baseline call and 12 debate calls. The default pilot costs 325 calls, and the full run costs 2,600.
- Variant authors can use `evaluate(question, expected, generate, debate=...)` with a debate function returning a transcript of response pairs. Keep variants in their respective member folders.

## Sources

- [Qwen2.5-3B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [GSM8K dataset card](https://huggingface.co/datasets/openai/gsm8k)

The runner records actual outputs when executed. Benchmark results are pending.
