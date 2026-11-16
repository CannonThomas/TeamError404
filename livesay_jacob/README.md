# Contradiction-feedback debate — experimental prototype

This prototype evaluates whether claim extraction and contradiction detection
improve multi-agent debate accuracy on GSM8K math problems.

## Design

Reuse the team's pinned Qwen2.5-3B-Instruct model, GSM8K test IDs, single-agent
baseline, four-question groups, float16 precision, and 512-token limit.
The initial answers are independent. Before each of five simultaneous
revision rounds, Qwen extracts up to two pairs of exact claims and assigns
entailment, contradiction, or neutral labels. Python validates the structure
and verifies claims are present in the agent outputs. Only contradiction
candidates are supplied as targeted feedback. The agents independently
recalculate; agreement is not itself proof. Invalid analysis falls back to
the shared base exchange. The additional analysis call does not vote on the
answer or see the gold answer.

The three labels are inspired by Williams et al., not by a trained MultiNLI
classifier. No MultiNLI data or fine-tuning is used in this prototype.

## Run on Colab

Extract this ZIP on your computer and open `colab_contradiction.ipynb` in
Google Colab. Select a GPU runtime, run each cell in order, and upload the
original ZIP when prompted. Google Drive stores checkpoint files. Start with
4 questions to check generation, then use a fresh directory for the shared
25-question pilot. The notebook provides both commands.

## Run from an extracted folder

Python 3.10–3.12; a GPU is recommended. From the folder containing runner.py:

```bash
python -m pip install -r requirements.txt
python -m unittest -v
python -u contradiction_debate.py --limit 4 --question-batch-size 4 --output results/smoke-v1
python -u contradiction_debate.py --limit 25 --question-batch-size 4 --output results/pilot-v1
```

Resume with the exact same settings and output path. Use a distinct fresh
directory when changing settings. Use `--limit 200` and a new output path
for the final evaluation. See SHARED_RUNNER_README.md for more details.

## Evaluation

Inspect summary.json and saved batch JSONs for actual accuracy, gain in
percentage points, wrong agreement, unresolved rate, mean rounds to first
agreement, and total model calls. Compare the same 25 test IDs and settings
with the team's base model. Each question requires 17 completions (2
initial + 5 × (1 analysis + 2 revisions)), versus 12 for the base; the
25-question pilot therefore requires 425 completions versus 300. The extra
analysis is counted by the runner, but analysis outputs themselves are not
included in the stored answer transcript. A future improvement is to persist
and manually audit the extracted labels. Quotation validation does not prove
semantic label accuracy. A small pilot is exploratory, not a conclusive
benchmark. The pilot and smoke-test results are stored separately.

## Sources

- Shared runner, question IDs, requirements, and runner tests: Cannon Thomas,
  TeamError404 project repository, supplied by Jacob. Original files retained.
- A. Williams, N. Nangia, and S. R. Bowman, "A Broad-Coverage Challenge Corpus
  for Sentence Understanding through Inference," NAACL-HLT 2018, pp. 1112–1122,
  doi:10.18653/v1/N18-1101. Inspiration: three claim-relation labels.
- Qwen model card: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- GSM8K dataset: https://huggingface.co/datasets/openai/gsm8k
