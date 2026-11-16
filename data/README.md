# GSM8K data

The runner downloads the [GSM8K main test split](https://huggingface.co/datasets/openai/gsm8k) from Hugging Face at revision `740312add88f781978c0658806c59bc2815b9866`. GSM8K is distributed under the MIT license. The test split contains 1,319 questions.

The shared list contains 200 zero-based test-row IDs in fixed order. The pilot uses the first 25. See [question_ids.json](../thomas_cannon/question_ids.json).

From `thomas_cannon/`, install the dependencies with `python -m pip install -r requirements.txt` and run `python runner.py`. The runner downloads the pinned split on first use. The setup and Colab instructions are in the [runner README](../thomas_cannon/README.md).
