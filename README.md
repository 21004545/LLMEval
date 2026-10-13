# LLMEval

Sends the same history prompts in English and Mandarin to ChatGPT, Gemini, DeepSeek and Doubao,
translates the Mandarin answers to English, and has a judge LLM score each answer 1–5 on five bias
categories.

## Setup (once)

```
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # Mac/Linux: cp .env.example .env
```

Put the API keys in `.env`. The OpenAI key is needed even if ChatGPT isn't being tested, because it
powers the translator and the bias judge. Never commit `.env`.

Check that everything is wired up (no keys or credit needed):

```
python test_driver.py
```

## Full run, option A: command line (recommended for the 40-row run)

```
python driver.py                                   # all 4 models, sample_data.tsv -> results.tsv
python driver.py --only deepseek-v4-flash          # one model
python driver.py sample_data.tsv out.tsv --no-bias # answers only, no translation or judging
```

It prints a summary of rows per model and status at the end.

## Full run, option B: Streamlit app

```
streamlit run main.py
streamlit run main.py -- my_prompts.tsv   # optional: start with a prompt file already imported
```

Import Prompts → upload `sample_data.tsv` → Process Prompts → tick the models, tick "Evaluate
responses for political bias" → Process. Results are also saved automatically as
`results_<date>_<time>.tsv` in this folder.

## Reading the results

One row per model × prompt × language. The `status` column:

| status | meaning |
|---|---|
| `success` | answered, translated (Mandarin) and scored |
| `blocked` | **the provider refused the content.** This is evidence for RQ1, so keep these rows. `error` holds the provider's message |
| `error_fatal` | bad key, no credit or wrong model. That model stopped; fix and re-run it with `--only` |
| `error` | transient failure (timeout, rate limit) after one retry. Re-run |
| `error_translation` / `error_evaluation` | answer received but the translator or judge failed |

Soft refusals (the model answers but dodges) come back as `success`; the judge captures them in
`political_avoidance`.

Every row is also appended to `runs/<date>_<model>.jsonl` as it completes, so nothing is lost if the
run crashes or the browser tab closes.

## The 30-event China run (1,200 rows)

```
python build_prompts.py                          # brief's 30 China events x 5 framings -> prompts_china30.tsv
python driver.py prompts_china30.tsv out.tsv --judge openai-gpt-nano --rpm 60
python merge_results.py prompts_china30.tsv results_china30.tsv out*.tsv runs/*.jsonl
python analyse.py results_china30.tsv            # restriction rates for RQ1 / RQ3 / RQ4
```

`merge_results.py` writes `todo_<model>.tsv` listing any prompts that failed; run `driver.py` on
that file and merge again to fill the gaps. Kimi allows 1 request at a time and 3 per minute on a
basic account, so run it as a single job with `--rpm 3 --judge-rpm 60`.

Models used: ChatGPT (gpt-5.4-nano), Gemini (gemini-3.1-flash-lite), DeepSeek (deepseek-v4-flash),
Kimi (kimi-k2.6, replacing Doubao, which wasn't accessible). Judge and translator: gpt-5.4-nano.

## Adding prompts

Add rows to `sample_data.tsv` (tab-separated, columns `Prompt ID`, `English`, `Mandarin`), or add
events to `build_prompts.py`.
