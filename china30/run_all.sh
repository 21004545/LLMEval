#!/bin/bash
# Full China-30 run: 3 chunks per paid model in parallel, Gemini (free tier) as one slower job.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=.venv/Scripts/python
for m in openai-gpt-nano deepseek-v4-flash kimi-k2; do
  for i in 1 2 3; do
    $PY driver.py china30/chunks/part$i.tsv china30/${m}_part$i.tsv --only $m --judge openai-gpt-nano --rpm 60 > china30/logs/${m}_part$i.log 2>&1 &
  done
done
$PY driver.py prompts_china30.tsv china30/gemini-3.1-flash-lite_all.tsv --only gemini-3.1-flash-lite --judge openai-gpt-nano --rpm 12 > china30/logs/gemini-3.1-flash-lite_all.log 2>&1 &
wait
echo ALL DONE
