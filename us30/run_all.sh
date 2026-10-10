#!/bin/bash
# Full US-30 run. Paid models in 3 parallel chunks; Gemini (free tier) and Kimi (1 concurrent, 3 RPM) as single jobs.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=.venv/Scripts/python
for m in openai-gpt-nano deepseek-v4-flash; do
  for i in 1 2 3; do
    $PY driver.py us30/chunks/part$i.tsv us30/${m}_part$i.tsv --only $m --judge openai-gpt-nano --rpm 60 > us30/logs/${m}_part$i.log 2>&1 &
  done
done
$PY driver.py prompts_us30.tsv us30/gemini-3.1-flash-lite_all.tsv --only gemini-3.1-flash-lite --judge openai-gpt-nano --rpm 12 --judge-rpm 60 > us30/logs/gemini-3.1-flash-lite_all.log 2>&1 &
$PY driver.py prompts_us30.tsv us30/kimi-k2_all.tsv --only kimi-k2 --judge openai-gpt-nano --rpm 3 --judge-rpm 60 > us30/logs/kimi-k2_all.log 2>&1 &
wait
rm -f todo_*.tsv
$PY merge_results.py prompts_us30.tsv results_us30.tsv us30/*.tsv
echo ALL DONE
