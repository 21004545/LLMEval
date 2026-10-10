"""Offline self-check for the driver: fake LLMs, no API keys or credit needed.

Run: python test_driver.py
"""

import tempfile
import driver
from driver import Driver


class FakeLLM:
    """Stands in for any responder. Behaviour is picked by the API key."""

    def __init__(self, api_key=None):
        self.key = api_key
        self.calls = 0

    def set_model(self, model):
        pass

    def send_prompt(self, prompt):
        self.calls += 1
        if self.key == "blocked" and "鸦片" in prompt:
            raise RuntimeError("Error code: 400 - {'error': {'message': 'Content Exists Risk'}}")
        if self.key == "badkey":
            raise RuntimeError("Error code: 401 - Incorrect API key provided")
        if self.key == "flaky" and self.calls == 1:
            raise RuntimeError("Error code: 503 - service temporarily unavailable")
        if prompt.startswith("Translate"):
            return "translated: " + prompt[-10:]
        return "answer to " + prompt


class FakeJudge:
    def evaluate(self, question, response):
        assert question.isascii(), "judge should get the English question"
        return {k: 2 for k in driver.SCORE_KEYS} | {"justification": "ok"}


for name in list(driver.RESPONDERS):
    driver.RESPONDERS[name] = (FakeLLM, "fake", "UNUSED")

# Same headers as sample_data.tsv
prompts = [
    {"Prompt ID": "1", "English": "What led up to The Opium Wars?", "Mandarin": "是什么导致了鸦片战争？"},
    {"Prompt ID": "2", "English": "Hello?", "Mandarin": "你好？"},
]
d = Driver(backup_dir=tempfile.mkdtemp(), retry_wait_seconds=0)
judge = FakeLLM("judge")

# 1. sample_data.tsv column names are read, both languages run, Mandarin is translated and judged
rows, fatal = d.process_prompts(prompts, "ok", "openai-gpt-nano", judge_client=judge, evaluator=FakeJudge())
assert not fatal and len(rows) == 4, rows
assert [r["status"] for r in rows] == ["success"] * 4
assert rows[1]["language"] == "Mandarin" and rows[1]["response_translated"].startswith("translated")
assert all(r["factual_balance"] == 2 for r in rows)

# 2. A content block is recorded as 'blocked' with the provider's message, and the run continues
rows, fatal = d.process_prompts(prompts, "blocked", "deepseek-v4-flash", judge_client=judge, evaluator=FakeJudge())
assert not fatal and len(rows) == 4
assert rows[1]["status"] == "blocked" and "Content Exists Risk" in rows[1]["error"]
assert rows[1]["factual_balance"] is None
assert rows[0]["status"] == rows[2]["status"] == "success"

# 3. A bad key stops that responder straight away with the real error message
rows, fatal = d.process_prompts(prompts, "badkey", "gemini-3.1-flash-lite")
assert fatal and len(rows) == 1 and rows[0]["status"] == "error_fatal" and "401" in rows[0]["error"]

# 4. A transient error is retried once
rows, fatal = d.process_prompts(prompts[:1], "flaky", "seed-2-0-lite-260428")
assert rows[0]["status"] == "success", rows[0]

# 5. Classifier covers each provider's block wording
for msg in ("Content Exists Risk", "InputTextSensitiveContentDetected",
            "Response blocked: finish_reason=content_filter",
            "Gemini returned no text: block_reason=BlockedReason.PROHIBITED_CONTENT, finish_reason=None"):
    assert Driver.classify_error(msg) == "blocked", msg
assert Driver.classify_error("Error code: 429 - Rate limit reached") == "error"

print("all driver checks passed")
