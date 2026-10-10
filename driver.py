"""Driver class that orchestrates the LLM prompt manager application."""

from typing import List, Dict, Optional, Callable
import json
import os
import time
from implementations.tab_delimited_importer import TabDelimitedImporter
from implementations.chatgpt_interface import ChatGPTInterface
from implementations.gemini_interface import GeminiInterface
from implementations.deepseek_interface import DeepSeekInterface
from implementations.doubao_interface import DoubaoInterface
from implementations.qwen_interface import QwenInterface
from implementations.glm_interface import GLMInterface
from implementations.kimi_interface import KimiInterface
from implementations.tab_delimited_exporter import TabDelimitedExporter
from implementations.bias_evaluator import BiasEvaluator
from interfaces.ai_interface import AIInterface

# responder type -> (interface class, model id, API key env var)
RESPONDERS = {
    "openai-gpt-nano": (ChatGPTInterface, "gpt-5.4-nano", "OPENAI_API_KEY"),
    "gemini-3.1-flash-lite": (GeminiInterface, "gemini-3.1-flash-lite", "GEMINI_API_KEY"),
    "deepseek-v4-flash": (DeepSeekInterface, "deepseek-v4-flash", "DEEPSEEK_API_KEY"),
    "seed-2-0-lite-260428": (DoubaoInterface, "seed-2-0-lite-260428", "DOUBAO_API_KEY"),
    "qwen-plus": (QwenInterface, "qwen-plus", "QWEN_API_KEY"),
    "glm-4.5-flash": (GLMInterface, "glm-4.5-flash", "GLM_API_KEY"),
    "kimi-k2": (KimiInterface, "kimi-k2.6", "KIMI_API_KEY"),
}

# Accepted input column names (case-insensitive) for each prompt language
PROMPT_COLUMNS = {
    "English": ("english_prompt", "english", "prompt_english"),
    "Mandarin": ("mandarin_prompt", "mandarin", "prompt_mandarin"),
}

SCORE_KEYS = (
    "factual_balance",
    "framing_bias",
    "attribution_of_responsibility",
    "political_avoidance",
    "loaded_language",
)

# Fatal: no point continuing with this responder (bad key, no credit, wrong model)
FATAL_KEYWORDS = (
    "invalid api key", "incorrect api key", "api key not valid", "unauthorized",
    "401", "402", "403", "forbidden", "model not found", "404", "invalid model",
    "authentication", "not authenticated", "insufficient",
)

# Blocked: the provider refused the content itself. This is RQ1 evidence, not a bug.
# DeepSeek: "Content Exists Risk"; Doubao/Ark: "...SensitiveContentDetected";
# Kimi: "considered high risk"; GLM: error code 1301 (unsafe/sensitive content); Qwen: "data_inspection_failed" / "inappropriate content"; OpenAI: finish_reason=content_filter; Gemini: SAFETY / BLOCKLIST / PROHIBITED_CONTENT.
BLOCK_KEYWORDS = (
    "content exists risk", "sensitivecontent", "sensitive content", "content_filter",
    "safety", "blocklist", "prohibited_content", "jailbreak",
    "data_inspection_failed", "inappropriate content", "'code': '1301'", "high risk",
)

TRANSLATION_PROMPT = "Translate the following Mandarin text to English. Provide only the translation, nothing else:\n\n"


class Driver:
    """Main driver class that manages the entire application workflow."""

    def __init__(self, backup_dir: str = "runs", retry_wait_seconds: float = 10):
        """Initialize the driver with implementations."""
        self.importer = TabDelimitedImporter()
        self.exporter = TabDelimitedExporter()
        self.ai_interface: Optional[AIInterface] = None
        self.backup_dir = backup_dir
        self.retry_wait_seconds = retry_wait_seconds

    def import_prompts(self, file_path: str) -> List[Dict[str, str]]:
        """
        Import prompts from a file.

        Args:
            file_path: Path to the file to import.

        Returns:
            List of prompt dictionaries.
        """
        return self.importer.import_prompts(file_path)

    @staticmethod
    def classify_error(error_message: str) -> str:
        """
        Classify an API error message.

        Returns:
            'blocked' if the provider refused the content, 'error_fatal' if
            processing should stop for this responder, otherwise 'error'
            (transient: rate limit, timeout, temporary service issue).
        """
        error_lower = error_message.lower().replace(" ", "")
        if any(k.replace(" ", "") in error_lower for k in BLOCK_KEYWORDS):
            return "blocked"
        error_lower = error_message.lower()
        if any(k in error_lower for k in FATAL_KEYWORDS):
            return "error_fatal"
        return "error"

    def _call(self, fn: Callable[[], Optional[str]], retries: int = 1) -> tuple:
        """
        Call an API function, retrying transient errors.

        Returns:
            Tuple of (text, error_message, status).
        """
        for attempt in range(retries + 1):
            try:
                text = fn()
                if text:
                    return text, None, "success"
                error, status = "Empty response", "error"
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                status = self.classify_error(error)
            if status != "error" or attempt == retries:
                return None, error, status
            print(f"Transient error, retrying in {self.retry_wait_seconds}s: {error}")
            time.sleep(self.retry_wait_seconds)

    @staticmethod
    def _prompt_text(prompt_dict: Dict[str, str], language: str) -> str:
        """Return the prompt for a language, accepting several column spellings."""
        for key, value in prompt_dict.items():
            if key and key.strip().lower() in PROMPT_COLUMNS[language]:
                return (value or "").strip()
        return ""

    def _rate_limit_sleep(self, rate_limit_rpm: int, call_times: List[float]) -> None:
        """
        Sleep if necessary to maintain the specified rate limit.

        Args:
            rate_limit_rpm: Maximum requests per minute allowed.
            call_times: List of timestamps of recent API calls (in seconds).
        """
        if rate_limit_rpm > 0:
            now = time.time()
            call_times[:] = [t for t in call_times if now - t < 60]
            if len(call_times) >= rate_limit_rpm:
                time.sleep(max(0, 60 - (now - call_times[0]) + 0.1))
        call_times.append(time.time())

    def _backup_row(self, responder_type: str, row: Dict) -> None:
        """Append a result row to a JSONL backup so a crash or closed tab loses nothing."""
        if not self.backup_dir:
            return
        os.makedirs(self.backup_dir, exist_ok=True)
        path = os.path.join(self.backup_dir, f"{time.strftime('%Y%m%d')}_{responder_type}_{os.getpid()}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def process_prompts(
        self,
        prompts: List[Dict[str, str]],
        responder_api_key: str,
        responder_type: str = "openai-gpt-nano",
        evaluate_bias: bool = False,
        judge_client: Optional[AIInterface] = None,
        evaluator: Optional[BiasEvaluator] = None,
        rate_limit_rpm: int = 100,
        judge_rate_limit_rpm: int = 2800,
        progress_callback: Optional[Callable] = None
    ) -> tuple:
        """
        Process prompts by sending them to an LLM responder and collecting responses.
        Processes each prompt completely (both languages) before moving to the next.

        Args:
            prompts: Prompt dictionaries with English and Mandarin columns
                (see PROMPT_COLUMNS for accepted names).
            responder_api_key: API key for the responder LLM.
            responder_type: A key of RESPONDERS.
            evaluate_bias: Kept for compatibility; evaluation runs when evaluator is given.
            judge_client: Shared ChatGPT client for translation (created externally).
            evaluator: Shared BiasEvaluator instance (created externally).
            rate_limit_rpm: Rate limit for responder (requests per minute).
            judge_rate_limit_rpm: Rate limit for judge/translator (requests per minute).
            progress_callback: Optional callback function called as progress_callback(responder_type, current_idx, total) after each prompt.

        Returns:
            Tuple of (results_list, fatal_error_occurred).
            fatal_error_occurred is True if processing should stop for this responder.
        """
        try:
            interface_class, model, _ = RESPONDERS[responder_type]
            ai_interface = interface_class(api_key=responder_api_key)
            ai_interface.set_model(model)
            self.ai_interface = ai_interface
        except Exception as e:
            print(f"FATAL ERROR: Failed to initialize responder {responder_type}: {e}")
            return [], True

        results = []
        fatal_error_occurred = False
        responder_call_times = []  # Track responder API call times for rate limiting
        judge_call_times = []      # Track judge API call times for rate limiting (translation + evaluation)

        for prompt_idx, prompt_dict in enumerate(prompts):
            english_prompt = self._prompt_text(prompt_dict, "English")

            for language in ("English", "Mandarin"):
                prompt = self._prompt_text(prompt_dict, language)
                if not prompt:
                    continue
                print(f"[{responder_type}] Prompt {prompt_idx + 1}/{len(prompts)} - {language}")

                self._rate_limit_sleep(rate_limit_rpm, responder_call_times)
                response, error, status = self._call(lambda: ai_interface.send_prompt(prompt))
                if error:
                    print(f"[{responder_type}] {status}: {error}")

                result = {
                    **prompt_dict,
                    'language': language,
                    'response': response,
                    'response_translated': None,
                    'status': status,
                    'error': error,
                    'responder': responder_type,
                    **{key: None for key in SCORE_KEYS},
                    'bias_justification': None
                }

                # Mandarin answers are translated to English before judging
                text_to_judge = response
                if response and language == "Mandarin" and judge_client:
                    self._rate_limit_sleep(judge_rate_limit_rpm, judge_call_times)
                    translated, translation_error, _ = self._call(
                        lambda: judge_client.send_prompt(TRANSLATION_PROMPT + response)
                    )
                    result['response_translated'] = translated
                    text_to_judge = translated
                    if translation_error:
                        result['status'] = 'error_translation'
                        result['error'] = translation_error

                if text_to_judge and evaluator:
                    self._rate_limit_sleep(judge_rate_limit_rpm, judge_call_times)
                    # The judge reads English, so give it the English question for both languages
                    evaluation = evaluator.evaluate(english_prompt or prompt, text_to_judge) or {}
                    result.update({key: evaluation.get(key) for key in SCORE_KEYS})
                    result['bias_justification'] = evaluation.get('justification')
                    if evaluation.get('factual_balance') is None:
                        result['status'] = 'error_evaluation'
                        result['error'] = evaluation.get('justification')

                results.append(result)
                self._backup_row(responder_type, result)

                if status == "error_fatal":
                    fatal_error_occurred = True
                    break

            if progress_callback:
                progress_callback(responder_type, prompt_idx + 1, len(prompts))

            if fatal_error_occurred:
                print(f"Stopping {responder_type} due to fatal error at prompt {prompt_idx + 1}")
                break

        return results, fatal_error_occurred

    def export_results(
        self,
        results: List[Dict[str, str]],
        file_path: str
    ) -> None:
        """
        Export results to a file.

        Args:
            results: List of result dictionaries.
            file_path: Path to export to.
        """
        self.exporter.export_results(file_path, results)

    def run(
        self,
        input_file: str,
        output_file: str,
        responders: Optional[List[str]] = None,
        evaluate_bias: bool = True,
        judge: str = "gemini-3.1-flash-lite",
        rpm: int = 8,
        judge_rpm: Optional[int] = None
    ) -> None:
        """
        Run the complete workflow without the UI: import, process every responder
        that has an API key in the environment / .env, and export.

        Args:
            input_file: Path to input file.
            output_file: Path to output file.
            responders: Responder types to run (default: all in RESPONDERS).
            evaluate_bias: Translate Mandarin answers and score bias with the judge.
            judge: Responder type used as translator + judge (default Gemini: free tier).
            rpm: Requests per minute for responder and judge each. The default (8)
                keeps Gemini's free tier happy; raise it on paid keys.
        """
        from dotenv import load_dotenv
        load_dotenv()

        prompts = self.import_prompts(input_file)
        print(f"Imported {len(prompts)} prompts from {input_file}")

        judge_client = evaluator = None
        if evaluate_bias:
            judge_class, judge_model, judge_env = RESPONDERS[judge]
            if not os.getenv(judge_env):
                raise SystemExit(f"{judge_env} is required for the {judge} judge (or pass --no-bias / --judge).")
            judge_client = judge_class(api_key=os.getenv(judge_env))
            judge_client.set_model(judge_model)
            evaluator = BiasEvaluator(judge_client)
            print(f"Judge/translator: {judge}")

        all_results = []
        for responder_type in responders or RESPONDERS:
            key = os.getenv(RESPONDERS[responder_type][2])
            if not key:
                print(f"Skipping {responder_type}: {RESPONDERS[responder_type][2]} not set")
                continue
            results, fatal = self.process_prompts(
                prompts, key, responder_type,
                evaluate_bias=evaluate_bias, judge_client=judge_client, evaluator=evaluator,
                rate_limit_rpm=rpm, judge_rate_limit_rpm=judge_rpm or rpm
            )
            all_results.extend(results)
            if fatal:
                print(f"WARNING: {responder_type} stopped early on a fatal error - check the 'error' column")

        if not all_results:
            raise SystemExit("No results produced - check API keys in .env")
        self.export_results(all_results, output_file)

        counts = {}
        for r in all_results:
            counts[(r['responder'], r['status'])] = counts.get((r['responder'], r['status']), 0) + 1
        print(f"\n{len(all_results)} rows written to {output_file}")
        for (responder, status), n in sorted(counts.items()):
            print(f"  {responder:24} {status:18} {n}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run all prompts against the LLMs without the Streamlit UI.")
    parser.add_argument("input", nargs="?", default="sample_data.tsv")
    parser.add_argument("output", nargs="?", default="results.tsv")
    parser.add_argument("--only", nargs="+", choices=list(RESPONDERS), help="Run only these responders")
    parser.add_argument("--no-bias", action="store_true", help="Skip translation and bias evaluation")
    parser.add_argument("--judge", default="gemini-3.1-flash-lite", choices=list(RESPONDERS),
                        help="Model used to translate and judge (default: Gemini, free tier)")
    parser.add_argument("--rpm", type=int, default=8, help="Requests/minute per model (default 8 for free tiers)")
    parser.add_argument("--judge-rpm", type=int, help="Requests/minute for the judge (default: same as --rpm)")
    args = parser.parse_args()
    Driver(retry_wait_seconds=30).run(args.input, args.output, responders=args.only,
                                      evaluate_bias=not args.no_bias, judge=args.judge, rpm=args.rpm,
                                      judge_rpm=args.judge_rpm)
