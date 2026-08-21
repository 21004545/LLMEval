"""Driver class that orchestrates the LLM prompt manager application."""

from typing import List, Dict, Optional, Callable
import time
from implementations.tab_delimited_importer import TabDelimitedImporter
from implementations.chatgpt_interface import ChatGPTInterface
from implementations.gemini_interface import GeminiInterface
from implementations.deepseek_interface import DeepSeekInterface
from implementations.doubao_interface import DoubaoInterface
from implementations.tab_delimited_exporter import TabDelimitedExporter
from implementations.bias_evaluator import BiasEvaluator
from interfaces.ai_interface import AIInterface
from implementations.bias_evaluator import BiasEvaluator
from interfaces.ai_interface import AIInterface


class Driver:
    """Main driver class that manages the entire application workflow."""

    def __init__(self):
        """Initialize the driver with implementations."""
        self.importer = TabDelimitedImporter()
        self.exporter = TabDelimitedExporter()
        self.ai_interface: Optional[AIInterface] = None

    def import_prompts(self, file_path: str) -> List[Dict[str, str]]:
        """
        Import prompts from a file.

        Args:
            file_path: Path to the file to import.

        Returns:
            List of prompt dictionaries.
        """
        return self.importer.import_prompts(file_path)

    def _is_fatal_error(self, error_message: str) -> bool:
        """
        Determine if an error is fatal (should stop processing) or transient (should continue).
        
        Fatal errors: Invalid auth, wrong model, invalid key
        Transient errors: Rate limit, timeout, temporary service issues
        
        Args:
            error_message: The error message from the API.
            
        Returns:
            True if the error is fatal, False if transient.
        """
        fatal_keywords = [
            'invalid api key',
            'unauthorized',
            '401',
            '403',
            'forbidden',
            'model not found',
            '404',
            'invalid model',
            'authentication failed',
            'not authenticated'
        ]
        
        error_lower = error_message.lower()
        return any(keyword in error_lower for keyword in fatal_keywords)

    def _rate_limit_sleep(self, rate_limit_rpm: int, call_times: List[float]) -> None:
        """
        Sleep if necessary to maintain the specified rate limit.
        
        Args:
            rate_limit_rpm: Maximum requests per minute allowed.
            call_times: List of timestamps of recent API calls (in seconds).
        """
        if rate_limit_rpm <= 0:
            return
        
        # Current time
        now = time.time()
        
        # Remove calls older than 60 seconds
        call_times[:] = [t for t in call_times if now - t < 60]
        
        # If we've already made the maximum number of calls in the last minute, sleep
        if len(call_times) >= rate_limit_rpm:
            # Sleep until the oldest call is more than 60 seconds old
            sleep_time = 60 - (now - call_times[0]) + 0.1  # Add 0.1s buffer
            if sleep_time > 0:
                time.sleep(sleep_time)
                # Clear old call times after sleeping
                now = time.time()
                call_times[:] = [t for t in call_times if now - t < 60]

    def _translate_to_english(self, text: str, judge_client: AIInterface, judge_call_times: List[float], judge_rate_limit_rpm: int) -> tuple:
        """
        Translate Mandarin text to English using the shared judge LLM client.

        Args:
            text: Text to translate (in Mandarin).
            judge_client: Shared ChatGPT client for translation.
            judge_call_times: List of judge API call timestamps for rate limiting.
            judge_rate_limit_rpm: Rate limit for judge calls (requests per minute).

        Returns:
            Tuple of (translated_text, error_message). error_message is None if successful.
        """
        try:
            if not judge_client:
                return None, "Translation failed: no judge client provided"
            
            # Apply rate limiting for judge
            self._rate_limit_sleep(judge_rate_limit_rpm, judge_call_times)
            judge_call_times.append(time.time())
            
            translation_prompt = f"Translate the following Mandarin text to English. Provide only the translation, nothing else:\n\n{text}"
            translated = judge_client.send_prompt(translation_prompt)
            
            if translated:
                return translated, None
            else:
                return None, "Translation failed: no response from translator"
        except Exception as e:
            error_msg = f"Translation error: {str(e)}"
            return None, error_msg

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
            prompts: List of prompt dictionaries with 'english_prompt' and 'mandarin_prompt' columns.
            responder_api_key: API key for the responder LLM.
            responder_type: Type of responder ('openai-gpt-nano', 'gemini-3.1-flash-lite', etc).
            evaluate_bias: Whether to evaluate responses for political bias.
            judge_client: Shared ChatGPT client for translation (created externally).
            evaluator: Shared BiasEvaluator instance (created externally).
            rate_limit_rpm: Rate limit for responder (requests per minute).
            judge_rate_limit_rpm: Rate limit for judge/translator (requests per minute).
            progress_callback: Optional callback function called as progress_callback(responder_type, current_idx, total) after each prompt.

        Returns:
            Tuple of (results_list, fatal_error_occurred).
            fatal_error_occurred is True if processing should stop for this responder.
        """
        # Initialize AI interface based on responder type
        try:
            if responder_type == "openai-gpt-nano":
                self.ai_interface = ChatGPTInterface(api_key=responder_api_key)
                self.ai_interface.set_model("gpt-5.4-nano")
            elif responder_type == "gemini-3.1-flash-lite":
                self.ai_interface = GeminiInterface(api_key=responder_api_key)
                self.ai_interface.set_model("gemini-3.1-flash-lite")
            elif responder_type == "deepseek-v4-flash":
                self.ai_interface = DeepSeekInterface(api_key=responder_api_key)
                self.ai_interface.set_model("deepseek-v4-flash")
            elif responder_type == "seed-2-0-lite-260428":
                self.ai_interface = DoubaoInterface(api_key=responder_api_key)
                self.ai_interface.set_model("seed-2-0-lite-260428")
            else:
                raise ValueError(f"Unknown responder type: {responder_type}")
        except Exception as e:
            error_msg = f"Failed to initialize responder: {str(e)}"
            print(f"FATAL ERROR: {error_msg}")
            return [], True

        results = []
        fatal_error_occurred = False
        responder_call_times = []  # Track responder API call times for rate limiting
        judge_call_times = []      # Track judge API call times for rate limiting (translation + evaluation)

        # Process each prompt completely (English + Mandarin) before moving to next
        for prompt_idx, prompt_dict in enumerate(prompts):
            # Get English and Mandarin prompts
            english_prompt = ''
            mandarin_prompt = ''
            
            for key, value in prompt_dict.items():
                key_lower = key.lower()
                if key_lower == 'english_prompt':
                    english_prompt = value
                elif key_lower == 'mandarin_prompt':
                    mandarin_prompt = value

            # ===== ENGLISH PROMPT =====
            if english_prompt:
                print(f"[{responder_type}] Processing prompt {prompt_idx + 1}/{len(prompts)} - English")
                
                # Apply rate limiting for responder
                self._rate_limit_sleep(rate_limit_rpm, responder_call_times)
                responder_call_times.append(time.time())
                
                print(f"[{responder_type}] Sending English prompt to responder...")
                english_response = self.ai_interface.send_prompt(english_prompt)
                english_error = None
                english_status = "success"
                
                if not english_response:
                    english_status = "error_response"
                    english_error = "No response received (check API key and available credits)"
                    print(f"[{responder_type}] Error: No response for English prompt")
                    
                    # Check if this is a fatal error
                    if self._is_fatal_error(english_error):
                        fatal_error_occurred = True
                        print(f"FATAL ERROR at row {prompt_idx}: {english_error}")
                
                # Create English result
                english_result = {
                    **prompt_dict,
                    'language': 'English',
                    'response': english_response,
                    'response_translated': None,
                    'status': english_status,
                    'error': english_error,
                    'responder': responder_type,
                    'factual_balance': None,
                    'framing_bias': None,
                    'attribution_of_responsibility': None,
                    'political_avoidance': None,
                    'loaded_language': None,
                    'bias_justification': None
                }
                
                # Evaluate English response if successful and evaluator is available
                if english_response and evaluator:
                    try:
                        print(f"[{responder_type}] Evaluating English response for bias...")
                        # Apply rate limiting for judge
                        self._rate_limit_sleep(judge_rate_limit_rpm, judge_call_times)
                        judge_call_times.append(time.time())
                        
                        evaluation = evaluator.evaluate(english_prompt, english_response)
                        if evaluation:
                            english_result.update({
                                'factual_balance': evaluation.get('factual_balance'),
                                'framing_bias': evaluation.get('framing_bias'),
                                'attribution_of_responsibility': evaluation.get('attribution_of_responsibility'),
                                'political_avoidance': evaluation.get('political_avoidance'),
                                'loaded_language': evaluation.get('loaded_language'),
                                'bias_justification': evaluation.get('justification')
                            })
                            print(f"[{responder_type}] English evaluation complete")
                    except Exception as e:
                        english_result['status'] = 'error_evaluation'
                        english_result['error'] = f"Evaluation failed: {str(e)}"
                        print(f"[{responder_type}] Error evaluating English response: {str(e)}")
                
                results.append(english_result)

            # ===== MANDARIN PROMPT =====
            if mandarin_prompt:
                print(f"[{responder_type}] Processing prompt {prompt_idx + 1}/{len(prompts)} - Mandarin")
                
                # Apply rate limiting for responder
                self._rate_limit_sleep(rate_limit_rpm, responder_call_times)
                responder_call_times.append(time.time())
                
                print(f"[{responder_type}] Sending Mandarin prompt to responder...")
                mandarin_response = self.ai_interface.send_prompt(mandarin_prompt)
                mandarin_error = None
                mandarin_status = "success"
                translated_response = None
                
                if not mandarin_response:
                    mandarin_status = "error_response"
                    mandarin_error = "No response received (check API key and available credits)"
                    
                    # Check if this is a fatal error
                    if self._is_fatal_error(mandarin_error):
                        fatal_error_occurred = True
                        print(f"FATAL ERROR at row {prompt_idx}: {mandarin_error}")
                
                # Translate Mandarin response if successful
                if mandarin_response and judge_client:
                    print(f"[{responder_type}] Translating Mandarin response to English...")
                    translated_response, translation_error = self._translate_to_english(
                        mandarin_response, judge_client, judge_call_times, judge_rate_limit_rpm
                    )
                    
                    if translation_error:
                        mandarin_status = "error_translation"
                        mandarin_error = translation_error
                        print(f"[{responder_type}] Translation error: {translation_error}")
                    else:
                        print(f"[{responder_type}] Translation complete")
                
                # Create Mandarin result
                mandarin_result = {
                    **prompt_dict,
                    'language': 'Mandarin',
                    'response': mandarin_response,
                    'response_translated': translated_response,
                    'status': mandarin_status,
                    'error': mandarin_error,
                    'responder': responder_type,
                    'factual_balance': None,
                    'framing_bias': None,
                    'attribution_of_responsibility': None,
                    'political_avoidance': None,
                    'loaded_language': None,
                    'bias_justification': None
                }
                
                # Evaluate translated response if translation was successful and evaluator is available
                if translated_response and evaluator:
                    try:
                        print(f"[{responder_type}] Evaluating Mandarin response for bias...")
                        # Apply rate limiting for judge
                        self._rate_limit_sleep(judge_rate_limit_rpm, judge_call_times)
                        judge_call_times.append(time.time())
                        
                        evaluation = evaluator.evaluate(mandarin_prompt, translated_response)
                        if evaluation:
                            mandarin_result.update({
                                'factual_balance': evaluation.get('factual_balance'),
                                'framing_bias': evaluation.get('framing_bias'),
                                'attribution_of_responsibility': evaluation.get('attribution_of_responsibility'),
                                'political_avoidance': evaluation.get('political_avoidance'),
                                'loaded_language': evaluation.get('loaded_language'),
                                'bias_justification': evaluation.get('justification')
                            })
                            print(f"[{responder_type}] Mandarin evaluation complete")
                    except Exception as e:
                        mandarin_result['status'] = 'error_evaluation'
                        mandarin_result['error'] = f"Evaluation failed: {str(e)}"
                        print(f"[{responder_type}] Error evaluating Mandarin response: {str(e)}")
                
                results.append(mandarin_result)
            
            # Report progress after each prompt is fully processed
            if progress_callback:
                progress_callback(responder_type, prompt_idx + 1, len(prompts))
            
            # Stop processing if we hit a fatal error
            if fatal_error_occurred:
                print(f"Stopping processing due to fatal error at row {prompt_idx}")
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
        api_key: str,
        model: str = "gpt-3.5-turbo"
    ) -> None:
        """
        Run the complete workflow: import, process, and export.

        Args:
            input_file: Path to input file.
            output_file: Path to output file.
            api_key: OpenAI API key.
            model: ChatGPT model to use.
        """
        print(f"Starting LLM Prompt Manager...")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Model: {model}")

        # Import prompts
        print("\n1. Importing prompts...")
        prompts = self.import_prompts(input_file)
        print(f"   Imported {len(prompts)} prompts")

        # Process prompts
        print("\n2. Processing prompts...")
        results, fatal_error = self.process_prompts(prompts, api_key, model)
        print(f"   Processed {len(results)} prompts")

        # Export results
        print("\n3. Exporting results...")
        self.export_results(results, output_file)

        print("\nWorkflow completed successfully!")
