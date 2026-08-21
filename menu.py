"""Streamlit menu interface for the LLM prompt manager."""

import streamlit as st
import tempfile
import os
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from driver import Driver
from implementations.chatgpt_interface import ChatGPTInterface
from implementations.bias_evaluator import BiasEvaluator

# Load environment variables from .env file
load_dotenv()

# Rate limit defaults (80% of provider limits, in requests per minute)
RATE_LIMIT_DEFAULTS = {
    "openai-gpt-nano": 2800,           # 80% of 3500 req/min
    "gemini-3.1-flash-lite": 1200,     # 80% of 1500 req/min
    "deepseek-v4-flash": 480,          # 80% of 600 req/min
    "seed-2-0-lite-260428": 24000      # 80% of 30000 req/min (Doubao)
}


class StreamlitMenu:
    """Streamlit-based menu for the LLM prompt manager."""

    def __init__(self):
        """Initialize the Streamlit menu."""
        self.driver = Driver()
        self.initialize_session_state()

    def initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if 'prompts' not in st.session_state:
            st.session_state.prompts = []
        if 'results' not in st.session_state:
            st.session_state.results = []
        if 'processing_active' not in st.session_state:
            st.session_state.processing_active = False
        if 'processing_params' not in st.session_state:
            st.session_state.processing_params = None

    def run(self) -> None:
        """Run the Streamlit menu interface."""
        st.set_page_config(
            page_title="LLM Prompt Manager",
            page_icon="🤖",
            layout="wide"
        )

        st.title("🤖 LLM Prompt Manager")
        st.markdown(
            "Automatically send prompts to multiple LLMs and analyze responses."
        )

        # Auto-route to view results after processing completes
        if st.session_state.get('redirect_to_results', False):
            st.session_state.redirect_to_results = False
            menu_option = "View Results"
        else:
            # Sidebar navigation - disable during processing
            menu_option = st.sidebar.radio(
                "Select an option:",
                [
                    "Import Prompts",
                    "Process Prompts",
                    "View Results"
                ],
                disabled=st.session_state.processing_active
            )

        if menu_option == "Import Prompts":
            self.import_prompts_section()
        elif menu_option == "Process Prompts":
            self.process_prompts_section()
        elif menu_option == "View Results":
            self.view_results_section()

    def import_prompts_section(self) -> None:
        """Render the import prompts section."""
        st.header("Import Prompts")

        uploaded_file = st.file_uploader(
            "Upload a tab-delimited file with prompts:",
            type=["txt", "tsv", "csv"]
        )

        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                # Import prompts
                prompts = self.driver.import_prompts(temp_path)
                st.session_state.prompts = prompts

                st.success(f"Successfully imported {len(prompts)} prompts!")

            except Exception as e:
                st.error(f"Error importing file: {str(e)}")
        
        # Display loaded prompts if they exist
        if st.session_state.prompts:
            st.subheader("Loaded Prompts")
            st.dataframe(st.session_state.prompts)

    def process_prompts_section(self) -> None:
        """Render the process prompts section."""
        st.header("Process Prompts")

        if not st.session_state.prompts:
            st.warning("No prompts loaded. Please import prompts first.")
            return

        st.info(f"Loaded {len(st.session_state.prompts)} prompts")

        # Responder selection with checkboxes
        st.subheader("Select Responder LLMs")
        responder_options = {
            "OpenAI GPT-Nano": {
                "type": "openai-gpt-nano",
                "env_var": "OPENAI_API_KEY",
                "url": "https://platform.openai.com/api-keys"
            },
            "Gemini 3.1 Flash-Lite": {
                "type": "gemini-3.1-flash-lite",
                "env_var": "GEMINI_API_KEY",
                "url": "https://aistudio.google.com/app/apikey"
            },
            "DeepSeek V4 Flash": {
                "type": "deepseek-v4-flash",
                "env_var": "DEEPSEEK_API_KEY",
                "url": "https://platform.deepseek.com"
            },
            "Doubao Seed 2.0 Lite": {
                "type": "seed-2-0-lite-260428",
                "env_var": "DOUBAO_API_KEY",
                "url": "https://bytedance.com"
            }
        }

        selected_responders = []
        for responder_name in responder_options.keys():
            if st.checkbox(responder_name, value=False, key=f"checkbox_{responder_name}", disabled=st.session_state.processing_active):
                selected_responders.append(responder_name)

        if not selected_responders:
            st.warning("Please select at least one responder LLM.")
            return

        # API key and rate limit configuration for selected responders
        st.subheader("API Keys and Rate Limits")
        responder_api_keys = {}
        responder_rate_limits = {}
        
        for responder_name in selected_responders:
            config = responder_options[responder_name]
            responder_type = config["type"]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                api_key = st.text_input(
                    f"{responder_name} API Key:",
                    value=os.getenv(config["env_var"], ""),
                    type="password",
                    help=f"Get your key from {config['url']}",
                    key=f"api_key_{responder_type}",
                    disabled=st.session_state.processing_active
                )
                responder_api_keys[responder_type] = api_key
            
            with col2:
                default_rate_limit = RATE_LIMIT_DEFAULTS.get(responder_type, 100)
                rate_limit = st.number_input(
                    f"{responder_name} Rate (req/min):",
                    value=default_rate_limit,
                    min_value=1,
                    step=100,
                    key=f"rate_limit_{responder_type}",
                    disabled=st.session_state.processing_active
                )
                responder_rate_limits[responder_type] = rate_limit
        
        # Bias evaluation options
        st.divider()
        evaluate_bias = st.checkbox("Evaluate responses for political bias", value=False, disabled=st.session_state.processing_active)
        
        judge_client = None
        evaluator = None
        judge_api_key = None
        judge_model = "gpt-5.4-nano"
        judge_rate_limit = 2800  # Default: 80% of 3500
        
        if evaluate_bias:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                judge_api_key = st.text_input(
                    "Enter OpenAI API Key for judge LLM (translation + bias evaluation):",
                    value=os.getenv("OPENAI_API_KEY", ""),
                    type="password",
                    help="Used for translating Mandarin responses and evaluating bias",
                    disabled=st.session_state.processing_active
                )
            
            with col2:
                judge_rate_limit = st.number_input(
                    "Judge Rate (req/min):",
                    value=2800,
                    min_value=1,
                    step=100,
                    key="judge_rate_limit",
                    disabled=st.session_state.processing_active
                )

        if st.button("Process All Prompts (Concurrent)", key="process_button", disabled=st.session_state.processing_active):
            if not selected_responders:
                st.error("Please select at least one responder LLM.")
                return
            
            # Validate all API keys are provided
            for responder_type, api_key in responder_api_keys.items():
                if not api_key:
                    st.error(f"Please enter an API key for the selected responder.")
                    return
            
            if evaluate_bias and not judge_api_key:
                st.error("Please enter an API key for the judge LLM.")
                return

            # Store processing parameters in session state
            st.session_state.processing_params = {
                "selected_responders": selected_responders,
                "responder_api_keys": responder_api_keys,
                "responder_rate_limits": responder_rate_limits,
                "evaluate_bias": evaluate_bias,
                "judge_api_key": judge_api_key,
                "judge_model": judge_model,
                "judge_rate_limit": judge_rate_limit
            }
            st.session_state.processing_active = True
            st.rerun()
        
        # If processing was requested, run it now with locked UI
        if st.session_state.processing_active and st.session_state.processing_params:
            params = st.session_state.processing_params
            selected_responders = params["selected_responders"]
            responder_api_keys = params["responder_api_keys"]
            responder_rate_limits = params["responder_rate_limits"]
            evaluate_bias = params["evaluate_bias"]
            judge_api_key = params["judge_api_key"]
            judge_model = params["judge_model"]
            judge_rate_limit = params["judge_rate_limit"]
            
            # Create UI placeholders for real-time updates
            status_placeholder = st.empty()
            progress_area = st.empty()

            try:
                all_results = []
                total_responders = len(responder_api_keys)
                completed_responders = 0
                total_prompts = len(st.session_state.prompts)
                
                # Initialize progress tracking for each responder
                responder_progress = {responder_type: {"current": 0, "total": total_prompts} 
                                     for responder_type in responder_api_keys.keys()}
                
                # Create progress callback
                def update_progress(responder_type: str, current: int, total: int) -> None:
                    responder_progress[responder_type]["current"] = current
                
                # Create judge client and evaluator once (shared across all responders)
                judge_client = None
                evaluator = None
                if evaluate_bias:
                    try:
                        judge_client = ChatGPTInterface(api_key=judge_api_key)
                        judge_client.set_model(judge_model)
                        evaluator = BiasEvaluator(api_key=judge_api_key, model=judge_model)
                    except Exception as e:
                        st.error(f"Failed to initialize judge LLM: {str(e)}")
                        st.session_state.processing_active = False
                        st.session_state.processing_params = None
                        st.rerun()
                        return
                
                # Process responders concurrently using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=total_responders) as executor:
                    # Submit all responder tasks
                    future_to_responder = {}
                    for responder_type, api_key in responder_api_keys.items():
                        rate_limit = responder_rate_limits[responder_type]
                        
                        future = executor.submit(
                            self.driver.process_prompts,
                            st.session_state.prompts,
                            api_key,
                            responder_type=responder_type,
                            evaluate_bias=evaluate_bias,
                            judge_client=judge_client,
                            evaluator=evaluator,
                            rate_limit_rpm=rate_limit,
                            judge_rate_limit_rpm=judge_rate_limit,
                            progress_callback=update_progress
                        )
                        future_to_responder[future] = responder_type
                    
                    # Track completion while displaying progress
                    import time
                    dot_cycle = 0
                    last_update = time.time()
                    dots_sequence = ["", ".", "..", "..."]
                    
                    while completed_responders < total_responders:
                        # Update display every 0.33 seconds for smooth cycling
                        current_time = time.time()
                        if current_time - last_update > 0.33:
                            dots = dots_sequence[dot_cycle % 4]
                            dot_cycle += 1
                            
                            # Update status with cycling dots
                            status_placeholder.text(f"Processing{dots}")
                            
                            # Update progress display
                            with progress_area.container():
                                # Per-responder progress
                                for responder_type, progress_info in responder_progress.items():
                                    pct = (progress_info["current"] / progress_info["total"]) * 100
                                    st.progress(pct / 100, text=f"{responder_type}: {progress_info['current']}/{progress_info['total']}")
                                
                                # Overall progress (average)
                                avg_pct = sum((p["current"] / p["total"]) for p in responder_progress.values()) / len(responder_progress) * 100
                                st.progress(avg_pct / 100, text=f"Overall: {avg_pct:.0f}%")
                            
                            last_update = current_time
                        
                        # Check if any futures are done
                        try:
                            for future in as_completed(future_to_responder, timeout=0.1):
                                responder_type = future_to_responder[future]
                                results, fatal_error = future.result()
                                all_results.extend(results)
                                completed_responders += 1
                                responder_progress[responder_type]["current"] = total_prompts
                        except:
                            pass
                        
                        time.sleep(0.05)

                # Processing complete
                st.session_state.results = all_results
                st.session_state.processing_active = False
                st.session_state.processing_params = None
                st.session_state.redirect_to_results = True
                
                with progress_area.container():
                    for responder_type in responder_progress.keys():
                        st.progress(1.0, text=f"{responder_type}: {total_prompts}/{total_prompts}")
                    st.progress(1.0, text="Overall: 100%")
                
                status_placeholder.empty()
                st.success(f"✅ Processing complete! {len(st.session_state.results)} total results from {total_responders} responder(s)")
                st.rerun()

            except Exception as e:
                st.session_state.processing_active = False
                st.session_state.processing_params = None
                st.error(f"Error processing prompts: {str(e)}")
                st.rerun()

    def view_results_section(self) -> None:
        """Render the view results section with download option."""
        st.header("View Results")

        if not st.session_state.results:
            st.warning("No results available. Please process prompts first.")
            return

        st.info(f"Displaying {len(st.session_state.results)} results")
        st.dataframe(st.session_state.results, width='stretch')
        
        # Download section
        st.divider()
        st.subheader("Download Results")
        
        try:
            # Generate TSV content
            tsv_content = self.driver.exporter.export_results_to_string(st.session_state.results)
            
            st.download_button(
                label="Download Results as TSV",
                data=tsv_content,
                file_name="results.tsv",
                mime="text/tab-separated-values",
                key="download_tsv_button"
            )

        except Exception as e:
            st.error(f"Error preparing export: {str(e)}")

def main() -> None:
    """Entry point for the Streamlit menu."""
    menu = StreamlitMenu()
    menu.run()


if __name__ == "__main__":
    main()
