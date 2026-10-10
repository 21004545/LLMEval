"""Google Gemini API interface implementation."""

from typing import Optional
from interfaces.ai_interface import AIInterface
import os


class GeminiInterface(AIInterface):
    """Interact with Google Gemini via Google AI API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini interface.

        Args:
            api_key: Google API key. If not provided, uses GOOGLE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.model = 'gemini-3.6-flash'

        if not self.api_key:
            raise ValueError(
                "Google API key not provided and GOOGLE_API_KEY environment "
                "variable not set."
            )

        try:
            import google.genai as genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "Google Generative AI library not installed. Install with: "
                "pip install google-genai"
            )

    def send_prompt(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to Gemini and get a response.

        Args:
            prompt: The prompt to send.

        Returns:
            The Gemini response. Raises on API errors or safety blocks.
        """
        # Errors propagate so the driver can record why a call failed
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        if not response.text:
            # Safety blocks come back as an empty response, not an exception
            block = response.prompt_feedback.block_reason if response.prompt_feedback else None
            finish = response.candidates[0].finish_reason if response.candidates else None
            raise RuntimeError(f"Gemini returned no text: block_reason={block}, finish_reason={finish}")
        return response.text

    def set_model(self, model: str) -> None:
        """
        Set the Gemini model to use.

        Args:
            model: The model identifier (e.g., 'gemini-1.5-flash', 'gemini-2.0-flash').
        """
        self.model = model
