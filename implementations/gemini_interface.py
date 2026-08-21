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
            The Gemini response, or None if an error occurs.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response.text

        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            return None

    def set_model(self, model: str) -> None:
        """
        Set the Gemini model to use.

        Args:
            model: The model identifier (e.g., 'gemini-1.5-flash', 'gemini-2.0-flash').
        """
        self.model = model
