"""ChatGPT API interface implementation."""

from typing import Optional
from interfaces.ai_interface import AIInterface
import os


class ChatGPTInterface(AIInterface):
    """Interact with ChatGPT via OpenAI API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize ChatGPT interface.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = 'gpt-5.4-nano'

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided and OPENAI_API_KEY environment "
                "variable not set."
            )

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

    def send_prompt(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to ChatGPT and get a response.

        Args:
            prompt: The prompt to send.

        Returns:
            The ChatGPT response, or None if an error occurs.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error calling ChatGPT API: {str(e)}")
            return None

    def set_model(self, model: str) -> None:
        """
        Set the ChatGPT model to use.

        Args:
            model: The model identifier (e.g., 'gpt-4', 'gpt-3.5-turbo').
        """
        self.model = model