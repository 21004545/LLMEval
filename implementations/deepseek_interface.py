"""DeepSeek API interface implementation."""

from typing import Optional
from interfaces.ai_interface import AIInterface
import os


class DeepSeekInterface(AIInterface):
    """Interact with DeepSeek via OpenAI-compatible API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize DeepSeek interface.

        Args:
            api_key: DeepSeek API key.
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.model = 'deepseek-v4-flash'

        if not self.api_key:
            raise ValueError(
                "DeepSeek API key not provided and DEEPSEEK_API_KEY environment "
                "variable not set."
            )

        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

    def send_prompt(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to DeepSeek and get a response.

        Args:
            prompt: The prompt to send.

        Returns:
            The response from DeepSeek, or None if an error occurs.
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
            print(f"Error getting response from DeepSeek: {str(e)}")
            return None

    def set_model(self, model: str) -> None:
        """
        Set the model to use.

        Args:
            model: The model name.
        """
        self.model = model
