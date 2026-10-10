"""Kimi (Moonshot AI) API interface implementation."""

from typing import Optional
from interfaces.ai_interface import AIInterface
import os


class KimiInterface(AIInterface):
    """Interact with Kimi via the Moonshot AI international API (OpenAI-compatible)."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Kimi interface.

        Args:
            api_key: Moonshot API key (platform.moonshot.ai).
        """
        self.api_key = api_key or os.getenv('KIMI_API_KEY')
        self.model = 'kimi-k2.6'

        if not self.api_key:
            raise ValueError(
                "Kimi API key not provided and KIMI_API_KEY environment "
                "variable not set."
            )

        try:
            from openai import OpenAI
            # International endpoint; keys from platform.moonshot.cn (mainland China) use api.moonshot.cn instead
            self.client = OpenAI(
                base_url="https://api.moonshot.ai/v1",
                api_key=self.api_key
            )
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

    def send_prompt(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to Kimi and get a response.

        Args:
            prompt: The prompt to send.

        Returns:
            The response from Kimi. Raises on API errors.
        """
        # Errors propagate so the driver can record why a call failed
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        choice = response.choices[0]
        if choice.finish_reason == "content_filter":
            raise RuntimeError("Response blocked: finish_reason=content_filter")
        return choice.message.content

    def set_model(self, model: str) -> None:
        """
        Set the model to use.

        Args:
            model: The model name (e.g. 'kimi-k2.6').
        """
        self.model = model
