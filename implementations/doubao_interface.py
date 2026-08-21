"""Doubao API interface implementation."""

from typing import Optional
from interfaces.ai_interface import AIInterface
import os


class DoubaoInterface(AIInterface):
    """Interact with Doubao via ByteDance Ark API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Doubao interface.

        Args:
            api_key: Doubao API key (format: ark-XXXXX).
        """
        self.api_key = api_key or os.getenv('DOUBAO_API_KEY')
        self.model = 'seed-2-0-lite-260428'  # Model endpoint ID

        if not self.api_key:
            raise ValueError(
                "Doubao API key not provided and DOUBAO_API_KEY environment "
                "variable not set."
            )

        try:
            from openai import OpenAI
            # Use BytePlus Ark API endpoint for ap-southeast-1 region
            self.client = OpenAI(
                base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
                api_key=self.api_key
            )
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

    def send_prompt(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to Doubao and get a response.

        Args:
            prompt: The prompt to send.

        Returns:
            The response from Doubao, or None if an error occurs.
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
            print(f"Error getting response from Doubao: {str(e)}")
            return None

    def set_model(self, model: str) -> None:
        """
        Set the model to use.

        Args:
            model: The model name (endpoint ID like 'seed-2-0-lite').
        """
        self.model = model
