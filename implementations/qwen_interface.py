"""Qwen (Alibaba) API interface implementation."""

from typing import Optional
from interfaces.ai_interface import AIInterface
import os


class QwenInterface(AIInterface):
    """Interact with Qwen via Alibaba Cloud Model Studio (OpenAI-compatible API)."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Qwen interface.

        Args:
            api_key: Alibaba Cloud Model Studio API key (format: sk-XXXXX).
        """
        self.api_key = api_key or os.getenv('QWEN_API_KEY')
        self.model = 'qwen-plus'

        if not self.api_key:
            raise ValueError(
                "Qwen API key not provided and QWEN_API_KEY environment "
                "variable not set."
            )

        try:
            from openai import OpenAI
            # International (Singapore) endpoint; keys from the mainland China console will not work here
            self.client = OpenAI(
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                api_key=self.api_key
            )
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

    def send_prompt(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to Qwen and get a response.

        Args:
            prompt: The prompt to send.

        Returns:
            The response from Qwen. Raises on API errors.
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
            model: The model name (e.g. 'qwen-plus', 'qwen-max').
        """
        self.model = model
