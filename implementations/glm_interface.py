"""GLM (Zhipu AI) API interface implementation."""

from typing import Optional
from interfaces.ai_interface import AIInterface
import os


class GLMInterface(AIInterface):
    """Interact with Zhipu GLM via the z.ai international API (OpenAI-compatible)."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize GLM interface.

        Args:
            api_key: z.ai API key.
        """
        self.api_key = api_key or os.getenv('GLM_API_KEY')
        self.model = 'glm-4.5-flash'  # free tier model

        if not self.api_key:
            raise ValueError(
                "GLM API key not provided and GLM_API_KEY environment "
                "variable not set."
            )

        try:
            from openai import OpenAI
            # International endpoint; keys from bigmodel.cn (mainland China) use open.bigmodel.cn instead
            self.client = OpenAI(
                base_url="https://api.z.ai/api/paas/v4",
                api_key=self.api_key
            )
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. Install with: pip install openai"
            )

    def send_prompt(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to GLM and get a response.

        Args:
            prompt: The prompt to send.

        Returns:
            The response from GLM. Raises on API errors.
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
            model: The model name (e.g. 'glm-4.5-flash', 'glm-4.6').
        """
        self.model = model
