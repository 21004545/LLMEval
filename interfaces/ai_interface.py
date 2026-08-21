"""Abstract interface for AI API operations."""

from abc import ABC, abstractmethod
from typing import Optional


class AIInterface(ABC):
    """Abstract base class for AI API interactions."""

    @abstractmethod
    def send_prompt(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to the AI API and get a response.

        Args:
            prompt: The prompt to send to the AI.

        Returns:
            The AI's response, or None if an error occurs.
        """
        pass

    @abstractmethod
    def set_model(self, model: str) -> None:
        """
        Set the AI model to use.

        Args:
            model: The model identifier.
        """
        pass
