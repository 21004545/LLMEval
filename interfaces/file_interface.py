"""Abstract interface for file import/export operations."""

from abc import ABC, abstractmethod
from typing import List, Dict


class FileImporter(ABC):
    """Abstract base class for file import operations."""

    @abstractmethod
    def import_prompts(self, file_path: str) -> List[Dict[str, str]]:
        """
        Import prompts from a file.

        Args:
            file_path: Path to the file to import from.

        Returns:
            List of dictionaries containing prompt data.
        """
        pass


class FileExporter(ABC):
    """Abstract base class for file export operations."""

    @abstractmethod
    def export_results(self, file_path: str, results: List[Dict[str, str]]) -> None:
        """
        Export results to a file.

        Args:
            file_path: Path to the file to export to.
            results: List of dictionaries containing result data.
        """
        pass
