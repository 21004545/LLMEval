"""Tab-delimited file importer implementation."""

from typing import List, Dict
import csv
from interfaces.file_interface import FileImporter


class TabDelimitedImporter(FileImporter):
    """Import prompts from tab-delimited files."""

    def import_prompts(self, file_path: str) -> List[Dict[str, str]]:
        """
        Import prompts from a tab-delimited file.

        Args:
            file_path: Path to the tab-delimited file.

        Returns:
            List of dictionaries containing prompt data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is invalid.
        """
        prompts = []

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file, delimiter='\t')

                if reader.fieldnames is None:
                    raise ValueError("File is empty or has no header row.")

                for row_num, row in enumerate(reader, start=2):
                    if row:
                        prompts.append(row)

            return prompts

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")
