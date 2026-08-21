#!/usr/bin/env python3
"""Main entry point for the LLM Prompt Manager application."""

import sys
from menu import StreamlitMenu


def main() -> None:
    """Entry point for the application."""
    menu = StreamlitMenu()
    menu.run()


if __name__ == "__main__":
    main()
